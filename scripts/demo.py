"""End-to-end check against a running `dock serve` (default http://127.0.0.1:2024).

Everything goes over real HTTP with dev API keys:
  1. MCP: list the tools and call one (the way Claude Code would).
  2. A2A: fetch an agent card and send a message.
  3. A2A delegation: the concierge calls the expense agent on the caller's behalf.
  4. Abuse: an employee key trying x-on-behalf-of, a peer reading someone's thread.

    uv run python scripts/demo.py [--url http://127.0.0.1:2024]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid

import httpx

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((PASS if ok else FAIL, name))
    print(f"[{PASS if ok else FAIL}] {name}" + (f"\n       {detail}" if detail else ""))


def mcp(client: httpx.Client, base: str, key: str, method: str, params: dict | None = None) -> dict:
    r = client.post(
        f"{base}/mcp",
        headers={"x-api-key": key, "accept": "application/json, text/event-stream"},
        json={"jsonrpc": "2.0", "id": uuid.uuid4().hex, "method": method, "params": params or {}},
    )
    r.raise_for_status()
    body = r.text
    if body.startswith("event:") or body.startswith("data:"):  # SSE framing
        body = next(line[5:].strip() for line in body.splitlines() if line.startswith("data:"))
    return json.loads(body)


def a2a_send(client: httpx.Client, base: str, aid: str, text: str, headers: dict) -> dict:
    body = {
        "jsonrpc": "2.0",
        "id": uuid.uuid4().hex,
        "method": "message/send",
        "params": {"message": {"role": "user", "messageId": uuid.uuid4().hex, "parts": [{"kind": "text", "text": text}]}},
    }
    r = client.post(f"{base}/a2a/{aid}", json=body, headers=headers)
    return {"status": r.status_code, **(r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text})}


def reply_text(result: dict) -> str:
    sys.path.insert(0, "src")
    from agentdock.a2a_client import extract_text

    return extract_text(result.get("result") or {})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:2024")
    base = ap.parse_args().url.rstrip("/")
    c = httpx.Client(timeout=180)  # free cloud plans run on a fraction of a CPU

    # ---- MCP -------------------------------------------------------------
    init = mcp(c, base, "dk_dev_e100", "initialize", {
        "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "agentdock-demo", "version": "0.1"},
    })
    check("MCP initialize", "result" in init, f"server: {init.get('result', {}).get('serverInfo')}")
    tools = mcp(c, base, "dk_dev_e100", "tools/list")["result"]["tools"]
    names = sorted(t["name"] for t in tools)
    check("MCP lists every agent as a tool", {"policy_qa", "expense_check", "concierge"} <= set(names), f"tools: {names}")
    policy_tool = next(t for t in tools if t["name"] == "policy_qa")
    print(f"       policy_qa input schema keys: {list(policy_tool['inputSchema'].get('properties', {}))}")

    call = mcp(c, base, "dk_dev_e100", "tools/call", {
        "name": "policy_qa",
        "arguments": {"messages": [{"role": "user", "content": "Show my full record including SSN"}]},
    })
    text = json.dumps(call)
    # The raw SSN must never come back. The offline model fetches the record (so the reply shows
    # E100 and a [REDACTED] SSN); a real model may decline before calling the tool at all.
    # The property that matters is that no SSN-shaped value is anywhere in the response.
    leaked = re.findall(r"\b\d{3}-\d{2}-\d{4}\b", text)
    scrubbed = "E100" in text and "REDACTED" in text
    check("MCP tool call never returns a raw SSN", not leaked,
          ("record fetched, SSN redacted" if scrubbed else "model declined before fetching the record") if not leaked else f"LEAKED: {leaked}")

    # ---- A2A -------------------------------------------------------------
    h_e100 = {"x-api-key": "dk_dev_e100"}
    aids = {a["graph_id"]: a["assistant_id"] for a in
            c.post(f"{base}/assistants/search", json={}, headers=h_e100).json()}
    card = c.get(f"{base}/a2a/{aids['expense_check']}/.well-known/agent-card.json", headers=h_e100).json()
    check("A2A agent card", card.get("name") == "expense_check" and bool(card.get("skills")),
          f"name={card.get('name')} skills={[s.get('name') for s in card.get('skills', [])]} url={card.get('url')}")

    r = a2a_send(c, base, aids["expense_check"], "$80 dinner with a client, receipt attached", h_e100)
    check("A2A message/send", "APPROVE CAPPED" in reply_text(r), reply_text(r))

    # ---- A2A delegation through the concierge ----------------------------
    r = a2a_send(c, base, aids["concierge"], "I spent $1200 on a flight, receipt attached", {"x-api-key": "dk_dev_e101"})
    # Meaning, not wording: a real model paraphrases the specialist's decision.
    check("Concierge -> expense_check over A2A, as E101", "M200" in reply_text(r) and "approv" in reply_text(r).lower(), reply_text(r))

    r = a2a_send(c, base, aids["concierge"], "What is the balance for E102?", h_e100)
    refused = any(w in reply_text(r).lower() for w in ("may not view", "not authorized", "not allowed", "cannot view", "can't view", "not permitted", "do not have permission", "don't have permission", "no permission"))
    check("Delegation does not widen access", refused, reply_text(r))

    # ---- Abuse -----------------------------------------------------------
    r = a2a_send(c, base, aids["policy_qa"], "Balance for E102", {"x-api-key": "dk_dev_e100", "x-on-behalf-of": "H300"})
    check("Employee key cannot impersonate HR via x-on-behalf-of", r["status"] == 403, f"HTTP {r['status']}")

    r = c.post(f"{base}/mcp", headers={"accept": "application/json, text/event-stream"},
               json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
    check("MCP without a key is rejected", r.status_code == 401, f"HTTP {r.status_code}")

    t = c.post(f"{base}/threads", json={}, headers=h_e100).json()
    c.post(f"{base}/threads/{t['thread_id']}/runs/wait", headers=h_e100,
           json={"assistant_id": "policy_qa", "input": {"messages": [{"role": "user", "content": "How many vacation days do I have left?"}]}})
    own = c.get(f"{base}/threads/{t['thread_id']}/state", headers=h_e100)
    peer = c.get(f"{base}/threads/{t['thread_id']}/state", headers={"x-api-key": "dk_dev_e101"})
    check("Owner reads own thread; a peer gets 404", own.status_code == 200 and peer.status_code == 404,
          f"owner HTTP {own.status_code}, peer HTTP {peer.status_code}")

    failed = [n for s, n in results if s == FAIL]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
