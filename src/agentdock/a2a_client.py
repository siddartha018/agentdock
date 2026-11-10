"""Call another platform agent over A2A, on behalf of the current caller.

Over HTTP (``AGENTDOCK_A2A_URL`` set) the request carries the service key plus
``x-on-behalf-of``, so the callee runs with the *person's* rights, never the
service's. Without a URL it loops back in-process with the same identity,
which is what evals and unit tests use.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import httpx

from agentdock.identity import Caller

_assistant_ids: dict[str, str] = {}


def _texts(parts: list[dict[str, Any]] | None) -> list[str]:
    return [p["text"] for p in parts or [] if p.get("kind") == "text" and p.get("text")]


def extract_text(result: dict[str, Any]) -> str:
    """Pull the reply text out of an A2A Task or Message result."""
    texts: list[str] = []
    for artifact in result.get("artifacts") or []:
        texts += _texts(artifact.get("parts"))
    if not texts:
        texts = _texts(((result.get("status") or {}).get("message") or {}).get("parts"))
    if not texts:
        texts = _texts(result.get("parts"))
    return "\n".join(texts).strip()


async def _assistant_id(client: httpx.AsyncClient, base: str, agent: str, headers: dict[str, str]) -> str:
    if agent not in _assistant_ids:
        r = await client.post(f"{base}/assistants/search", json={"graph_id": agent, "limit": 1}, headers=headers)
        r.raise_for_status()
        found = r.json()
        if not found:
            raise LookupError(f"no agent named {agent!r} on {base}")
        _assistant_ids[agent] = found[0]["assistant_id"]
    return _assistant_ids[agent]


async def ask(agent: str, text: str, caller: Caller) -> str:
    base = os.environ.get("AGENTDOCK_A2A_URL", "").rstrip("/")
    if not base:
        return await _loopback(agent, text, caller)
    headers = {"x-api-key": os.environ["AGENTDOCK_SERVICE_KEY"], "x-on-behalf-of": caller.identity}
    async with httpx.AsyncClient(timeout=60) as client:
        aid = await _assistant_id(client, base, agent, headers)
        body = {
            "jsonrpc": "2.0",
            "id": uuid.uuid4().hex,
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "messageId": uuid.uuid4().hex,
                    "parts": [{"kind": "text", "text": text}],
                }
            },
        }
        r = await client.post(f"{base}/a2a/{aid}", json=body, headers=headers)
        r.raise_for_status()
        payload = r.json()
    if "error" in payload:
        raise RuntimeError(f"A2A error from {agent}: {payload['error'].get('message')}")
    return extract_text(payload["result"]) or "(no reply)"


async def _loopback(agent: str, text: str, caller: Caller) -> str:
    from langchain_core.messages import HumanMessage

    from agentdock.entry import AGENTS

    if agent not in AGENTS:
        raise LookupError(f"no agent named {agent!r}")
    user = {**caller.as_auth_dict(), "via": "svc-concierge"}
    out = await AGENTS[agent].ainvoke(
        {"messages": [HumanMessage(text)]}, {"configurable": {"langgraph_auth_user": user}}
    )
    from agentdock.messages import text_of

    return text_of(out["messages"][-1])
