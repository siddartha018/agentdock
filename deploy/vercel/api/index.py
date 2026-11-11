"""agentdock, serverless edition (Vercel).

The Agent Server is a long-running process with background workers, which a
serverless function can't host. This app runs the *same* platform code per
request (auth.resolve -> platform wrapper -> audit) and exposes small MCP and A2A
endpoints. Live requests use a real model (AGENTDOCK_MODEL, e.g. Gemini) with the
offline policy as a fallback when the free tier is rate-limited; the eval gate
always uses the offline model so its result is reproducible. The full Agent
Server version is `dock serve` / the Docker stack.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("AGENTDOCK_AUDIT_LOG", "/tmp/agentdock-audit.jsonl")
os.environ.setdefault("AGENTDOCK_MODEL", "offline")
os.environ.setdefault("AGENTDOCK_FALLBACK", "offline")
os.environ.setdefault("AGENTDOCK_PII_ENGINE", "patterns")

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langgraph_sdk import Auth  # noqa: E402

from agentdock import audit  # noqa: E402
from agentdock.auth import resolve  # noqa: E402
from agentdock import entry  # noqa: E402
from agentdock.evals import gate, load_baseline, run_agent  # noqa: E402
from agentdock.identity import directory  # noqa: E402
from agentdock.messages import text_of  # noqa: E402
from agentdock.models import model_used  # noqa: E402
from agentdock.platform import wrap  # noqa: E402
from agentdock.registry import discover, import_agent_module  # noqa: E402

app = FastAPI(title="agentdock serverless", docs_url=None, redoc_url=None, openapi_url=None)
MANIFESTS = {m.name: m for m in discover()}
LIVE = dict(entry.AGENTS)  # built with AGENTDOCK_MODEL (the real model)
AGENTS = LIVE
CONFIGURED = os.environ["AGENTDOCK_MODEL"]


def _offline_agents() -> dict[str, Any]:
    prev = os.environ["AGENTDOCK_MODEL"]
    os.environ["AGENTDOCK_MODEL"] = "offline"  # chat_model() reads this at build time
    try:
        return {m.name: wrap(m, import_agent_module(m).build()) for m in MANIFESTS.values()}
    finally:
        os.environ["AGENTDOCK_MODEL"] = prev


OFFLINE = LIVE if CONFIGURED == "offline" else _offline_agents()
_gate_lock = asyncio.Lock()
MAX_INPUT = 600
SAMPLES = {
    "policy_qa": [
        "How many vacation days do I have left?",
        "Show my full record including SSN and bank details",
        "What is the balance for E101?",
        "Can I carry over vacation? My name is Asha Rao, DOB: 04/12/1990",
    ],
    "expense_check": [
        "$80 dinner with a client, receipt attached",
        "Flight to Berlin $1,450 with receipt",
        "$30 of wine at the team dinner",
    ],
    "concierge": [
        "I spent $1200 on a flight, receipt attached",
        "What is the balance for E102?",
        "How many sick days do we get per year?",
    ],
}

# --- per-IP limits: the demo is public and runs on a free plan ---------------
_hits: dict[str, deque[float]] = defaultdict(deque)
# Live runs share one free model quota, so they're limited harder than the offline gate.
LIMITS = {"run": (8, 60.0), "gate": (6, 60.0), "mcp": (20, 60.0)}


def _limited(request: Request, kind: str) -> bool:
    ip = (request.headers.get("x-forwarded-for") or request.client.host if request.client else "?").split(",")[0]
    n, window = LIMITS[kind]
    q = _hits[f"{kind}:{ip}"]
    now = time.monotonic()
    while q and now - q[0] > window:
        q.popleft()
    if len(q) >= n:
        return True
    q.append(now)
    return False


def _caller(request: Request) -> dict[str, Any]:
    return resolve({k.lower(): v for k, v in request.headers.items()})


def _deny(exc: Auth.exceptions.HTTPException) -> JSONResponse:
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


def _msg(m: Any) -> dict[str, Any]:
    kind = "ai" if isinstance(m, AIMessage) else "tool" if isinstance(m, ToolMessage) else "human"
    out: dict[str, Any] = {"type": kind, "content": text_of(m)}
    if isinstance(m, AIMessage) and m.tool_calls:
        out["tool_calls"] = [{"name": tc["name"], "args": tc["args"]} for tc in m.tool_calls]
    if isinstance(m, ToolMessage):
        out["name"] = m.name
    return out


async def run_events(agent: str, message: str, user: dict[str, Any]):
    """Yield ("delta", text) as redacted chunks arrive, then ("done", payload)."""
    started = time.perf_counter()
    mark = len(audit.read())
    deltas: list[str] = []
    final: dict[str, Any] | None = None
    config = {"configurable": {"langgraph_auth_user": user, "thread_id": f"web-{uuid.uuid4().hex[:12]}"}}
    async for mode, data in AGENTS[agent].astream(
        {"messages": [HumanMessage(message)]}, config, stream_mode=["custom", "values"]
    ):
        if mode == "custom":
            deltas.append(data["delta"])
            yield "delta", data["delta"]
        else:
            final = data
    records = audit.read()[mark:]
    msgs = (final or {}).get("messages", [])
    ai = [m for m in msgs if isinstance(m, AIMessage)]
    yield "done", {
        "model": model_used(ai[-1]) if ai else "none",
        "configured_model": CONFIGURED,
        "agent": agent,
        "caller": {k: user.get(k) for k in ("identity", "role", "team", "via")},
        "answer": text_of(msgs[-1]) if msgs else "",
        "deltas": deltas,
        "messages": [_msg(m) for m in msgs],
        "audit": [r for r in records if r.get("agent") == agent][-1:] or records[-1:],
        "sub_runs": [r for r in records if r.get("agent") != agent],
        "ms": round((time.perf_counter() - started) * 1000, 1),
    }


async def run_once(agent: str, message: str, user: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    async for kind, data in run_events(agent, message, user):
        if kind == "done":
            payload = data
    return payload


def _ndjson_stream(agent: str, message: str, user: dict[str, Any]) -> StreamingResponse:
    """Newline-delimited JSON: start, then each redacted delta as it is produced, then done."""

    async def lines():
        caller = {k: user.get(k) for k in ("identity", "role", "team", "via")}
        yield json.dumps({"type": "start", "caller": caller, "agent": agent}) + "\n"
        try:
            async for kind, data in run_events(agent, message, user):
                if kind == "delta":
                    yield json.dumps({"type": "delta", "text": data}) + "\n"
                else:
                    yield json.dumps({"type": "done", **data}) + "\n"
        except Exception as exc:  # report, don't hang the reader
            yield json.dumps({"type": "error", "error": f"{type(exc).__name__}: {exc}"[:300]}) + "\n"

    return StreamingResponse(lines(), media_type="application/x-ndjson",
                             headers={"cache-control": "no-cache", "x-accel-buffering": "no"})


@app.get("/api/meta")
async def meta() -> dict[str, Any]:
    people = [
        {"id": uid, "name": r["name"], "role": r["role"], "team": r.get("team"), "manager": r.get("manager"),
         "key": f"dk_dev_{uid.lower()}"}
        for uid, r in directory().users.items() if r["role"] != "service"
    ]
    agents = [
        {"name": m.name, "description": m.description, "kind": m.kind, "roles": m.allowed_roles,
         "stream": m.stream, "owner": m.owner_team, "samples": SAMPLES.get(m.name, [])}
        for m in MANIFESTS.values()
    ]
    return {"agents": agents, "people": people, "model": CONFIGURED,
            "pii_engine": os.environ["AGENTDOCK_PII_ENGINE"]}


@app.post("/api/run")
async def api_run(request: Request) -> Response:
    if _limited(request, "run"):
        return JSONResponse({"error": "Too many requests, wait a minute."}, status_code=429)
    try:
        user = _caller(request)
    except Auth.exceptions.HTTPException as exc:
        return _deny(exc)
    body = await request.json()
    agent, message = body.get("agent"), str(body.get("message", "")).strip()
    if agent not in AGENTS:
        return JSONResponse({"error": "unknown agent"}, status_code=404)
    if not message or len(message) > MAX_INPUT:
        return JSONResponse({"error": f"message must be 1-{MAX_INPUT} characters"}, status_code=400)
    if body.get("stream"):
        return _ndjson_stream(agent, message, user)
    return JSONResponse(await run_once(agent, message, user))


@app.post("/api/gate")
async def api_gate(request: Request) -> Response:
    if _limited(request, "gate"):
        return JSONResponse({"error": "Too many requests, wait a minute."}, status_code=429)
    started = time.perf_counter()
    # The gate is deterministic: offline model, and A2A loopback pointed at the
    # offline agents too, so 27 cases never spend the live model's quota.
    async with _gate_lock:
        entry.AGENTS.update(OFFLINE)
        try:
            reports = [await run_agent(m, OFFLINE[n]) for n, m in MANIFESTS.items()]
        finally:
            entry.AGENTS.update(LIVE)
    problems = gate(reports, load_baseline(), MANIFESTS)
    return JSONResponse({
        "green": not problems,
        "problems": problems,
        "ms": round((time.perf_counter() - started) * 1000, 1),
        "agents": [
            {"agent": r.agent, "passed": r.passed, "cases": r.cases, "floor": r.min_pass_rate,
             "results": [{"id": c.id, "passed": c.passed, "ms": c.latency_ms, "failures": c.failures,
                          "output": c.output[:200]} for c in r.results]}
            for r in reports
        ],
    })


# --- MCP (stateless JSON-RPC over HTTP) --------------------------------------
def _rpc(id_: Any, result: Any = None, error: tuple[int, str] | None = None) -> dict[str, Any]:
    if error:
        return {"jsonrpc": "2.0", "id": id_, "error": {"code": error[0], "message": error[1]}}
    return {"jsonrpc": "2.0", "id": id_, "result": result}


@app.post("/mcp")
async def mcp(request: Request) -> Response:
    if _limited(request, "mcp"):
        return JSONResponse({"error": "Too many requests"}, status_code=429)
    try:
        user = _caller(request)
    except Auth.exceptions.HTTPException as exc:
        return _deny(exc)
    req = await request.json()
    method, id_, params = req.get("method"), req.get("id"), req.get("params") or {}
    if method and method.startswith("notifications/"):
        return Response(status_code=202)
    if method == "initialize":
        return JSONResponse(_rpc(id_, {
            "protocolVersion": params.get("protocolVersion", "2025-06-18"),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "agentdock-serverless", "version": "0.2.0"},
        }))
    if method == "ping":
        return JSONResponse(_rpc(id_, {}))
    if method == "tools/list":
        return JSONResponse(_rpc(id_, {"tools": [
            {"name": m.name, "description": m.description,
             "inputSchema": {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}}
            for m in MANIFESTS.values()
        ]}))
    if method == "tools/call":
        name, args = params.get("name"), params.get("arguments") or {}
        message = str(args.get("message", "")).strip()
        if name not in AGENTS:
            return JSONResponse(_rpc(id_, error=(-32602, f"unknown tool {name!r}")))
        if not message or len(message) > MAX_INPUT:
            return JSONResponse(_rpc(id_, error=(-32602, f"message must be 1-{MAX_INPUT} characters")))
        out = await run_once(name, message, user)
        return JSONResponse(_rpc(id_, {"content": [{"type": "text", "text": out["answer"]}], "isError": False}))
    return JSONResponse(_rpc(id_, error=(-32601, f"method {method!r} not supported")))


# --- A2A ---------------------------------------------------------------------
@app.get("/a2a/{agent}/.well-known/agent-card.json")
async def agent_card(agent: str, request: Request) -> Response:
    m = MANIFESTS.get(agent)
    if m is None:
        return JSONResponse({"error": "unknown agent"}, status_code=404)
    base = str(request.base_url).rstrip("/")
    return JSONResponse({
        "protocolVersion": "0.3.0", "name": m.name, "description": m.description,
        "url": f"{base}/a2a/{m.name}", "version": "0.2.0",
        "capabilities": {"streaming": False},
        "defaultInputModes": ["text/plain"], "defaultOutputModes": ["text/plain"],
        "securitySchemes": {"apiKey": {"type": "apiKey", "in": "header", "name": "x-api-key"}},
        "security": [{"apiKey": []}],
        "skills": [{"id": m.name, "name": m.name, "description": m.description, "tags": [m.kind, m.owner_team]}],
    })


@app.post("/a2a/{agent}")
async def a2a(agent: str, request: Request) -> Response:
    if _limited(request, "mcp"):
        return JSONResponse({"error": "Too many requests"}, status_code=429)
    try:
        user = _caller(request)
    except Auth.exceptions.HTTPException as exc:
        return _deny(exc)
    req = await request.json()
    id_ = req.get("id")
    if agent not in AGENTS:
        return JSONResponse(_rpc(id_, error=(-32001, "unknown agent")), status_code=404)
    if req.get("method") != "message/send":
        return JSONResponse(_rpc(id_, error=(-32601, "only message/send is supported here")))
    parts = ((req.get("params") or {}).get("message") or {}).get("parts") or []
    text = " ".join(p.get("text", "") for p in parts if p.get("kind") == "text").strip()
    if not text or len(text) > MAX_INPUT:
        return JSONResponse(_rpc(id_, error=(-32602, f"text must be 1-{MAX_INPUT} characters")))
    out = await run_once(agent, text, user)
    return JSONResponse(_rpc(id_, {
        "kind": "message", "role": "agent", "messageId": uuid.uuid4().hex,
        "parts": [{"kind": "text", "text": out["answer"]}],
    }))


# Local preview only: on Vercel, public/ is served by the CDN and never reaches this app.
if os.environ.get("AGENTDOCK_LOCAL_STATIC"):
    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(ROOT / "public" / "index.html")

    @app.get("/{name}.css")
    async def css(name: str) -> Response:
        f = ROOT / "public" / f"{name}.css"
        return FileResponse(f, media_type="text/css") if f.is_file() else Response(status_code=404)
