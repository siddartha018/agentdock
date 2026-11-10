"""The wrapper that turns a team's graph into a platform agent.

A team's ``build()`` returns any LangGraph graph with a ``messages`` key. The
platform puts it inside one node that, on every run:

1. resolves the caller (set by the auth handler) and enforces ``allowed_roles``;
2. redacts PII in the person's messages before the agent sees them;
3. runs the team's graph as a subgraph (streaming and checkpoints still work);
4. redacts PII in everything the agent produced, including tool results that
   are kept in thread state;
5. writes one audit record.

The server then exposes the wrapped graph over REST, MCP and A2A unchanged.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agentdock import audit
from agentdock.identity import current_user
from agentdock.messages import text_of
from agentdock.models import model_name
from agentdock.pii import redact_content
from agentdock.registry import Manifest

log = logging.getLogger("agentdock")


def _merge(into: dict[str, int], more: dict[str, int]) -> None:
    for k, v in more.items():
        into[k] = into.get(k, 0) + v


def _text(msg: BaseMessage | None, limit: int = 300) -> str:
    if msg is None:
        return ""
    return text_of(msg)[:limit]


def _compile(inner: Any) -> CompiledStateGraph:
    graph = inner.compile(checkpointer=False) if isinstance(inner, StateGraph) else inner
    if not isinstance(graph, CompiledStateGraph):
        raise TypeError("build() must return a StateGraph or a compiled graph")
    # Never persist the team graph's own checkpoints: they would hold the
    # unredacted messages. Only the wrapper's (scrubbed) state is stored.
    graph.checkpointer = False
    if "messages" not in graph.get_input_jsonschema().get("properties", {}):
        raise TypeError("the agent's state must have a 'messages' key (A2A and MCP speak in messages)")
    return graph


def wrap(manifest: Manifest, inner: Any) -> CompiledStateGraph:
    agent = _compile(inner)
    allowed = set(manifest.allowed_roles)

    async def run(state: MessagesState, config: RunnableConfig) -> dict[str, Any]:
        started = time.perf_counter()
        caller = current_user(config)
        messages = list(state["messages"])
        configurable = config.get("configurable") or {}
        event: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "agent": manifest.name,
            "thread_id": configurable.get("thread_id"),
            "caller": caller.identity if caller else None,
            "role": caller.role if caller else None,
            "via": caller.via if caller else None,
            "model": model_name(),
            "input": _text(next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)),
        }

        async def finish(new: list[BaseMessage], **extra: Any) -> dict[str, Any]:
            event.update(extra)
            event["output"] = _text(new[-1] if new else None)
            event["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
            # File I/O off the event loop: the server shares one loop across all runs.
            await asyncio.to_thread(audit.record, event)
            return {"messages": new}

        if caller is None:
            return await finish([AIMessage("Access denied: this agent needs an authenticated caller.")], denied="no_caller")
        if caller.role not in allowed:
            return await finish(
                [AIMessage(f"Access denied: {manifest.name} is not available to the {caller.role} role.")],
                denied="role",
            )

        redacted_in: dict[str, int] = {}
        replaced: list[BaseMessage] = []
        cleaned: list[BaseMessage] = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                content, counts = redact_content(msg.content)
                if counts:
                    msg = msg.model_copy(update={"content": content})
                    replaced.append(msg)
                    _merge(redacted_in, counts)
            cleaned.append(msg)
        if redacted_in:  # keep the redacted text in the event, never the raw input
            event["input"] = _text(next(m for m in reversed(cleaned) if isinstance(m, HumanMessage)))

        seen = {m.id for m in cleaned if m.id}
        # Run the team's graph sealed off from the server's run: a fresh context
        # (no inherited callbacks, stream writers or checkpointer) and a config
        # that carries only the caller. Otherwise a client asking for
        # `stream_subgraphs` would receive its raw, unredacted state.
        sealed_config: RunnableConfig = {
            "configurable": {"langgraph_auth_user": configurable.get("langgraph_auth_user")},
            "recursion_limit": config.get("recursion_limit", 25),
            "metadata": {"agentdock_agent": manifest.name, "thread_id": configurable.get("thread_id")},
        }
        try:
            result = await asyncio.create_task(
                agent.ainvoke({"messages": cleaned}, sealed_config), context=contextvars.Context()
            )
        except Exception as exc:  # the caller gets a clean message; operators get the traceback
            log.exception("agent %s failed for %s", manifest.name, caller.identity)
            return await finish(
                replaced + [AIMessage(f"The {manifest.name} agent hit an internal error. It has been logged.")],
                error=type(exc).__name__,
                redacted_in=redacted_in,
            )

        produced = [m for m in result["messages"] if not m.id or m.id not in seen]
        redacted_out: dict[str, int] = {}
        scrubbed: list[BaseMessage] = []
        tools: list[str] = []
        for msg in produced:
            if isinstance(msg, AIMessage):
                tools.extend(tc["name"] for tc in msg.tool_calls)
            if isinstance(msg, (AIMessage, ToolMessage)):
                content, counts = redact_content(msg.content)
                if counts:
                    msg = msg.model_copy(update={"content": content})
                    _merge(redacted_out, counts)
            scrubbed.append(msg)
        return await finish(replaced + scrubbed, tools=tools, redacted_in=redacted_in, redacted_out=redacted_out)

    outer = StateGraph(MessagesState)
    outer.add_node(manifest.name, run)
    outer.add_edge(START, manifest.name)
    outer.add_edge(manifest.name, END)
    compiled = outer.compile(name=manifest.name)
    compiled.dock_manifest = manifest  # type: ignore[attr-defined]
    return compiled
