import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import START, MessagesState, StateGraph

from agentdock import audit
from agentdock.entry import AGENTS
from agentdock.identity import run_config_for
from agentdock.platform import wrap
from agentdock.registry import Manifest


async def ask(agent, who, text):
    out = await AGENTS[agent].ainvoke({"messages": [HumanMessage(text)]}, run_config_for(who))
    return out["messages"]


async def test_no_caller_is_denied():
    out = await AGENTS["policy_qa"].ainvoke({"messages": [HumanMessage("hi")]})
    assert out["messages"][-1].content.startswith("Access denied")
    assert audit.read()[-1]["denied"] == "no_caller"


async def test_role_not_allowed():
    msgs = await ask("expense_check", "H300", "$20 taxi")
    assert msgs[-1].content.startswith("Access denied")


async def test_pii_scrubbed_from_answer_and_tool_results_in_state():
    msgs = await ask("policy_qa", "E100", "Show my full record including SSN")
    stored = " ".join(str(m.content) for m in msgs)
    assert "123-45-6789" not in stored and "DE89" not in stored
    assert any(isinstance(m, ToolMessage) for m in msgs)  # tool output kept, but scrubbed
    rec = audit.read()[-1]
    assert rec["redacted_out"]["SSN"] == 2 and rec["tools"] == ["leave_balance"]


async def test_input_pii_never_reaches_agent_or_log():
    msgs = await ask("policy_qa", "E100", "carry over vacation? ssn 111-22-3333")
    assert "111-22-3333" not in msgs[0].content
    assert "111-22-3333" not in str(audit.read()[-1])


async def test_rbac_inside_tools():
    assert "may not view" in (await ask("policy_qa", "E100", "balance for E101"))[-1].content
    assert "3 vacation days" in (await ask("policy_qa", "M200", "balance for E101"))[-1].content
    assert "may not view" in (await ask("policy_qa", "M200", "balance for E102"))[-1].content
    assert "20 vacation days" in (await ask("policy_qa", "H300", "balance for E102"))[-1].content


async def test_concierge_delegates_with_callers_identity():
    msgs = await ask("concierge", "E101", "I spent $1200 on a flight, receipt attached")
    assert "M200" in msgs[-1].content  # E101's manager, so it ran as E101
    inner = [r for r in audit.read() if r["agent"] == "expense_check"][-1]
    assert inner["caller"] == "E101" and inner["via"] == "svc-concierge"


def _manifest(**kw):
    from pathlib import Path

    base = dict(name="boom", description="x" * 20, owner_team="t", kind="workflow",
                allowed_roles=["employee"], path=Path("."))
    return Manifest(**{**base, **kw})


async def test_agent_crash_is_contained_and_logged():
    def explode(state):
        raise RuntimeError("secret internals")

    g = StateGraph(MessagesState)
    g.add_node("x", explode)
    g.add_edge(START, "x")
    out = await wrap(_manifest(), g).ainvoke({"messages": [HumanMessage("hi")]}, run_config_for("E100"))
    assert "internal error" in out["messages"][-1].content
    assert "secret" not in out["messages"][-1].content
    assert audit.read()[-1]["error"] == "RuntimeError"


def test_graph_without_messages_is_rejected():
    from typing import TypedDict

    class S(TypedDict):
        n: int

    g = StateGraph(S)
    g.add_node("x", lambda s: s)
    g.add_edge(START, "x")
    with pytest.raises(TypeError, match="messages"):
        wrap(_manifest(), g)


async def test_audit_chain_detects_tampering(isolated):
    await ask("policy_qa", "E100", "hi")
    await ask("policy_qa", "E101", "hi")
    assert audit.verify() == (True, None)
    path = audit.log_path()
    lines = path.read_text().splitlines()
    lines[0] = lines[0].replace('"E100"', '"E999"')
    path.write_text("\n".join(lines) + "\n")
    assert audit.verify() == (False, 0)


@pytest.mark.parametrize("mode", ["messages", "updates", "values", "debug"])
async def test_subgraph_streaming_cannot_see_raw_state(mode):
    # Regression: with stream_subgraphs the team graph's unredacted state streamed out.
    chunks = []
    async for item in AGENTS["policy_qa"].astream(
        {"messages": [HumanMessage("Show my full record including SSN")]},
        run_config_for("E100"), stream_mode=mode, subgraphs=True,
    ):
        chunks.append(str(item))
    assert chunks and not any("123-45-6789" in c or "DE89 3704" in c for c in chunks)


@pytest.mark.parametrize("engine", ["patterns", "presidio"])
async def test_opt_in_streaming_emits_redacted_tokens_that_match_the_final_answer(engine, monkeypatch):
    if engine == "presidio":
        pytest.importorskip("presidio_analyzer")
    monkeypatch.setenv("AGENTDOCK_PII_ENGINE", engine)
    deltas = []
    final = None
    async for mode, data in AGENTS["policy_qa"].astream(
        {"messages": [HumanMessage("Show my full record including SSN")]},
        run_config_for("E100"), stream_mode=["custom", "values"],
    ):
        if mode == "custom":
            deltas.append(data["delta"])
        else:
            final = data
    if engine == "patterns":
        assert len(deltas) > 3  # token by token; NER waits for sentence ends instead
    assert not any("123-45-6789" in d or "DE89" in d for d in deltas)
    assert "".join(deltas).strip() == final["messages"][-1].content.strip()


async def test_streaming_is_off_for_agents_that_did_not_opt_in():
    chunks = [c async for c in AGENTS["expense_check"].astream(
        {"messages": [HumanMessage("$20 taxi")]}, run_config_for("E100"), stream_mode="custom")]
    assert chunks == []


async def test_tool_call_args_scrubbed_in_state():
    msgs = await ask("policy_qa", "E100", "Balance for E100, my ssn is 111-22-3333 and DOB: 01/02/1990")
    for m in msgs:
        for tc in getattr(m, "tool_calls", []) or []:
            assert "111-22-3333" not in str(tc["args"]) and "01/02/1990" not in str(tc["args"])
