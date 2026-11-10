"""HR policy Q&A: a ReAct loop with two tools.

This file is all the team wrote. Auth, PII redaction, audit, MCP and A2A come
from the platform wrapper.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agentdock.identity import DATA, current_user, directory
from agentdock.models import chat_model

HR = json.loads((DATA / "hr.json").read_text())
SYSTEM = (
    "You are the HR policy assistant. Use lookup_policy for policy questions and leave_balance "
    "for balances. Answer in one or two sentences using only tool results. Never reveal SSNs "
    "or bank details."
)
TOPIC_WORDS = {
    "vacation": ["vacation", "pto", "holiday leave", "carry over", "carryover", "annual leave"],
    "sick": ["sick", "doctor", "ill"],
    "parental": ["parental", "maternity", "paternity", "baby", "adoption"],
    "remote": ["remote", "work from home", "wfh", "another country", "abroad"],
    "expenses": ["expense", "reimburse", "meal", "hotel", "receipt", "alcohol"],
    "holidays": ["public holiday", "holidays", "bank holiday"],
}
STRONG_BALANCE_WORDS = ("balance", "days left", "days do i have", "full record", "my record")
BALANCE_WORDS = ("balance", "days left", "days do i have", "days does", "ssn", "bank", "full record", "my record")


def _match_topic(text: str) -> str | None:
    text = text.lower()
    for topic, words in TOPIC_WORDS.items():
        if any(w in text for w in words):
            return topic
    return None


@tool
def lookup_policy(topic: str) -> str:
    """Look up the company policy for a topic: vacation, sick, parental, remote, expenses or holidays."""
    key = topic.lower().strip()
    if key not in HR["policies"]:
        key = _match_topic(key) or ""
    if key not in HR["policies"]:
        return f"No policy found for {topic!r}. Known topics: {', '.join(HR['policies'])}."
    return f"[{key}] {HR['policies'][key]}"


@tool
def leave_balance(employee_id: str, config: RunnableConfig) -> str:
    """Get an employee's remaining vacation and sick days. Use employee_id 'me' for the caller."""
    caller = current_user(config)
    if caller is None:
        return "error: no authenticated caller"
    target = caller.identity if employee_id.lower() in ("me", "self", "my") else employee_id.upper()
    allowed = (
        target == caller.identity
        or caller.role == "hr"
        or (caller.role == "manager" and target in directory().reports_of(caller.identity))
    )
    if not allowed:
        return f"error: {caller.identity} ({caller.role}) may not view {target}'s leave balance"
    record = HR["leave"].get(target)
    if record is None:
        return f"error: no employee {target}"
    # The upstream HR API returns the whole record, SSN included. The platform's
    # output guard is what keeps that out of answers and thread state.
    return json.dumps({"employee_id": target, **record})


TOOLS = [lookup_policy, leave_balance]


def offline_policy(messages: Sequence[BaseMessage], tool_names: list[str]) -> AIMessage:
    """Scripted stand-in for an LLM so the agent runs with no API key."""
    question = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    q = (question.content if question and isinstance(question.content, str) else "").lower()
    q = re.sub(r"\[[a-z]+ redacted\]", "", q)  # placeholders left by the platform's PII guard
    last = messages[-1]
    if isinstance(last, ToolMessage):
        out = str(last.content)
        if out.startswith("error:"):
            return AIMessage(f"I can't do that: {out[len('error: '):]}.")
        if last.name == "leave_balance":
            rec = json.loads(out)
            if any(w in q for w in ("ssn", "bank", "full record", "my record")):
                # A careless model echoing the raw record: exactly what the PII guard is for.
                return AIMessage(f"Here is the full record: {out}")
            return AIMessage(
                f"{rec['employee_id']} has {rec['vacation_days']} vacation days and "
                f"{rec['sick_days']} sick days left."
            )
        return AIMessage(f"Per policy: {out.split('] ', 1)[-1]}")

    topic = _match_topic(q)
    wants_balance = any(w in q for w in BALANCE_WORDS)
    # A policy topic wins unless the question is clearly about a balance or a record.
    if topic and not any(w in q for w in STRONG_BALANCE_WORDS):
        wants_balance = False
    if "leave_balance" in tool_names and wants_balance:
        m = re.search(r"\b([emh]\d{3})\b", q)
        return _call("leave_balance", employee_id=m.group(1).upper() if m else "me")
    if topic and "lookup_policy" in tool_names:
        return _call("lookup_policy", topic=topic)
    return AIMessage(
        "I can answer HR policy questions (vacation, sick, parental, remote work, expenses, "
        "holidays) and look up leave balances."
    )


def _call(name: str, **args: str) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}"}])


def build() -> StateGraph:
    model = chat_model(offline_policy).bind_tools(TOOLS)

    def agent(state: MessagesState) -> dict:
        return {"messages": [model.invoke([SystemMessage(SYSTEM), *state["messages"]])]}

    g = StateGraph(MessagesState)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)
    g.add_edge("tools", "agent")
    return g
