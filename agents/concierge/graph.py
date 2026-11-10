"""Concierge: routes a question to a specialist agent over A2A.

The specialist runs as the same person who asked the concierge (delegation via
``x-on-behalf-of``), so routing through the concierge never widens access.
"""

from __future__ import annotations

import uuid
from typing import Literal, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agentdock import a2a_client
from agentdock.identity import current_user
from agentdock.models import chat_model

SYSTEM = (
    "You route employee questions. Call ask_agent with agent='expense_check' for a specific "
    "expense to check, or agent='policy_qa' for HR policy and leave balances. Pass the "
    "question through unchanged, then relay the answer briefly."
)
EXPENSE_WORDS = ("expense", "reimburse", "claim", "spent", "$", "usd", "dollars")


@tool
async def ask_agent(agent: Literal["policy_qa", "expense_check"], question: str, config: RunnableConfig) -> str:
    """Ask a specialist agent: 'policy_qa' (HR policy, leave balances) or 'expense_check' (one expense)."""
    caller = current_user(config)
    if caller is None:
        return "error: no authenticated caller"
    try:
        return await a2a_client.ask(agent, question, caller)
    except Exception as exc:
        return f"error: {agent} is unavailable ({type(exc).__name__})"


def offline_policy(messages: Sequence[BaseMessage], tool_names: list[str]) -> AIMessage:
    last = messages[-1]
    if isinstance(last, ToolMessage):
        return AIMessage(str(last.content))
    question = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    text = question.content if question and isinstance(question.content, str) else ""
    agent = "expense_check" if any(w in text.lower() for w in EXPENSE_WORDS) else "policy_qa"
    return AIMessage(
        "",
        tool_calls=[{"name": "ask_agent", "args": {"agent": agent, "question": text}, "id": f"call_{uuid.uuid4().hex[:8]}"}],
    )


def build() -> StateGraph:
    model = chat_model(offline_policy).bind_tools([ask_agent])

    async def agent(state: MessagesState) -> dict:
        return {"messages": [await model.ainvoke([SystemMessage(SYSTEM), *state["messages"]])]}

    g = StateGraph(MessagesState)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode([ask_agent]))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)
    g.add_edge("tools", "agent")
    return g
