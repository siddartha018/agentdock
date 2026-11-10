"""__NAME__: a ReAct agent. Replace the tool and the offline policy with your own.

You only write this file, agent.toml and evals.jsonl. The platform adds auth,
role checks, PII redaction, the audit log and the MCP + A2A endpoints.
"""

from __future__ import annotations

import uuid
from typing import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agentdock.identity import current_user
from agentdock.models import chat_model

SYSTEM = "You are __NAME__. Use your tools, then answer in one or two sentences."


@tool
def whoami(config: RunnableConfig) -> str:
    """Return the caller's identity and role."""
    caller = current_user(config)
    return f"{caller.identity} ({caller.role})" if caller else "unknown"


TOOLS = [whoami]


def offline_policy(messages: Sequence[BaseMessage], tool_names: list[str]) -> AIMessage:
    """What the agent does with no LLM configured. Keep it simple and deterministic."""
    last = messages[-1]
    if isinstance(last, ToolMessage):
        return AIMessage(f"You are {last.content}.")
    question = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    if question and "who" in str(question.content).lower():
        return AIMessage("", tool_calls=[{"name": "whoami", "args": {}, "id": f"call_{uuid.uuid4().hex[:8]}"}])
    return AIMessage("Hello from __NAME__.")


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
