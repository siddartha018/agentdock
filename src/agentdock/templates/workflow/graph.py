"""__NAME__: a deterministic workflow. Replace the steps with your own.

You only write this file, agent.toml and evals.jsonl. The platform adds auth,
role checks, PII redaction, the audit log and the MCP + A2A endpoints.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph


class State(MessagesState):
    words: int


def count(state: State) -> dict:
    last = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    return {"words": len(str(last.content).split()) if last else 0}


def reply(state: State) -> dict:
    return {"messages": [AIMessage(f"__NAME__ counted {state['words']} words.")]}


def build() -> StateGraph:
    g = StateGraph(State)
    g.add_node("count", count)
    g.add_node("reply", reply)
    g.add_edge(START, "count")
    g.add_edge("count", "reply")
    g.add_edge("reply", END)
    return g
