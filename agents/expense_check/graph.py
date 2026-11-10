"""Expense check: a deterministic workflow (parse -> rules -> reply).

No model call at all. Same platform features as the LLM agents: callers reach
it over MCP or A2A and every decision lands in the audit log.
"""

from __future__ import annotations

import re
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph

from agentdock.identity import current_user

LIMITS = {"meal": 60.0, "hotel": 250.0}
RECEIPT_OVER = 75.0
MANAGER_OVER = 1000.0
CATEGORIES = {
    "alcohol": ("alcohol", "wine", "beer", "bar tab", "drinks"),
    "meal": ("meal", "lunch", "dinner", "breakfast", "food"),
    "hotel": ("hotel", "lodging", "airbnb", "night"),
    "flight": ("flight", "airfare", "plane"),
    "taxi": ("taxi", "uber", "cab", "lyft"),
}
NO_RECEIPT = ("no receipt", "without receipt", "without a receipt", "lost the receipt", "lost receipt", "don't have a receipt")


class ExpenseState(MessagesState):
    amount: float | None
    category: str
    has_receipt: bool
    decision: str
    reason: str


def parse(state: ExpenseState) -> dict:
    last = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    text = (last.content if last and isinstance(last.content, str) else "").lower()
    m = re.search(r"\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)|(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:usd|dollars|\$)", text)
    amount = float((m.group(1) or m.group(2)).replace(",", "")) if m else None
    category = next((c for c, words in CATEGORIES.items() if any(w in text for w in words)), "other")
    has_receipt = "receipt" in text and not any(p in text for p in NO_RECEIPT)
    return {"amount": amount, "category": category, "has_receipt": has_receipt}


def route(state: ExpenseState) -> Literal["rules", "ask_amount"]:
    return "rules" if state.get("amount") is not None else "ask_amount"


def ask_amount(state: ExpenseState) -> dict:
    return {
        "decision": "need_info",
        "messages": [AIMessage("Tell me the amount and what it was for, e.g. '$45 taxi to the airport, receipt attached'.")],
    }


def rules(state: ExpenseState, config: RunnableConfig) -> dict:
    amount, category, receipt = state["amount"], state["category"], state["has_receipt"]
    caller = current_user(config)
    approver = (caller.manager if caller and caller.manager else "your manager")
    if category == "alcohol":
        decision, reason = "reject", "alcohol is never reimbursed"
    elif amount > RECEIPT_OVER and not receipt:
        decision, reason = "needs_receipt", f"expenses over ${RECEIPT_OVER:.0f} need a receipt"
    elif category in LIMITS and amount > LIMITS[category]:
        cap = LIMITS[category]
        decision, reason = "approve_capped", f"{category} is capped at ${cap:.0f}, so ${cap:.2f} is approved and ${amount - cap:.2f} is not"
    elif amount > MANAGER_OVER:
        decision, reason = "needs_manager", f"expenses over ${MANAGER_OVER:.0f} need approval from {approver}"
    else:
        decision, reason = "approve", "within policy"
    text = f"Decision: {decision.replace('_', ' ').upper()} for ${amount:.2f} ({category}): {reason}."
    return {"decision": decision, "reason": reason, "messages": [AIMessage(text)]}


def build() -> StateGraph:
    g = StateGraph(ExpenseState)
    g.add_node("parse", parse)
    g.add_node("rules", rules)
    g.add_node("ask_amount", ask_amount)
    g.add_edge(START, "parse")
    g.add_conditional_edges("parse", route)
    g.add_edge("rules", END)
    g.add_edge("ask_amount", END)
    return g
