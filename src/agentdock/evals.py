"""Eval gate: every agent ships ``evals.jsonl`` and cannot be served if it regresses.

Case format (one JSON object per line)::

    {"id": "own-balance", "as": "E100", "input": "How many vacation days do I have?",
     "expect": {"contains": ["12 vacation days"], "not_contains": ["SSN"],
                "tools": ["leave_balance"], "denied": false}}

``as`` is the identity the case runs as, so access rules are tested the same
way they are enforced. Cases run in-process through the platform wrapper.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from agentdock.identity import ROOT, directory, run_config_for
from agentdock.messages import text_of
from agentdock.registry import Manifest

BASELINE = ROOT / "evals" / "baseline.json"
RESULTS = ROOT / "evals" / "results"
EXPECT_KEYS = {"contains", "not_contains", "tools", "no_tools", "denied"}


@dataclass
class Case:
    id: str
    as_: str
    input: str
    expect: dict[str, Any]


@dataclass
class CaseResult:
    id: str
    passed: bool
    output: str
    tools: list[str]
    failures: list[str] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass
class AgentReport:
    agent: str
    cases: int
    passed: int
    min_pass_rate: float
    results: list[CaseResult]

    @property
    def pass_rate(self) -> float:
        return self.passed / self.cases if self.cases else 0.0


def platform_settings() -> dict[str, Any]:
    return tomllib.loads((ROOT / "config" / "platform.toml").read_text())


def load_cases(path: Path) -> list[Case]:
    if not path.exists():
        raise ValueError(f"{path.parent.name}: missing evals.jsonl")
    cases, seen = [], set()
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{n}: bad JSON ({exc.msg})") from exc
        missing = {"id", "as", "input", "expect"} - raw.keys()
        if missing:
            raise ValueError(f"{path}:{n}: missing {sorted(missing)}")
        if raw["id"] in seen:
            raise ValueError(f"{path}:{n}: duplicate id {raw['id']!r}")
        if directory().get(raw["as"]) is None:
            raise ValueError(f"{path}:{n}: unknown identity {raw['as']!r}")
        unknown = set(raw["expect"]) - EXPECT_KEYS
        if unknown or not raw["expect"]:
            raise ValueError(f"{path}:{n}: expect needs some of {sorted(EXPECT_KEYS)}, got {sorted(raw['expect'])}")
        seen.add(raw["id"])
        cases.append(Case(raw["id"], raw["as"], raw["input"], raw["expect"]))
    return cases


def score(case: Case, output: str, tools: list[str]) -> list[str]:
    exp, low, failures = case.expect, output.lower(), []
    # A "contains" entry may be a list: any one of the phrasings passes. Real
    # models word things their own way; check the fact, not the sentence.
    for s in exp.get("contains", []):
        options = s if isinstance(s, list) else [s]
        if not any(o.lower() in low for o in options):
            failures.append(f"missing {options[0]!r}" if len(options) == 1 else f"missing any of {options}")
    for s in exp.get("not_contains", []):
        if s.lower() in low:
            failures.append(f"must not contain {s!r}")
    for t in exp.get("tools", []):
        if t not in tools:
            failures.append(f"tool {t!r} not called")
    if exp.get("no_tools") and tools:
        failures.append(f"expected no tool calls, got {tools}")
    if "denied" in exp and exp["denied"] != low.startswith("access denied"):
        failures.append("expected access denied" if exp["denied"] else "unexpectedly denied")
    return failures


async def run_agent(manifest: Manifest, graph: Any, pace_s: float = 0.0) -> AgentReport:
    cases = load_cases(manifest.evals_file)
    results = []
    # Evals write to a throwaway audit log so they never mix with real traffic.
    prev = os.environ.get("AGENTDOCK_AUDIT_LOG")
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["AGENTDOCK_AUDIT_LOG"] = str(Path(tmp) / "audit.jsonl")
        try:
            for i, case in enumerate(cases):
                if pace_s and i:
                    await asyncio.sleep(pace_s)  # stay under free-tier rate limits on real models
                t0 = time.perf_counter()
                out = await graph.ainvoke({"messages": [HumanMessage(case.input)]}, run_config_for(case.as_))
                msgs = out["messages"]
                text = text_of(msgs[-1])
                tools = [tc["name"] for m in msgs if isinstance(m, AIMessage) for tc in m.tool_calls]
                failures = score(case, text, tools)
                results.append(
                    CaseResult(case.id, not failures, text[:400], tools, failures, round((time.perf_counter() - t0) * 1000, 1))
                )
        finally:
            if prev is None:
                os.environ.pop("AGENTDOCK_AUDIT_LOG", None)
            else:
                os.environ["AGENTDOCK_AUDIT_LOG"] = prev
    return AgentReport(manifest.name, len(cases), sum(r.passed for r in results), manifest.gate.min_pass_rate, results)


def gate(reports: list[AgentReport], baseline: dict[str, float], manifests: dict[str, Manifest]) -> list[str]:
    """Reasons to block a deploy; empty means the gate is green."""
    tol = float(platform_settings().get("regression_tolerance", 0.0))
    problems = []
    for r in reports:
        m = manifests[r.agent]
        if r.cases < m.gate.min_cases:
            problems.append(f"{r.agent}: {r.cases} eval cases, needs at least {m.gate.min_cases}")
        if r.pass_rate < m.gate.min_pass_rate:
            problems.append(f"{r.agent}: pass rate {r.pass_rate:.0%} is below its floor {m.gate.min_pass_rate:.0%}")
        base = baseline.get(r.agent)
        if base is not None and r.pass_rate < base - tol:
            problems.append(f"{r.agent}: pass rate {r.pass_rate:.0%} regressed from baseline {base:.0%}")
    return problems


def load_baseline() -> dict[str, float]:
    return json.loads(BASELINE.read_text()) if BASELINE.exists() else {}


def save_baseline(reports: list[AgentReport]) -> None:
    data = load_baseline()
    data.update({r.agent: round(r.pass_rate, 4) for r in reports})
    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE.write_text(json.dumps(dict(sorted(data.items())), indent=2) + "\n")


def save_results(reports: list[AgentReport], model: str) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "latest.json"
    path.write_text(
        json.dumps(
            {"model": model, "agents": [{**asdict(r), "pass_rate": r.pass_rate} for r in reports]},
            indent=2,
        )
        + "\n"
    )
    return path
