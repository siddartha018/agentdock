import json
import shutil

import pytest

from agentdock import cli
from agentdock.evals import AgentReport, Case, gate, load_cases, score
from agentdock.registry import ManifestError, load_manifest


def test_score_checks():
    c = Case("x", "E100", "q", {"contains": ["12"], "not_contains": ["ssn"], "tools": ["t"], "denied": False})
    assert score(c, "you have 12 days", ["t"]) == []
    assert len(score(c, "Access denied, ssn", [])) == 4


def test_gate_blocks_regression_floor_and_thin_suites():
    from agentdock.registry import discover

    ms = {m.name: m for m in discover()}
    good = AgentReport("policy_qa", 12, 12, 0.9, [])
    assert gate([good], {"policy_qa": 1.0}, ms) == []
    regressed = AgentReport("policy_qa", 12, 11, 0.9, [])
    assert any("regressed" in p for p in gate([regressed], {"policy_qa": 1.0}, ms))
    floor = AgentReport("policy_qa", 12, 9, 0.9, [])
    assert any("below its floor" in p for p in gate([floor], {}, ms))
    thin = AgentReport("policy_qa", 3, 3, 0.9, [])
    assert any("needs at least" in p for p in gate([thin], {}, ms))


def test_bad_eval_files_are_rejected(tmp_path):
    f = tmp_path / "evals.jsonl"
    f.write_text('{"id": "a", "as": "NOBODY", "input": "q", "expect": {"contains": ["x"]}}\n')
    with pytest.raises(ValueError, match="unknown identity"):
        load_cases(f)
    f.write_text('{"id": "a", "as": "E100", "input": "q", "expect": {"vibes": true}}\n')
    with pytest.raises(ValueError, match="expect"):
        load_cases(f)


def test_manifest_validation(tmp_path):
    d = tmp_path / "bad_agent"
    d.mkdir()
    (d / "agent.toml").write_text('name = "bad_agent"\ndescription = "short"\nkind = "chat"\nallowed_roles = ["ceo"]\n')
    with pytest.raises(ManifestError) as e:
        load_manifest(d)
    for word in ("description", "owner_team", "kind", "allowed_roles"):
        assert word in str(e.value)


def test_repo_agents_pass_check():
    assert cli.check() == []


@pytest.mark.parametrize("kind", ["react", "workflow"])
def test_dock_new_scaffolds_an_agent_that_passes_its_own_gate(kind, monkeypatch, tmp_path):
    """A new agent is valid and green straight from the template."""
    import agentdock.registry as reg

    agents = tmp_path / "agents"
    agents.mkdir()
    monkeypatch.setattr(reg, "AGENTS_DIR", agents)
    monkeypatch.setattr(cli, "AGENTS_DIR", agents)
    monkeypatch.setattr(cli, "write_langgraph_config", lambda *a, **k: None)
    monkeypatch.setattr(cli, "discover", lambda: reg.discover(agents))
    name = f"demo_{kind}"
    assert cli.main(["new", name, "--kind", kind, "--team", "qa"]) == 0
    assert cli.main(["new", name, "--kind", kind, "--team", "qa"]) == 2  # no overwrite

    import asyncio

    from agentdock.evals import run_agent
    from agentdock.platform import wrap
    from agentdock.registry import import_agent_module, load_manifest

    m = load_manifest(agents / name)
    report = asyncio.run(run_agent(m, wrap(m, import_agent_module(m).build())))
    assert report.cases >= m.gate.min_cases and report.pass_rate == 1.0, [r for r in report.results if not r.passed]
