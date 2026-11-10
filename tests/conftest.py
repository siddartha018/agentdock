import pytest


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Each test gets its own audit log, offline model and in-process A2A."""
    monkeypatch.setenv("AGENTDOCK_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("AGENTDOCK_MODEL", "offline")
    monkeypatch.delenv("AGENTDOCK_A2A_URL", raising=False)
    yield tmp_path
