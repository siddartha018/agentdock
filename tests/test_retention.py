import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from langgraph_sdk import Auth

from agentdock.auth import own_threads, resolve
from agentdock.retention import sweep_once

NOW = datetime(2026, 11, 11, 12, 0, tzinfo=timezone.utc)


class FakeThreads:
    def __init__(self, ages_min):
        self.rows = [
            {"thread_id": f"t{i}", "updated_at": (NOW - timedelta(minutes=a)).isoformat()}
            for i, a in enumerate(ages_min)
        ]
        self.deleted = []

    async def search(self, limit, sort_by, sort_order):
        assert sort_by == "updated_at" and sort_order == "asc"
        return sorted(self.rows, key=lambda r: r["updated_at"])[:limit]

    async def delete(self, tid):
        self.deleted.append(tid)
        self.rows = [r for r in self.rows if r["thread_id"] != tid]


class FakeClient:
    def __init__(self, ages):
        self.threads = FakeThreads(ages)


def test_deletes_only_expired_threads():
    c = FakeClient([5, 120, 61, 59, 3000])
    n = asyncio.run(sweep_once(c, retention_minutes=60, now=NOW))
    assert n == 3 and sorted(c.threads.deleted) == ["t1", "t2", "t4"]
    assert {r["thread_id"] for r in c.threads.rows} == {"t0", "t3"}


def test_pages_through_more_than_one_batch():
    c = FakeClient([100] * 250 + [1])
    assert asyncio.run(sweep_once(c, retention_minutes=60, now=NOW)) == 250


class _Ctx:
    def __init__(self, action, user):
        self.action, self.user = action, type("U", (), user)()


@pytest.mark.parametrize("action", ["search", "delete"])
def test_sweeper_may_search_and_delete_any_thread(action):
    user = resolve({b"x-api-key": b"dk_dev_svc-retention"})
    ctx = _Ctx(action, {"identity": user["identity"], "permissions": user["permissions"]})
    assert asyncio.run(own_threads(ctx, {})) is True


@pytest.mark.parametrize("action", ["read", "create", "create_run", "update"])
def test_sweeper_cannot_read_or_write_conversations(action):
    ctx = _Ctx(action, {"identity": "svc-retention", "permissions": ["retention"]})
    with pytest.raises(Auth.exceptions.HTTPException):
        asyncio.run(own_threads(ctx, {"metadata": {}}))
