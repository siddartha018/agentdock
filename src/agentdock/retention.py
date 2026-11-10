"""Retention: delete conversations older than ``retention_minutes``.

The Agent Server has its own TTL config, but the in-memory dev runtime ships
it as a stub ("Not implemented for inmem server"), so the platform enforces
retention itself. The sweeper runs inside the server (see ``http_app.py``)
and calls the server's own API as ``svc-retention``, an identity whose only
extra right is to list and delete expired threads.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

log = logging.getLogger("agentdock.retention")


def settings() -> tuple[float, float]:
    """(retention minutes, sweep interval seconds); env overrides for tests."""
    from agentdock.evals import platform_settings

    s = platform_settings()
    minutes = float(os.environ.get("AGENTDOCK_RETENTION_MINUTES", s.get("retention_minutes", 43200)))
    every = float(os.environ.get("AGENTDOCK_SWEEP_SECONDS", s.get("sweep_interval_seconds", 600)))
    return minutes, every


def _parse(ts: Any) -> datetime:
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))


async def sweep_once(client: Any, retention_minutes: float, now: datetime | None = None) -> int:
    """Delete every thread last updated before the cutoff. Returns how many."""
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(minutes=retention_minutes)
    deleted = 0
    while True:
        batch = await client.threads.search(limit=100, sort_by="updated_at", sort_order="asc")
        expired = [t for t in batch if _parse(t["updated_at"]) < cutoff]
        for t in expired:
            await client.threads.delete(t["thread_id"])
            deleted += 1
        if len(expired) < len(batch) or len(batch) < 100:
            return deleted


def _say(msg: str) -> None:
    # The server configures its own logging; stderr always reaches its log.
    print(f"[agentdock.retention] {msg}", file=sys.stderr, flush=True)


async def run_forever() -> None:
    try:
        await _run()
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # never die silently: a dead sweeper means data kept forever
        _say(f"FATAL: sweeper stopped: {type(exc).__name__}: {exc}")
        raise


async def _run() -> None:
    from langgraph_sdk import get_client

    key = os.environ.get("AGENTDOCK_RETENTION_KEY")
    if not key:
        _say("AGENTDOCK_RETENTION_KEY not set: retention sweeper DISABLED")
        return
    minutes, every = settings()
    # api_key=, not a header: the SDK rejects x-api-key in custom headers.
    client = get_client(api_key=key)  # no url: in-process, talks to this server
    _say(f"sweeper on: delete threads idle > {minutes} min, every {every} s")
    while True:
        await asyncio.sleep(every)
        try:
            n = await sweep_once(client, minutes)
            if n:
                _say(f"deleted {n} expired thread(s)")
        except Exception as exc:
            _say(f"sweep failed: {type(exc).__name__}: {exc}")
