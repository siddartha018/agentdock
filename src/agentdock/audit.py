"""Append-only, hash-chained audit log of every agent run.

One JSON line per run: who called, through which service, which tools ran,
what was redacted, how long it took. Each line carries the hash of the previous
line, so editing or deleting a past record breaks ``verify()``.
LangSmith tracing is added automatically when ``LANGSMITH_API_KEY`` is set;
this log is the part that works with no account at all.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

from agentdock.identity import ROOT

_lock = threading.Lock()
GENESIS = "0" * 64


def log_path() -> Path:
    return Path(os.environ.get("AGENTDOCK_AUDIT_LOG", ROOT / ".traces" / "audit.jsonl"))


def _digest(record: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def _last_hash(path: Path) -> str:
    if not path.exists() or path.stat().st_size == 0:
        return GENESIS
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        pos = f.tell() - 1
        while pos > 0:
            f.seek(pos - 1)
            if f.read(1) == b"\n":
                break
            pos -= 1
        f.seek(max(pos, 0))
        last = f.readline()
    return json.loads(last)["hash"]


def record(event: dict[str, Any]) -> dict[str, Any]:
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        body = {**event, "prev": _last_hash(path)}
        body["hash"] = _digest(body)
        with path.open("a") as f:
            f.write(json.dumps(body, sort_keys=True) + "\n")
    return body


def read(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or log_path()
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify(path: Path | None = None) -> tuple[bool, int | None]:
    """(ok, index of the first bad line or None)."""
    prev = GENESIS
    for i, rec in enumerate(read(path)):
        claimed = rec.get("hash")
        body = {k: v for k, v in rec.items() if k != "hash"}
        if rec.get("prev") != prev or _digest(body) != claimed:
            return False, i
        prev = claimed
    return True, None
