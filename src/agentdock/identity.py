"""Who is calling: the people directory, API keys and the caller attached to a run.

The server puts the authenticated user into ``config["configurable"]["langgraph_auth_user"]``.
Agents never parse headers themselves; they call :func:`current_user` and get a
:class:`Caller` with a role they can check.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


@dataclass(frozen=True)
class Caller:
    identity: str
    role: str
    team: str = ""
    name: str = ""
    manager: str | None = None
    scopes: tuple[str, ...] = ()
    via: str | None = None  # service that delegated on the caller's behalf

    def as_auth_dict(self) -> dict[str, Any]:
        """Shape returned from the auth handler (becomes the server's user object)."""
        return {
            "identity": self.identity,
            "display_name": self.name or self.identity,
            "is_authenticated": True,
            "permissions": list(self.scopes),
            "role": self.role,
            "team": self.team,
            "manager": self.manager,
            "via": self.via,
        }


@dataclass
class Directory:
    users: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get(self, identity: str) -> Caller | None:
        rec = self.users.get(identity)
        if rec is None:
            return None
        return Caller(
            identity=identity,
            role=rec["role"],
            team=rec.get("team", ""),
            name=rec.get("name", ""),
            manager=rec.get("manager"),
            scopes=tuple(rec.get("scopes", ())),
        )

    def reports_of(self, manager_id: str) -> set[str]:
        return {uid for uid, r in self.users.items() if r.get("manager") == manager_id}


@lru_cache(maxsize=1)
def directory() -> Directory:
    raw = json.loads((DATA / "directory.json").read_text())
    return Directory(users=raw["users"])


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


@lru_cache(maxsize=1)
def key_table() -> dict[str, str]:
    """sha256(api key) -> identity. Only hashes are stored on disk.

    ``AGENTDOCK_KEYS_FILE`` points at the table; the default is the dev table
    in ``config/dev_keys.json``, whose plaintext keys are listed in the README.
    """
    path = Path(os.environ.get("AGENTDOCK_KEYS_FILE", ROOT / "config" / "dev_keys.json"))
    return json.loads(path.read_text())["keys"]


def caller_for_key(key: str | None) -> Caller | None:
    if not key:
        return None
    identity = key_table().get(hash_key(key))
    return directory().get(identity) if identity else None


def _read(user: Any, attr: str, default: Any = None) -> Any:
    if user is None:
        return default
    if isinstance(user, Mapping):
        return user.get(attr, default)
    try:
        return user[attr]
    except (KeyError, TypeError, IndexError):
        return getattr(user, attr, default)


def current_user(config: Mapping[str, Any] | None) -> Caller | None:
    """The authenticated caller for this run, or None for unauthenticated local runs."""
    user = ((config or {}).get("configurable") or {}).get("langgraph_auth_user")
    identity = _read(user, "identity")
    if not identity:
        return None
    role = _read(user, "role")
    if role is None:  # e.g. Studio user: fall back to the directory
        found = directory().get(identity)
        return found
    return Caller(
        identity=identity,
        role=role,
        team=_read(user, "team", "") or "",
        name=_read(user, "display_name", "") or "",
        manager=_read(user, "manager"),
        scopes=tuple(_read(user, "permissions", ()) or ()),
        via=_read(user, "via"),
    )


def run_config_for(identity: str, **extra: Any) -> dict[str, Any]:
    """Config for invoking a graph in-process as a given person (evals, tests, A2A loopback)."""
    caller = directory().get(identity)
    if caller is None:
        raise KeyError(f"unknown identity {identity!r}")
    return {"configurable": {"langgraph_auth_user": caller.as_auth_dict(), **extra}}
