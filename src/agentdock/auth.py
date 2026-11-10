"""Server auth: every route (REST, /mcp, /a2a) goes through these handlers.

- Callers send ``x-api-key: <key>`` or ``Authorization: Bearer <key>``.
- A service with the ``delegate`` scope may add ``x-on-behalf-of: <identity>``;
  the run then executes as that person, with ``via`` set to the service. This is
  how one agent calls another over A2A without gaining extra rights.
- Threads and runs are owner-scoped: nobody can read another person's
  conversation, including through MCP or A2A.
"""

from __future__ import annotations

from typing import Any

from langgraph_sdk import Auth

from agentdock.identity import caller_for_key, directory

auth = Auth()


def _header(headers: Any, name: str) -> str | None:
    for k, v in (headers or {}).items():
        key = k.decode() if isinstance(k, bytes) else str(k)
        if key.lower() == name:
            return v.decode() if isinstance(v, bytes) else str(v)
    return None


def resolve(headers: Any) -> dict[str, Any]:
    """Pure function behind the handler so it can be unit-tested without a server."""
    key = _header(headers, "x-api-key")
    if key is None:
        bearer = _header(headers, "authorization") or ""
        if bearer.lower().startswith("bearer "):
            key = bearer[7:].strip()
    caller = caller_for_key(key)
    if caller is None:
        raise Auth.exceptions.HTTPException(status_code=401, detail="Missing or unknown API key")

    obo = _header(headers, "x-on-behalf-of")
    if obo:
        if "delegate" not in caller.scopes:
            raise Auth.exceptions.HTTPException(
                status_code=403, detail=f"{caller.identity} may not act on behalf of others"
            )
        person = directory().get(obo)
        if person is None or person.role == "service":
            raise Auth.exceptions.HTTPException(status_code=403, detail="Unknown delegated identity")
        return {**person.as_auth_dict(), "via": caller.identity}
    return caller.as_auth_dict()


@auth.authenticate
async def authenticate(headers: dict) -> Auth.types.MinimalUserDict:
    return resolve(headers)


@auth.on.threads
async def own_threads(ctx: Auth.types.AuthContext, value: dict) -> dict | bool:
    """Stamp new threads with their owner and only ever show a caller their own.

    The one exception is the retention sweeper, which may search and delete any
    thread and nothing else.
    """
    if "retention" in (getattr(ctx.user, "permissions", None) or ()):
        if ctx.action in ("search", "delete"):
            return True
        raise Auth.exceptions.HTTPException(status_code=403, detail="svc-retention may only search and delete")
    if ctx.action == "create" or "metadata" in value:
        # Mutate the server's dict in place: it keeps its own reference, so
        # replacing an empty {} with a new dict would silently drop the owner.
        metadata = value.get("metadata")
        if metadata is None:
            metadata = value["metadata"] = {}
        metadata["owner"] = ctx.user.identity
    return {"owner": ctx.user.identity}


@auth.on.assistants
async def assistants_read_only(ctx: Auth.types.AuthContext, value: dict) -> bool:
    """Anyone authenticated may list and read agents; only the platform registers them."""
    if ctx.action in ("read", "search"):
        return True
    raise Auth.exceptions.HTTPException(status_code=403, detail="Agents are registered by the platform")


@auth.on.store
async def own_store(ctx: Auth.types.AuthContext, value: dict) -> None:
    namespace = tuple(value.get("namespace") or ())
    if not namespace or namespace[0] != ctx.user.identity:
        raise Auth.exceptions.HTTPException(status_code=403, detail="Store namespace must start with your identity")
