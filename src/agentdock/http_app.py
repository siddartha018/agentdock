"""Custom app mounted by the Agent Server (``http.app`` in langgraph.json).

It adds no routes; it exists for its lifespan, which runs the retention sweeper.
"""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager

from starlette.applications import Starlette

from agentdock import retention


@asynccontextmanager
async def lifespan(app: Starlette):
    task = asyncio.create_task(retention.run_forever())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


app = Starlette(lifespan=lifespan)
