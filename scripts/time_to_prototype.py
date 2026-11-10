"""Measure: `dock new` -> gate -> live server -> answered over MCP and A2A.

Creates a throwaway agent, serves it on a spare port, calls it as E100, then
removes it. Prints each step's wall-clock time.

    uv run python scripts/time_to_prototype.py
"""

from __future__ import annotations

import json
import os
import signal
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
NAME = "hello_bot"
PORT = 2077
BASE = f"http://127.0.0.1:{PORT}"
DOCK = [sys.executable, "-m", "agentdock.cli"]


def step(label: str, t0: float) -> float:
    now = time.perf_counter()
    print(f"  {label:44} {now - t0:6.2f}s")
    return now


def main() -> int:
    agent_dir = ROOT / "agents" / NAME
    if agent_dir.exists():
        shutil.rmtree(agent_dir)
    try:
        httpx.get(f"{BASE}/ok", timeout=1)
        print(f"port {PORT} is already serving; stop it first")
        return 1
    except httpx.TransportError:
        pass
    start = t = time.perf_counter()
    server = None
    try:
        subprocess.run([*DOCK, "new", NAME, "--kind", "react", "--team", "demo"], cwd=ROOT, check=True, capture_output=True)
        t = step("dock new (scaffold + register)", t)
        server = subprocess.Popen(
            [*DOCK, "serve", "--port", str(PORT), "--no-reload"], cwd=ROOT,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,  # own process group, so the langgraph child dies with it
        )
        with httpx.Client(timeout=30) as c:
            while True:
                try:
                    if c.get(f"{BASE}/ok").status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                if server.poll() is not None:
                    print("server exited (gate failed?)")
                    return 1
                time.sleep(0.2)
            t = step("dock serve (check + eval gate + server up)", t)

            h = {"x-api-key": "dk_dev_e100", "accept": "application/json, text/event-stream"}
            r = c.post(f"{BASE}/mcp", headers=h, json={
                "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": {"name": NAME, "arguments": {"messages": [{"role": "user", "content": "who am I?"}]}},
            })
            body = r.text
            if body.startswith(("event:", "data:")):
                body = next(l[5:] for l in body.splitlines() if l.startswith("data:"))
            mcp_ok = "E100 (employee)" in json.dumps(json.loads(body))
            t = step(f"MCP tools/call answered as E100: {mcp_ok}", t)

            aid = c.post(f"{BASE}/assistants/search", json={"graph_id": NAME}, headers=h).json()[0]["assistant_id"]
            r = c.post(f"{BASE}/a2a/{aid}", headers=h, json={
                "jsonrpc": "2.0", "id": 2, "method": "message/send",
                "params": {"message": {"role": "user", "messageId": uuid.uuid4().hex, "parts": [{"kind": "text", "text": "who am I?"}]}},
            })
            a2a_ok = "E100 (employee)" in r.text
            t = step(f"A2A message/send answered as E100: {a2a_ok}", t)
        print(f"  {'TOTAL: new agent to live MCP + A2A':44} {time.perf_counter() - start:6.2f}s")
        return 0 if mcp_ok and a2a_ok else 1
    finally:
        if server:
            os.killpg(server.pid, signal.SIGTERM)
            server.wait(timeout=15)
        shutil.rmtree(agent_dir, ignore_errors=True)
        subprocess.run([*DOCK, "check"], cwd=ROOT, capture_output=True)  # regenerate langgraph.json


if __name__ == "__main__":
    raise SystemExit(main())
