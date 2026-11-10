"""Prove conversations are actually deleted after the retention TTL.

Starts an Agent Server with a 0.1-minute TTL and a 1-minute sweep, creates a
thread as E100, then polls until the sweeper removes it (or gives up at 3 min).

    uv run python scripts/check_retention.py            # dev (in-memory) runtime
    uv run python scripts/check_retention.py --url URL  # an already-running server
                                                         # started with a short TTL
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentdock.registry import discover, langgraph_config  # noqa: E402

H = {"x-api-key": "dk_dev_e100"}


def wait_up(base: str, proc: subprocess.Popen | None) -> None:
    for _ in range(300):
        try:
            if httpx.get(f"{base}/ok", timeout=1).status_code == 200:
                return
        except httpx.TransportError:
            pass
        if proc and proc.poll() is not None:
            raise SystemExit("server exited")
        time.sleep(0.2)
    raise SystemExit("server did not start")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    ap.add_argument("--port", type=int, default=2078)
    args = ap.parse_args()
    proc, cfg = None, ROOT / ".langgraph.retention-test.json"
    base = (args.url or f"http://127.0.0.1:{args.port}").rstrip("/")
    try:
        if not args.url:
            conf = langgraph_config(discover(), retention_minutes=0.1)
            conf["checkpointer"]["ttl"]["sweep_interval_minutes"] = 1
            conf.pop("env", None)
            cfg.write_text(json.dumps(conf))
            env = {**os.environ, "AGENTDOCK_RETENTION_MINUTES": "0.1", "AGENTDOCK_SWEEP_SECONDS": "5",
                   "AGENTDOCK_RETENTION_KEY": "dk_dev_svc-retention"}
            exe = str(Path(sys.executable).parent / "langgraph")
            log = open(os.environ.get("RETENTION_SERVER_LOG", os.devnull), "w")
            proc = subprocess.Popen(
                [exe, "dev", "--no-browser", "--no-reload", "--port", str(args.port), "--config", str(cfg)],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        wait_up(base, proc)
        with httpx.Client(timeout=30, headers=H) as c:
            t = c.post(f"{base}/threads", json={}).json()["thread_id"]
            c.post(f"{base}/threads/{t}/runs/wait", json={
                "assistant_id": "policy_qa",
                "input": {"messages": [{"role": "user", "content": "How many vacation days do I have left?"}]},
            })
            assert c.get(f"{base}/threads/{t}").status_code == 200
            start = time.time()
            print(f"thread {t} created; retention 6 s, platform sweep every 5 s")
            while time.time() - start < float(os.environ.get("RETENTION_TIMEOUT", 180)):
                code = c.get(f"{base}/threads/{t}").status_code
                if code == 404:
                    print(f"PASS: deleted by the sweeper after {time.time() - start:.0f} s")
                    return 0
                time.sleep(5)
        print("FAIL: thread still present after 180 s")
        return 1
    finally:
        if proc:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=15)
        cfg.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
