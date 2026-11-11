"""Warm the server once it is healthy: load every graph and build its schemas.

On a 0.1-CPU free plan the first schema request per agent takes minutes because it
imports the model clients and compiles the graph. MCP tools/list needs all of them,
so without this the first visitor's request times out.
"""
import json
import os
import time
import urllib.request

base = f"http://127.0.0.1:{os.environ.get('PORT', '8000')}"
key = os.environ.get("AGENTDOCK_SERVICE_KEY", "")


def call(method, path, body=None, timeout=600):
    req = urllib.request.Request(base + path, method=method, data=None if body is None else json.dumps(body).encode(),
                                 headers={"x-api-key": key, "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"null")


for _ in range(180):
    try:
        urllib.request.urlopen(base + "/ok", timeout=5)
        break
    except Exception:
        time.sleep(5)
start = time.time()
try:
    for a in call("POST", "/assistants/search", {"limit": 50}):
        call("GET", f"/assistants/{a['assistant_id']}/schemas")
    print(f"[agentdock] warmed all agents in {time.time() - start:.0f}s", flush=True)
except Exception as e:  # warming is best effort
    print(f"[agentdock] warm-up skipped: {e}", flush=True)
