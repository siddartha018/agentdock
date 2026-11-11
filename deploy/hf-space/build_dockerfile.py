"""Turn `langgraph dockerfile` output into the Space's Dockerfile (Postgres + Redis + uid 1000)."""
import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text()
lines = src.splitlines()
assert lines[0].startswith("FROM langchain/langgraph-api"), lines[0]
head = [lines[0], "", "# Postgres + Redis run inside the Space container",
        "RUN apt-get update && apt-get install -y --no-install-recommends postgresql redis-server && rm -rf /var/lib/apt/lists/*"]
tail = [
    "",
    "# Precompile bytecode: the server runs as uid 1000 and can't write .pyc files into root-owned",
    "# site-packages, so without this every start recompiles everything (minutes on a 0.1-CPU free plan).",
    "RUN python -m compileall -q -j 0 $(python -c 'import site; print(site.getsitepackages()[0])') /deps/agentdock/src > /dev/null || true",
    "",
    "# Hugging Face runs Spaces as uid 1000",
    "RUN (id -u 1000 >/dev/null 2>&1 || useradd -m -u 1000 user) && chown -R 1000:1000 /deps",
    "ENV PORT=7860 \\",
    "    DATABASE_URI=postgres://agentdock@127.0.0.1:5432/agentdock?sslmode=disable \\",
    "    REDIS_URI=redis://127.0.0.1:6379 \\",
    "    N_JOBS_PER_WORKER=8 \\",
    "    AGENTDOCK_SERVICE_KEY=dk_dev_svc-concierge \\",
    "    AGENTDOCK_RETENTION_KEY=dk_dev_svc-retention \\",
    "    AGENTDOCK_AUDIT_LOG=/tmp/agentdock-audit.jsonl \\",
    "    AGENTDOCK_FALLBACK=offline",
    "USER 1000",
    "EXPOSE 7860",
    'ENTRYPOINT ["/deps/agentdock/deploy/hf-space/start.sh"]',
]
Path(sys.argv[2]).write_text("\n".join(head + lines[1:] + tail) + "\n")
print("wrote", sys.argv[2])
