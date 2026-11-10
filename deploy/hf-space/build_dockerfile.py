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
    "# Hugging Face runs Spaces as uid 1000",
    "RUN (id -u 1000 >/dev/null 2>&1 || useradd -m -u 1000 user) && chown -R 1000:1000 /deps",
    "ENV PORT=7860 \\",
    "    DATABASE_URI=postgres://agentdock@127.0.0.1:5432/agentdock?sslmode=disable \\",
    "    REDIS_URI=redis://127.0.0.1:6379 \\",
    "    N_JOBS_PER_WORKER=8 \\",
    "    AGENTDOCK_A2A_URL=http://127.0.0.1:7860 \\",
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
