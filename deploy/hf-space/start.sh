#!/bin/sh
# Container entrypoint (Render / any single-container host): Postgres + Redis inside the container, then the
# LangGraph Agent Server (production runtime). Works as root or as uid 1000.
set -e
# Keys pasted into a host's multi-line value box often carry a trailing newline, which makes the
# HTTP header invalid ("Illegal header value"). Strip whitespace from every key-like variable.
for v in LANGSMITH_API_KEY GEMINI_API_KEY LANGGRAPH_CLOUD_LICENSE_KEY CLOUDFLARE_AI_TOKEN CLOUDFLARE_ACCOUNT_ID AGENTDOCK_MODEL; do
  eval "val=\${$v:-}"
  if [ -n "$val" ]; then
    export "$v=$(printf '%s' "$val" | tr -d '[:space:]')"
  fi
done
PGBIN=$(ls -d /usr/lib/postgresql/*/bin | tail -1)
export PATH="$PGBIN:$PATH"
PGDATA=/tmp/pgdata
if [ ! -s "$PGDATA/PG_VERSION" ]; then
  initdb -D "$PGDATA" -U agentdock --auth=trust >/dev/null
fi
pg_ctl -D "$PGDATA" -o "-c listen_addresses=127.0.0.1 -c unix_socket_directories=/tmp" -l /tmp/postgres.log start -w
createdb -h 127.0.0.1 -U agentdock agentdock 2>/dev/null || true
redis-server --daemonize yes --dir /tmp --bind 127.0.0.1 --save "" --appendonly no
# A2A loopback must hit this server's own port, whatever the host assigns.
export AGENTDOCK_A2A_URL="${AGENTDOCK_A2A_URL:-http://127.0.0.1:$PORT}"
echo "[agentdock] postgres + redis up; starting Agent Server on :$PORT"
# On a tiny CPU share the server's license check (an HTTPS call to LangSmith made while everything else
# is still starting) can time out, and the server exits with status 3. Retry that case a few times.
n=0
while :; do
  /storage/entrypoint.sh &
  child=$!
  python /deps/agentdock/deploy/hf-space/warm.py &
  trap 'kill -TERM $child 2>/dev/null' TERM INT
  wait $child
  code=$?
  n=$((n + 1))
  if [ "$code" -ne 3 ] || [ "$n" -ge 5 ]; then
    exit "$code"
  fi
  echo "[agentdock] server exited during startup (status 3), usually the license check timing out on a slow CPU; retry $n of 4"
  sleep 5
done
