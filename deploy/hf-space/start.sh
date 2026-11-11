#!/bin/sh
# Container entrypoint (Render / any single-container host): Postgres + Redis inside the container, then the
# LangGraph Agent Server (production runtime). Works as root or as uid 1000.
set -e
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
exec /storage/entrypoint.sh
