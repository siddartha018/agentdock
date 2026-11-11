#!/usr/bin/env bash
# Regenerates deploy/render/Dockerfile (production runtime + Postgres + Redis in one image).
# Run after adding or renaming agents; CI fails if it's stale.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp)"
cd "$ROOT"
uv run langgraph dockerfile -c langgraph.json "$TMP" >/dev/null
python3 deploy/hf-space/build_dockerfile.py "$TMP" deploy/render/Dockerfile >/dev/null
rm -f "$TMP"
echo "deploy/render/Dockerfile"
