#!/usr/bin/env bash
# Assemble the serverless bundle and deploy it to Vercel.
#   scripts/deploy_vercel.sh            # build bundle + deploy --prod
#   scripts/deploy_vercel.sh --bundle   # build bundle only (prints its path)
# The bundle is a plain copy (no .git), so the deploy never carries repo metadata.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${AGENTDOCK_BUNDLE_DIR:-$ROOT/.vercel-bundle/agentdock}"  # folder name = Vercel project name
rm -rf "$OUT"
mkdir -p "$OUT/evals"
cp -R "$ROOT/deploy/vercel/." "$OUT/"
cp -R "$ROOT/src" "$ROOT/agents" "$ROOT/data" "$ROOT/config" "$OUT/"
cp "$ROOT/evals/baseline.json" "$OUT/evals/"
find "$OUT" -name __pycache__ -type d -prune -exec rm -rf {} +
rm -rf "$OUT/src/agentdock/templates" "$OUT/public/.hallmark"
if [[ -f "$ROOT/deploy/vercel-project.json" ]]; then
  mkdir -p "$OUT/.vercel" && cp "$ROOT/deploy/vercel-project.json" "$OUT/.vercel/project.json"
fi
echo "bundle: $OUT"
[[ "${1:-}" == "--bundle" ]] && exit 0
cd "$OUT"
vercel deploy --prod --yes
if [[ -f .vercel/project.json && ! -f "$ROOT/deploy/vercel-project.json" ]]; then
  cp .vercel/project.json "$ROOT/deploy/vercel-project.json"
fi
