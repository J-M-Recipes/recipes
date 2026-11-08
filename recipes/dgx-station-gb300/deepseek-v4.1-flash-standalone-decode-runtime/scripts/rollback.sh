#!/usr/bin/env bash
# rollback.sh - stop the standalone decode runtime's container and hand the GB300 back.
#
# There is nothing to roll back in a serving sense: this recipe opens no port and replaced no lane. Since the
# 2026-11-06 cutover (repo CUTOVER.md) the Station is the project's dev box and production DSV4.1 runs on Sparks 1-4;
# the former Station lane (Pg61, sibling recipe deepseek-v4.1-flash-vllm-uva-dspark-sidecar) is stopped-and-kept.
# This script only (1) stops a running headline container and (2) prints how to bring the sibling lane back.
#
#   bash scripts/rollback.sh --dry-run     # prints what it would do; needs no docker
#   NAME=dsv41-decode bash scripts/rollback.sh
#
# bash -n and --dry-run only: not exercised against a live container by the recipe writer.
set -euo pipefail
NAME=${NAME:-dsv41-decode}
DRY=0; [[ "${1:-}" == "--dry-run" ]] && DRY=1

run() { if (( DRY )); then echo "[dry-run] $*"; else "$@"; fi; }

if (( DRY )); then
  echo "[dry-run] would: docker ps --filter name=^/${NAME}\$ ; docker stop ${NAME} (the run is --rm, so the container disappears)"
else
  if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    docker stop "$NAME"
  else
    echo "no running container named $NAME"
  fi
fi
run echo "GPU should now hold no decode-runtime memory: check nvidia-smi (the runtime pins ~260 GiB of Grace memory while it runs; it is freed on exit)."
cat <<'EOF'
To restore the former Station serving lane (a separate recipe; this one never touched it):
  recipes/dgx-station-gb300/deepseek-v4.1-flash-vllm-uva-dspark-sidecar - see its README '## Rollback' and scripts/ (Pg61 launcher: launch-pg61.sh up),
  and the project CUTOVER.md rollback section (repoint consumers, re-arm the peer-tier watchdog, re-enable the lane check).
Do that only on James's say-so: the Station is the decode project's dev box until he releases it.
EOF
