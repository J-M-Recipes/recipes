#!/bin/bash
# Stop the Kimi-K3 four-tier pod: engine on the Station, nine workers. Nothing is installed system-wide.
#   SPARK_HOSTS="s1 ... s8" STATION=station bash stop-pod.sh [--dry-run]
set -uo pipefail
: "${SPARK_HOSTS:?set SPARK_HOSTS}" "${STATION:?set STATION}"
DRY=${1:-}
run() { if [ "$DRY" = --dry-run ]; then echo "would run on $1: $2"; else ssh -o BatchMode=yes "$1" "$2"; fi; }
run "$STATION" 'docker ps --format "{{.Names}}" | grep -E "^k3-pod-n[0-9]+$" | xargs -r docker rm -f; docker rm -f k3-worker-rtx 2>/dev/null; true'
for h in $SPARK_HOSTS; do run "$h" 'docker rm -f k3-worker 2>/dev/null; true'; done
echo "pod stopped; restore the previous Station lane with its own recipe's launcher"
