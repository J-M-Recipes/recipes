#!/usr/bin/env bash
set -Eeuo pipefail
root=~/glm53-k-20261004; cd "$root"
log() { echo "[$(date -Is)] $*"; }
trap "rc=\$?; if (( rc != 0 )); then docker rm -f glm53-k-rank0 >/dev/null 2>&1 || true; ssh -o BatchMode=yes milo@<rdma-peer-2> docker rm -f glm53-k-rank1 >/dev/null 2>&1 || true; log CLEANUP_RAN; fi" EXIT
log "G1_START"
rsync -aH --exclude raw/ --exclude "*.log" "$root/" "milo@<rdma-peer-2>:$root/"
log "LAUNCH_k7_BEGIN"
bash "$root/launch-candidate-k.sh" k7 > "$root/launch-k7.log" 2>&1
log "LAUNCH_k7_DONE"
log "G1_BEGIN"
python3 "$root/g1_regimes.py" "$root/raw/g1" > "$root/g1.log" 2>&1
log "G1_DONE"
docker rm -f glm53-k-rank0 || true; ssh -o BatchMode=yes milo@<rdma-peer-2> "docker rm -f glm53-k-rank1" || true; sleep 5
log "MEM_AFTER head=$(awk "/MemAvailable:/ {print \$2}" /proc/meminfo)"
log "RUNNER_DONE"
