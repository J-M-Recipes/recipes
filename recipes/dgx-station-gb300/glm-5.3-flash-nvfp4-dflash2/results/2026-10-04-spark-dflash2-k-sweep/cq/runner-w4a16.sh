#!/usr/bin/env bash
set -Eeuo pipefail
root=~/glm53-cq-20261004; cd "$root"
log() { echo "[$(date -Is)] $*"; }
trap "rc=\$?; if (( rc != 0 )); then docker rm -f glm53-cq-rank0 >/dev/null 2>&1 || true; ssh -o BatchMode=yes milo@<rdma-peer-2> docker rm -f glm53-cq-rank1 >/dev/null 2>&1 || true; log CLEANUP_RAN rc=\$rc; fi" EXIT
log "W4A16_START"
until grep -q FETCH_DONE fetch-w4a16.log; do sleep 20; done
grep -E "worker_ok_files" fetch-w4a16.log
ok=$(grep -oE "worker_ok_files=[0-9]+" fetch-w4a16.log | cut -d= -f2); (( ok >= 13 )) || { log "VERIFY_SHORT ok=$ok"; exit 2; }
[[ -z "$(docker ps -q)" && -z "$(ssh -o BatchMode=yes milo@<rdma-peer-2> docker ps -q)" ]] || { log REFUSE_RUNNING; exit 3; }
rsync -aH --exclude raw/ --exclude "*.log" "$root/" "milo@<rdma-peer-2>:$root/"
log "LAUNCH_w4a16_BEGIN"
set +e; bash "$root/launch-candidate-cq.sh" w4a16 > "$root/launch-w4a16.log" 2>&1; lrc=$?; set -e
log "LAUNCH_w4a16_DONE rc=$lrc"
if (( lrc != 0 )); then log BOOT_FAIL; docker logs --tail 300 glm53-cq-rank0 > rank0-fail-w4a16.log 2>&1 || true; exit 1; fi
docker logs glm53-cq-rank0 2>&1 | grep -n -i -E "draft|dflash|speculat|DRAFTER-GROUP|quant|marlin|moe" | grep -v -E "non-default args|Initializing a V1" | cut -c1-300 > boot-w4a16.txt || true
log "MATRIX_w4a16_BEGIN"
python3 "$root/diag_matrix.py" --mode dflash --out "$root/raw/w4a16" > "$root/matrix-w4a16.log" 2>&1
log "MATRIX_w4a16_DONE"
log "G1_w4a16_BEGIN"
python3 "$root/g1_regimes.py" "$root/raw/w4a16-g1" > "$root/g1-w4a16.log" 2>&1 || log "G1_w4a16_FAILED"
log "G1_w4a16_DONE"
docker rm -f glm53-cq-rank0 || true; ssh -o BatchMode=yes milo@<rdma-peer-2> "docker rm -f glm53-cq-rank1" || true; sleep 5
log "MEM_AFTER head=$(awk "/MemAvailable:/ {print \$2}" /proc/meminfo)"
log "RUNNER_DONE"
