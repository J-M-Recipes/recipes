#!/usr/bin/env bash
set -Eeuo pipefail
root=~/glm53-cq-20261004; cd "$root"
log() { echo "[$(date -Is)] $*"; }
trap "rc=\$?; if (( rc != 0 )); then docker rm -f glm53-cq-rank0 >/dev/null 2>&1 || true; ssh -o BatchMode=yes milo@<rdma-peer-2> docker rm -f glm53-cq-rank1 >/dev/null 2>&1 || true; log CLEANUP_RAN rc=\$rc; fi" EXIT
log "CQ_START"
until grep -q COPY_END image-copy.log; do sleep 15; done
grep -E "PIPE_RC|sha256" image-copy.log | tr "\n" " "; echo
grep -q "PIPE_RC=0" image-copy.log
[[ -z "$(docker ps -q)" && -z "$(ssh -o BatchMode=yes milo@<rdma-peer-2> docker ps -q)" ]] || { log REFUSE_RUNNING; exit 3; }
rsync -aH --exclude raw/ --exclude "*.log" "$root/" "milo@<rdma-peer-2>:$root/"
log "LAUNCH_cq_BEGIN"
set +e; bash "$root/launch-candidate-cq.sh" cq > "$root/launch-cq.log" 2>&1; lrc=$?; set -e
log "LAUNCH_cq_DONE rc=$lrc"
if (( lrc != 0 )); then log BOOT_FAIL; docker logs --tail 300 glm53-cq-rank0 > rank0-fail-cq.log 2>&1 || true; exit 1; fi
docker logs glm53-cq-rank0 2>&1 | grep -n -i -E "draft|dflash|speculat|DRAFTER-GROUP|mask" | grep -v -E "non-default args|Initializing a V1" | cut -c1-300 > boot-draft-cq.txt || true
log "MATRIX_cq_BEGIN"
python3 "$root/diag_matrix.py" --mode dflash --out "$root/raw/cq" > "$root/matrix-cq.log" 2>&1
log "MATRIX_cq_DONE"
log "G1_cq_BEGIN"
python3 "$root/g1_regimes.py" "$root/raw/cq-g1" > "$root/g1-cq.log" 2>&1 || log "G1_cq_FAILED"
log "G1_cq_DONE"
docker rm -f glm53-cq-rank0 || true; ssh -o BatchMode=yes milo@<rdma-peer-2> "docker rm -f glm53-cq-rank1" || true; sleep 5
log "MEM_AFTER head=$(awk "/MemAvailable:/ {print \$2}" /proc/meminfo)"
log "RUNNER_DONE"
