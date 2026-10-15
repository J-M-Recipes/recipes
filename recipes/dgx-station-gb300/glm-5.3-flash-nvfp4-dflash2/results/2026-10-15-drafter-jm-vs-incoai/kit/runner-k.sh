#!/usr/bin/env bash
set -Eeuo pipefail
root=~/glm53-k-20261004; cd "$root"
log() { echo "[$(date -Is)] $*"; }
trap "rc=\$?; log RUNNER_ERR line=\$LINENO rc=\$rc" ERR
trap "rc=\$?; if (( rc != 0 )); then docker rm -f glm53-k-rank0 >/dev/null 2>&1 || true; ssh -o BatchMode=yes milo@<rdma-peer-2> docker rm -f glm53-k-rank1 >/dev/null 2>&1 || true; log CLEANUP_RAN; fi" EXIT
mem_head() { awk "/MemAvailable:/ {print \$2}" /proc/meminfo; }
mem_worker() { ssh -o BatchMode=yes milo@<rdma-peer-2> "awk /MemAvailable:/ {print $2} /proc/meminfo"; }
log "RUNNER_START $(uname -r)"
[[ -z "$(docker ps -q)" && -z "$(ssh -o BatchMode=yes milo@<rdma-peer-2> docker ps -q)" ]] || { log REFUSE_RUNNING; exit 3; }
log "MEM_HEAD $(mem_head) MEM_WORKER $(mem_worker) CLOCKS $(nvidia-smi --query-gpu=clocks.sm,clocks.max.sm --format=csv,noheader)"
rsync -aH --exclude raw/ --exclude "*.log" "$root/" "milo@<rdma-peer-2>:$root/"
for k in k4 k3; do
  [[ -f STOP ]] && { log STOP; exit 0; }
  log "LAUNCH_${k}_BEGIN"
  set +e; bash "$root/launch-candidate-k.sh" $k > "$root/launch-$k.log" 2>&1; lrc=$?; set -e
  log "LAUNCH_${k}_DONE rc=$lrc"
  if (( lrc != 0 )); then
    log "BOOT_FAIL $k"; docker logs --tail 200 glm53-k-rank0 > "$root/rank0-fail-$k.log" 2>&1 || true
    docker rm -f glm53-k-rank0 || true; ssh -o BatchMode=yes milo@<rdma-peer-2> "docker rm -f glm53-k-rank1" || true; sleep 5; continue
  fi
  curl -fsS http://127.0.0.1:8888/metrics | grep -E "per_pos" | head -3 > "$root/perpos-at-boot-$k.txt" || true
  log "MATRIX_${k}_BEGIN"
  python3 "$root/diag_matrix.py" --mode dflash --out "$root/raw/$k" > "$root/matrix-$k.log" 2>&1
  log "MATRIX_${k}_DONE"
  docker rm -f glm53-k-rank0 || true; ssh -o BatchMode=yes milo@<rdma-peer-2> "docker rm -f glm53-k-rank1" || true; sleep 5
  log "MEM_AFTER_${k} head=$(mem_head) worker=$(mem_worker)"
done
log "PIN_END $(sha256sum ~/models/GLM-5.3-Flash-DFlash2-7d74cdd/model.safetensors | cut -c1-16)"
log "RUNNER_DONE"
