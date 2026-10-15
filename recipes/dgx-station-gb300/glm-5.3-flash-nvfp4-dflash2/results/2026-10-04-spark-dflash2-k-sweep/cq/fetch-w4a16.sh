#!/usr/bin/env bash
set -Eeuo pipefail
log() { echo "[$(date -Is)] $*"; }
D=~/models/GLM-5.3-Flash-W4A16-MTP-5681f7c
R=~/glm53-cq-20261004
attempt=0
while :; do
  attempt=$((attempt+1)); log "DOWNLOAD_ATTEMPT $attempt"
  ~/.local/bin/hf download canada-quant/GLM-5.3-Flash-W4A16-MTP --revision 5681f7c2dc6436d63f82ab6fb892b4f5062b8304 --local-dir "$D" > "$R/hf-attempt-$attempt.log" 2>&1 &
  pid=$!
  last=$(du -sb "$D" | cut -f1); stalled=0
  while kill -0 $pid 2>/dev/null; do
    sleep 60
    now=$(du -sb "$D" | cut -f1)
    if (( now > last )); then stalled=0; else stalled=$((stalled+1)); fi
    last=$now
    log "PROGRESS bytes=$now stalled_min=$stalled"
    if (( stalled >= 5 )); then log "STALL_KILL attempt=$attempt"; kill $pid 2>/dev/null || true; sleep 5; kill -9 $pid 2>/dev/null || true; break; fi
  done
  if wait $pid 2>/dev/null; then log "DOWNLOAD_DONE attempt=$attempt"; break; fi
  (( attempt >= 12 )) && { log "GIVE_UP"; exit 1; }
  sleep 10
done
inc=$(find "$D" -name "*.incomplete" | wc -l); log "incomplete_files=$inc"; (( inc == 0 ))
du -sh "$D"
log "SHA_BEGIN"; (cd "$D" && sha256sum *.safetensors config.json model.safetensors.index.json > "$R/w4a16-sha256.txt"); log "SHA_DONE"
log "RSYNC_BEGIN"; rsync -aH --exclude .cache "$D/" "milo@<rdma-peer-2>:$D/"; log "RSYNC_DONE"
log "VERIFY_BEGIN"; ssh -o BatchMode=yes milo@<rdma-peer-2> "cd $D && sha256sum -c -" < "$R/w4a16-sha256.txt" | grep -c ": OK" | xargs -I{} echo "worker_ok_files={}"; log "VERIFY_DONE"
log "FETCH_DONE"
