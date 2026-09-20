#!/usr/bin/env bash
# PROMOTE v19 (2026-09-20) — James approved 11:35 CDT after v19a/b/c/d met every bar (C1 +6.6% x6 runs, fund C24 +8%,
# C16 p95 4 windows under bar, tools 64/64, replay tool-turn +4/+10%). Waits for the fidelity runner, then:
#   v19c EXP  -> dsv41-vllm-v19-nightly-hook-off54-fp8kv-ddf01704-BOUND-REF   (the :30006 reference)
#   v18 REF   -> dsv41-vllm-v18-cgsizes-RETIRED-REF                            (instant rollback: docker start)
# then starts v19, waits for bind, runs smoke + one knee, records. No rm. STOP file honoured.
set -uo pipefail
cd /home/milo/dsv41
CAMP=/home/milo/dsv41/overnight-campaign-2026-09-18.log; LEDGER=/home/milo/dsv41/results/ledger.md; STOPF=/home/milo/dsv41/STOP-CAMPAIGN
OLD=dsv41-vllm-v18-cgsizes-BOUND-REF; OLDR=dsv41-vllm-v18-cgsizes-RETIRED-REF
CAND=dsv41-vllm-v19c-nightly-hook-off54-fp8kv-ddf01704-EXP; NEW=dsv41-vllm-v19-nightly-hook-off54-fp8kv-ddf01704-BOUND-REF
D=/home/milo/dsv41/promote-v19-2026-09-20; mkdir -p "$D"
exec >> "$CAMP" 2>&1
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
led(){ printf '\n## %s — %s\nworker: milo-promote-v19 (nohup) · approved by James\n%s\n' "$(date '+%F %H:%M %Z')" "$1" "$2" >> "$LEDGER"; }
wait_bind(){ local n=$1 m=$2 i; for ((i=0;i<m*6;i++)); do curl -s -m 2 http://127.0.0.1:30006/v1/models 2>/dev/null | grep -q dsv41-flash-uva && { log "BOUND $n after $((i*10))s"; return 0; }; docker ps --format '{{.Names}}' | grep -qx "$n" || { log "DIED $n"; return 1; }; sleep 10; done; log "TIMEOUT $n"; return 1; }
for ((i=0;i<720;i++)); do pgrep -f "bash .*fidelity-runner" >/dev/null || break; sleep 10; done
log "===== PROMOTE v19 START ====="
[[ -f $STOPF ]] && { log "STOP file — abort, nothing renamed"; exit 0; }
docker ps -a --format '{{.Names}}' | grep -qx "$CAND" || { log "candidate $CAND missing — abort"; exit 3; }
docker ps -a --format '{{.Names}}' | grep -qx "$NEW" && { log "$NEW already exists — abort"; exit 5; }
for n in $(docker ps --format '{{.Names}}' | grep '^dsv41-vllm-'); do log "stop-and-keep $n"; docker stop "$n" >/dev/null; done; sleep 3
docker rename "$OLD" "$OLDR" && log "renamed $OLD -> $OLDR"
docker rename "$CAND" "$NEW" && log "renamed $CAND -> $NEW"
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
docker start "$NEW" >/dev/null
if wait_bind "$NEW" 25; then
  docker logs "$NEW" 2>&1 | grep -E 'KV cache format|HBM_expert=|GPU KV cache size|Loaded [0-9]+ configs' | sed 's/^.*\] //' | cut -c1-140 | tail -4 | tee "$D/boot.txt"
  bash /home/milo/dsv41/smoke_vllm.sh 2>&1 | tail -6 | tee "$D/smoke.txt"
  bash knee.sh v19ref-1 > "$D/knee-v19ref-1.log" 2>&1; cp -f knee-v19ref-1.json "$D/"; grep -h 'conc= 1:\|conc=16:' "$D/knee-v19ref-1.log" | cut -c1-80
  docker inspect "$NEW" --format '{{.Config.Image}} {{.Image}}' > "$D/image.txt"; docker ps -a --format '{{.Names}}\t{{.Status}}' | grep -E 'v19|v18' > "$D/containers.txt"
  led "v19 PROMOTED to :30006" "$NEW live; $OLDR stopped-and-kept (rollback = docker stop $NEW && docker start $OLDR). Nightly dee37d89 + v15 hook + off54 + fp8_ds_mla + pinned ddf01704 autotune set. Bars: C1 183 (+6.6%, 6 runs), fund C24 +8%, C16 p95 ok x4, tools 64/64, replay tool-turn +4/+10%. Receipts results/2026-09-20-v19-pair-async + v19c/v19d + $D."
  log "===== PROMOTE v19 END (live) ====="
else
  log "v19 failed to bind — rolling back names"; docker stop "$NEW" >/dev/null 2>&1; docker rename "$NEW" "$CAND"; docker rename "$OLDR" "$OLD"; docker start "$OLD" >/dev/null; wait_bind "$OLD" 15
  led "v19 promotion FAILED to bind — rolled back to v18" "$(docker logs $NEW 2>&1 | tail -3 | cut -c1-200)"; log "===== PROMOTE v19 END (rolled back) ====="
fi
