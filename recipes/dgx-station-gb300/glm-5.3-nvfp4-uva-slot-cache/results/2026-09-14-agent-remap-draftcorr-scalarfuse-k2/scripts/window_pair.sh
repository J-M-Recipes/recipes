#!/usr/bin/env bash
# window_pair.sh <lane-name> — measure the CURRENT :30001 lane: wait ready, warm, greedy x1, speed_reps (C1 512 x3), needle ladder 32K/64K/128K (prefill tok/s), slot-cache stats. Read-only vs server.
set -eu
LANE="$1"; OUT=~/window-20260914/$LANE; mkdir -p "$OUT"
S=~/window-20260913/scripts
export BASE_URL=http://127.0.0.1:30001 MODEL=glm-5.3-big API_KEY="$(cat ~/.glm_api_key)" LANE
log(){ echo "$(date '+%F %T') $*" | tee -a "$OUT/window.log"; }
for i in $(seq 1 360); do
  code=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $API_KEY" $BASE_URL/v1/models || true)
  [ "$code" = 200 ] && break; sleep 10
done
[ "$code" = 200 ] || { log "NOT READY after 60min"; exit 2; }
log "READY $LANE $(docker ps --format '{{.Names}}' | grep glm53-big | head -1)"
curl -s -H "Authorization: Bearer $API_KEY" -H 'Content-Type: application/json' $BASE_URL/v1/chat/completions \
  -d '{"model":"glm-5.3-big","max_tokens":256,"temperature":0,"messages":[{"role":"user","content":"Warm up: describe a lighthouse in 150 words."}]}' >/dev/null
log "warm done"
python3 $S/greedy_equiv.py "$OUT/greedy-r1.json" | tee -a "$OUT/window.log"
python3 $S/speed_reps.py "$OUT/speed-reps.json" | tee -a "$OUT/window.log"
BASE_URL=http://127.0.0.1:30001/v1 RUNGS="32000 64000 128000" STOP_ON_FAIL=0 python3 $S/needle_ladder.py "$OUT/needle-ladder.jsonl" | tee -a "$OUT/window.log" || true
docker logs --since 30m "$(docker ps --format '{{.Names}}' | grep glm53-big | head -1)" 2>&1 | tr '\r' '\n' | grep "SLOT_CACHE STATS" | tail -40 > "$OUT/slotcache-stats.txt" || true
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "$OUT/nvidia-smi.txt"
log "DONE $LANE"
