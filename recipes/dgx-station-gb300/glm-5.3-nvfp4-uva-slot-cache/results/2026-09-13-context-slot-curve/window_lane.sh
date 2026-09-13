#!/usr/bin/env bash
# window_lane.sh <lane-name> — wait ready, warm, greedy x3 (self-repeat), 512-tok speed probe. Read-only vs server.
set -eu
LANE="$1"; OUT=~/window-20260913/$LANE; mkdir -p "$OUT"
S=~/window-20260913/scripts
export BASE_URL=http://127.0.0.1:30001/v1 MODEL=glm-5.3-big API_KEY="$(cat ~/.glm_api_key)"
log(){ echo "$(date '+%F %T') $*" | tee -a "$OUT/window.log"; }
# 1. wait ready (up to 60 min)
for i in $(seq 1 360); do
  code=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $API_KEY" http://127.0.0.1:30001/v1/models || true)
  [ "$code" = 200 ] && break; sleep 10
done
[ "$code" = 200 ] || { log "NOT READY after 60min"; exit 2; }
log "READY $LANE"
# 2. warm: one 256-tok prose, discarded
curl -s -H "Authorization: Bearer $API_KEY" -H 'Content-Type: application/json' $BASE_URL/chat/completions \
  -d '{"model":"glm-5.3-big","max_tokens":256,"temperature":0,"messages":[{"role":"user","content":"Warm up: describe a lighthouse in 150 words."}]}' >/dev/null
log "warm done"
# 3. greedy x3 (self-repeatability + cross-lane)
for r in 1 2 3; do python3 $S/greedy_equiv.py "$OUT/greedy-r$r.json" | tee -a "$OUT/window.log"; done
python3 $S/greedy_equiv.py --compare "$OUT/greedy-r1.json" "$OUT/greedy-r2.json" | tee "$OUT/selfrepeat-1v2.txt"
python3 $S/greedy_equiv.py --compare "$OUT/greedy-r1.json" "$OUT/greedy-r3.json" | tee "$OUT/selfrepeat-1v3.txt"
# 4. speed probe: 4 prompts x 512 tok, twice
API_KEY_FOR_METRICS="$(cat ~/.glm_api_key)" BASE_URL=http://127.0.0.1:30001 API_KEY_FILE=~/.glm_api_key python3 $S/dflash2_acceptance_probe.py --max-tokens 512 --out "$OUT/probe-512-a.json" | tee -a "$OUT/window.log"
API_KEY_FOR_METRICS="$(cat ~/.glm_api_key)" BASE_URL=http://127.0.0.1:30001 API_KEY_FILE=~/.glm_api_key python3 $S/dflash2_acceptance_probe.py --max-tokens 512 --out "$OUT/probe-512-b.json" | tee -a "$OUT/window.log"
# 5. slot-cache stats lines from container log (last 40)
docker logs --since 30m "$(docker ps --format '{{.Names}}' | grep glm53-big | head -1)" 2>&1 | tr '\r' '\n' | grep -i "SLOT_CACHE.*hit\|misses\|hit_rate" | tail -40 > "$OUT/slotcache-stats.txt" || true
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "$OUT/nvidia-smi.txt"
log "DONE $LANE"
