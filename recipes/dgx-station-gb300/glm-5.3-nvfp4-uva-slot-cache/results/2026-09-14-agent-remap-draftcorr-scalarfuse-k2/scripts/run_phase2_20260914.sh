#!/usr/bin/env bash
# phase 2: quality receipts + compounding gate. Live now: control-20260914 (daily, stock hook).
set -u
C=/home/milo/big-v1-campaign; L=$C/LEDGER.md; cd $C
S=~/window-20260913/scripts; W=~/window-20260914; mkdir -p $W
export MODEL=glm-5.3-big API_KEY="$(cat ~/.glm_api_key)"
cur(){ docker ps --format '{{.Names}}' | grep glm53-big | head -1; }
step(){ echo "$(date -Is) $*" >> $L; echo "$(date -Is) $*"; }
ready(){ for i in $(seq 1 360); do c=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $API_KEY" http://127.0.0.1:30001/v1/models || true); [ "$c" = 200 ] && return 0; sleep 10; done; return 2; }
warm(){ curl -s -H "Authorization: Bearer $API_KEY" -H 'Content-Type: application/json' http://127.0.0.1:30001/v1/chat/completions -d '{"model":"glm-5.3-big","max_tokens":256,"temperature":0,"messages":[{"role":"user","content":"Warm up: describe a lighthouse in 150 words."}]}' >/dev/null; }
greedy(){ BASE_URL=http://127.0.0.1:30001/v1 python3 $S/greedy_equiv.py "$1"; }
step "PHASE2 start; live=$(cur)"
# 1. control greedy (D lane live)
ready && warm && greedy $W/D-control/greedy.json && greedy $W/D-control/greedy-r2.json
python3 $S/greedy_equiv.py --compare $W/D-control/greedy.json $W/D-control/greedy-r2.json > $W/D-control/selfrepeat.txt; cat $W/D-control/selfrepeat.txt
# 2. K2 + scalar-fuse compounding lane
docker stop -t 60 "$(cur)"; sleep 5
./launch-k2-scalarfuse.sh && step "launched K2+scalarfuse (mtp K=2, hook v2 SCALAR_FUSE=1)"
mkdir -p $W/E-k2-scalarfuse; ready && warm && greedy $W/E-k2-scalarfuse/greedy.json
LANE=E-k2-scalarfuse BASE_URL=http://127.0.0.1:30001 python3 $S/speed_reps.py $W/E-k2-scalarfuse/speed-reps.json > $W/E-k2-scalarfuse/speed.log
docker logs "$(cur)" 2>&1 | tr '\r' '\n' | grep "SLOT_CACHE STATS" | tail -20 > $W/E-k2-scalarfuse/slotcache-stats.txt
python3 $S/greedy_equiv.py --compare $W/D-control/greedy.json $W/E-k2-scalarfuse/greedy.json > $W/E-k2-scalarfuse/greedy-vs-D.txt; cat $W/E-k2-scalarfuse/greedy-vs-D.txt
# 3. B scalar-fuse again: greedy vs D + teacher-forced margins on B server for any divergences
docker stop -t 60 "$(cur)"; sleep 5
docker start glm53-big-sc13g-mtp-ctx256k-scalarfuse-20260914 && step "restarted scalarfuse (B) for quality receipts"
ready && warm && greedy $W/B-scalarfuse/greedy.json
python3 $S/greedy_equiv.py --compare $W/D-control/greedy.json $W/B-scalarfuse/greedy.json > $W/B-scalarfuse/greedy-vs-D.txt; cat $W/B-scalarfuse/greedy-vs-D.txt
LANE=B-scalarfuse BASE_URL=http://127.0.0.1:30001 python3 $S/divergence_margin.py $W/D-control/greedy.json $W/B-scalarfuse/greedy.json $W/B-scalarfuse/margin-on-B.json > $W/B-scalarfuse/margin.log 2>&1; tail -5 $W/B-scalarfuse/margin.log
step "PHASE2 done; live=$(cur) (B scalar-fuse left serving)"
