#!/usr/bin/env bash
set -u
C=/home/milo/big-v1-campaign; L=$C/LEDGER.md; cd $C
S=~/window-20260913/scripts; W=~/window-20260914
export MODEL=glm-5.3-big API_KEY="$(cat ~/.glm_api_key)"
cur(){ docker ps --format '{{.Names}}' | grep glm53-big | head -1; }
step(){ echo "$(date -Is) $*" >> $L; echo "$(date -Is) $*"; }
ready(){ for i in $(seq 1 360); do c=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $API_KEY" http://127.0.0.1:30001/v1/models || true); [ "$c" = 200 ] && return 0; sleep 10; done; return 2; }
warm(){ curl -s -H "Authorization: Bearer $API_KEY" -H 'Content-Type: application/json' http://127.0.0.1:30001/v1/chat/completions -d '{"model":"glm-5.3-big","max_tokens":256,"temperature":0,"messages":[{"role":"user","content":"Warm up: describe a lighthouse in 150 words."}]}' >/dev/null; }
step "PHASE3 start after electrician shutdown; resuming B quality receipts"
docker start glm53-big-sc13g-mtp-ctx256k-scalarfuse-20260914 && step "started scalarfuse (B)"
ready && warm
BASE_URL=http://127.0.0.1:30001/v1 python3 $S/greedy_equiv.py $W/B-scalarfuse/greedy.json
python3 $S/greedy_equiv.py --compare $W/D-control/greedy.json $W/B-scalarfuse/greedy.json | tee $W/B-scalarfuse/greedy-vs-D.txt
LANE=B-scalarfuse BASE_URL=http://127.0.0.1:30001 python3 $S/divergence_margin.py $W/D-control/greedy.json $W/B-scalarfuse/greedy.json $W/B-scalarfuse/margin-on-B.json 2>&1 | tail -6 | tee $W/B-scalarfuse/margin.log
LANE=B-scalarfuse-scoring-E BASE_URL=http://127.0.0.1:30001 python3 $S/divergence_margin.py $W/D-control/greedy.json $W/E-k2-scalarfuse/greedy.json $W/E-k2-scalarfuse/margin-on-B.json 2>&1 | tail -6 | tee $W/E-k2-scalarfuse/margin.log
step "PHASE3 done; live=$(cur)"
