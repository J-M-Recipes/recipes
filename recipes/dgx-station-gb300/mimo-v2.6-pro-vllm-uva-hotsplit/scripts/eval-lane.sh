#!/usr/bin/env bash
# eval-lane.sh TAG : wait ready, warm, ttft/decode, 2 fixture runs, prose C1 x6, C8 knee point
cd ${WORKDIR:-$PWD}; N=mimo26-pro-$1
for i in $(seq 1 45); do sleep 20; docker ps -q --filter name=$N | grep -q . || { echo "[$1] DIED"; grep -B2 -A12 Traceback $N.log | grep EngineCore | tail -12 | cut -c1-230; exit 1; }; curl -sf -m 5 http://127.0.0.1:30007/v1/models >/dev/null && break; done
grep -E "hotsplit plan|hotsplit done|Available KV cache memory|GPU KV cache size" $N.log | cut -c1-200
./agent_fixture.sh $1-warm >/dev/null 2>&1
python3 ttft_bench.py $1
for r in 2 3; do ./agent_fixture.sh $1-r$r 2>&1 | grep -E "^\[" | grep -v weighted; done
python3 - "$1" <<PY
import json,time,urllib.request,sys
B="http://127.0.0.1:30007/v1/chat/completions"
def one(i,n=192):
    body={"model":"mimo26-pro","messages":[{"role":"user","content":f"Write a detailed paragraph about the number {i}, its history and uses. No lists."}],"max_tokens":n,"temperature":0,"chat_template_kwargs":{"enable_thinking":False},"ignore_eos":True}
    t=time.time(); r=json.load(urllib.request.urlopen(urllib.request.Request(B,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"}),timeout=900))
    return r["usage"]["completion_tokens"]/(time.time()-t)
one(999,16); print(f"[{sys.argv[1]}] prose C1 x6:", [round(one(i),1) for i in range(6)])
PY
./knee.sh $1 2>&1 | grep -E "conc= (1|8):"
date "+%I:%M %p"
