#!/usr/bin/env bash
# campaign_profile_v18.sh — one DIAGNOSTIC boot: exact v18-cgsizes config + torch profiler.
# Captures decode-step traces (C1 prose, C1 shell, C8 prose) so the next lever is chosen from
# where a v18 step actually goes (post pin-hot-experts), not from the v12 profile.
# Stop-and-keep only. Never :30003. Restores v18 REF at the end. Written by Milo for James, 2026-09-27.
set -euo pipefail
REF=dsv41-vllm-v18-cgsizes-BOUND-REF
TAG=PROF-v18
NAME=dsv41-vllm-$TAG
LOG=~/dsv41/$NAME.log
PROF=~/dsv41/prof-$TAG
LEDGER=~/dsv41/results/ledger.md
MODEL=/models/DeepSeek-V4.1-Flash-df42c109f1defefcbfcedbe7d905718a12266e40
IMAGE=vllm/vllm-openai:deepseekv41-flash-0909
B=http://127.0.0.1:30006
log(){ echo "[$(date '+%F %T')] $*"; }
die(){ log "FATAL: $*"; restore_ref || true; exit 1; }
drop_caches(){ sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; }
restore_ref(){
  log "RESTORE REF"
  docker ps -q --filter name=dsv41-vllm- | xargs -r docker stop || true
  drop_caches
  docker start "$REF"
  local t0=$SECONDS
  until curl -sf -m 3 $B/v1/models >/dev/null 2>&1; do sleep 20; [[ $((SECONDS-t0)) -gt 1800 ]] && { log "RESTORE bind timeout"; return 1; }; done
  log "REF rebound after $((SECONDS-t0))s"
}
# safety: refuse if any non-dsv41 GPU container is up
other=$(docker ps --format '{{.Names}}' | grep -vE '^dsv41-vllm-' || true)
[[ -n "$other" ]] && { log "other GPU container running: $other — refuse"; exit 4; }

log "PROFILE v18 campaign start"
docker ps --format '{{.Names}}' | grep -q "^$REF$" && { log "stopping $REF (keep)"; docker stop "$REF"; }
drop_caches
sudo rm -rf "$PROF"; mkdir -p "$PROF"
docker rm -f "$NAME" 2>/dev/null || true
launched=$(date '+%H:%M')
docker run -d --name "$NAME" --gpus all --ipc host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK \
  -v "$MODEL":/model:ro -v /mnt/miloark:/mnt/miloark:ro -v ~/dsv41/vllm-cache:/root/.cache/vllm \
  -v ~/pin-hot-experts/hook/sitecustomize.py:/usr/lib/python3.12/sitecustomize.py:ro \
  -v ~/pin-hot-experts/hook:/w:ro -v ~/pin-hot-experts/v16:~/pin-hot-experts/v16 \
  -v "$PROF":"$PROF" \
  -e VLLM_LOGGING_LEVEL=INFO -e PIN_MODE=split -e PIN_HOOK=/w/pin_hot_experts_hook.py -e PIN_ROWMAP=/w/rowmap-static-v1.json \
  "$IMAGE" \
  --model /model --served-model-name dsv41-flash-uva --trust-remote-code --tensor-parallel-size 1 \
  --offload-backend uva --cpu-offload-gb 60 --cpu-offload-params routed_experts.w13_weight routed_experts.w2_weight \
  --engram-config '{"cpu_offload": true}' \
  --max-model-len 1048576 --max-num-seqs 24 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.97 \
  --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 --enable-auto-tool-choice \
  --long-prefill-token-threshold 6144 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":[[1,4,5],[5,24,1]]}' \
  --cudagraph-capture-sizes 1 2 4 6 8 12 16 18 24 32 40 48 64 96 128 \
  --profiler-config.profiler=torch --profiler-config.torch_profiler_dir="$PROF" \
  --profiler-config.torch_profiler_with_stack=false --profiler-config.torch_profiler_use_gzip=true \
  --profiler-config.torch_profiler_record_shapes=false --profiler-config.max_iterations=60 \
  --host 0.0.0.0 --port 30006
nohup docker logs -f "$NAME" > "$LOG" 2>&1 &
disown
log "launched $NAME; log $LOG"
t0=$SECONDS
until curl -sf -m 3 $B/v1/models >/dev/null 2>&1; do
  sleep 20
  if ! docker ps --format '{{.Names}}' | grep -q "^$NAME$"; then die "container exited during boot"; fi
  grep -qE "ValueError:|Traceback|RuntimeError:" "$LOG" && die "boot error in log"
  [[ $((SECONDS-t0)) -gt 6000 ]] && die "bind timeout"
done
log "BOUND after $((SECONDS-t0))s"
grep -E "HBM_expert|pinned_expert|Loaded [0-9]+ configs|GPU KV cache size|Total CPU offloaded" "$LOG" | head -8 | tee "$PROF/facts.txt"
grep -q "HBM_expert" "$LOG" || die "hook did not re-home (no HBM_expert line) — not a v18 boot"

# warm
python3 - <<'PY'
import json, urllib.request
B="http://127.0.0.1:30006/v1/chat/completions"
def run(p, n=192):
    body={"model":"dsv41-flash-uva","messages":[{"role":"user","content":p}],"max_tokens":n,"temperature":0,"chat_template_kwargs":{"thinking":False},"ignore_eos":True}
    r=urllib.request.Request(B,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"}); urllib.request.urlopen(r,timeout=600).read()
for _ in range(2):
    run("Write a detailed paragraph about the number 7, its history and uses. No lists.")
    run("Write a bash script that finds all .log files under /var/log older than 7 days, compresses each with gzip, and prints a summary. Only code.")
PY
capture(){ # name prompt conc
  local name=$1; local prompt=$2; local c=$3
  log "CAPTURE $name C$c"
  curl -sf -m 10 -X POST $B/start_profile >/dev/null || die "start_profile failed"
  python3 - "$prompt" "$c" <<'PY'
import json, sys, urllib.request, threading
B="http://127.0.0.1:30006/v1/chat/completions"; p=sys.argv[1]; c=int(sys.argv[2])
def run(i):
    body={"model":"dsv41-flash-uva","messages":[{"role":"user","content":p+f" (variant {i})"}],"max_tokens":150,"temperature":0,"chat_template_kwargs":{"thinking":False},"ignore_eos":True}
    r=urllib.request.Request(B,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"}); urllib.request.urlopen(r,timeout=600).read()
ts=[threading.Thread(target=run,args=(i,)) for i in range(c)]
[t.start() for t in ts]; [t.join() for t in ts]
PY
  curl -sf -m 120 -X POST $B/stop_profile >/dev/null || die "stop_profile failed"
  sleep 15
  local newest; newest=$(ls -td "$PROF"/*/ 2>/dev/null | head -1)
  [[ -n "$newest" ]] && sudo mv "$newest" "$PROF/trace-$name-C$c" || log "no trace produced for $name"
  log "captured $name -> $(du -sh "$PROF/trace-$name-C$c" 2>/dev/null | cut -f1)"
}
capture prose "Write a detailed paragraph about the number 42, its history and uses. No lists." 1
capture shell "Write a bash script that rotates nginx logs weekly, keeps 8 archives, and emails a summary. Only code." 1
capture prose "Write a detailed paragraph about the number 42, its history and uses. No lists." 8
cd ~/dsv41 && bash knee.sh "$TAG" || true
c1=$(python3 -c 'import json;print(round(json.load(open("~/dsv41/knee-PROF-v18.json"))["rows"][0][1],1))' 2>/dev/null || echo NA)
docker stop "$NAME" || true
docker rename "$NAME" "$NAME-DIAGNOSTIC" || true
sudo chown -R milo:milo "$PROF" || true
{ echo; echo "## $(date '+%F %H:%M CDT') — PROF-v18 diagnostic"; echo "worker: Milo (hermes milo) · James: go 2026-09-27 · launched $launched · knee C1=$c1 (profiler on) · traces $PROF · v18 restored"; } >> "$LEDGER"
restore_ref
log "PROFILE v18 campaign done; knee C1=$c1"
