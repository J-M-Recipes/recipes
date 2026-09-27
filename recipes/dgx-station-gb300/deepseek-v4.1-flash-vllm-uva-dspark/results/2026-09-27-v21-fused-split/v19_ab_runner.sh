#!/usr/bin/env bash
# v19 same-window A/B: v18 hook (six-op split_ids) vs v19 hook (fused Triton split_ids). 2026-09-27, Milo for James.
# Order: CTL (v18 hook, fresh boot) -> CAND (v19 hook) -> CTL2 (v18 hook) so drift is bracketed.
# Each arm: bind -> run_window_T1T2 (knee r1/r2, knee6, agent fixture, replay x2) -> e2c_parity greedy capture.
# Stop-and-keep only. Never :30003. Restores v18 REF at the end. Autotune hash unchanged (python-only change).
set -uo pipefail
D=~/dsv41/v19; R=$D/results; mkdir -p "$R"
REF=dsv41-vllm-v18-cgsizes-BOUND-REF
MODEL=/models/DeepSeek-V4.1-Flash-df42c109f1defefcbfcedbe7d905718a12266e40
IMAGE=vllm/vllm-openai:deepseekv41-flash-0909
S=~/pin-hot-experts/scripts; PROMPTS=$S/e5_parity_prompts.json
LEDGER=~/dsv41/results/ledger.md
exec >> "$R/runner.log" 2>&1
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
drop_caches(){ sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; }
stop_all(){ local n; for n in $(docker ps --format '{{.Names}}' | grep '^dsv41-vllm-'); do log "stop-and-keep $n"; docker stop "$n" >/dev/null; done; sleep 3; }
wait_bind(){ local n=$1 i
  for ((i=0;i<600;i++)); do
    curl -s -m 2 http://127.0.0.1:30006/v1/models 2>/dev/null | grep -q dsv41-flash-uva && { log "BOUND $n after $((i*10))s"; return 0; }
    docker ps --format '{{.Names}}' | grep -qx "$n" || { log "DIED $n"; return 1; }
    sleep 10
  done; log "TIMEOUT $n"; return 1; }
arm(){ # tag hookdir
  local tag=$1 hook=$2 name=dsv41-vllm-v19ab-$1
  stop_all; drop_caches
  docker rm -f "$name" >/dev/null 2>&1 || true
  docker run -d --name "$name" --gpus all --ipc host --network host \
    --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK \
    -v "$MODEL":/model:ro -v /mnt/miloark:/mnt/miloark:ro -v ~/dsv41/vllm-cache:/root/.cache/vllm \
    -v "$hook/sitecustomize.py":/usr/lib/python3.12/sitecustomize.py:ro -v "$hook":/w:ro \
    -v ~/pin-hot-experts/v16:~/pin-hot-experts/v16 \
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
    --host 0.0.0.0 --port 30006 >/dev/null
  log "launched $name hook=$hook"
  if ! wait_bind "$name"; then docker logs "$name" > "$R/boot-$tag.log" 2>&1; docker stop "$name" >/dev/null 2>&1; docker rename "$name" "$name-BOOT-FAIL"; return 1; fi
  docker logs "$name" 2>&1 | grep -E "PIN_HOT installed|PIN_HOT rehome done|Loaded [0-9]+ configs|GPU KV cache size" > "$R/facts-$tag.txt"
  grep -q "rehome done" "$R/facts-$tag.txt" || { log "ARM $tag: hook did not re-home — invalid"; return 1; }
  bash ~/dsv41/run_window_T1T2.sh "V19AB-$tag" > "$R/nohup-$tag.log" 2>&1
  mv ~/dsv41/overnight-T1T2/*"V19AB-$tag"* "$R"/ 2>/dev/null
  python3 "$S/e2c_parity.py" "V19AB-$tag" "$PROMPTS" "$R/parity" > "$R/parity-log-$tag.txt" 2>&1 || log "parity $tag nonzero"
  log "ARM $tag: $(grep -h 'conc=' "$R/window-V19AB-$tag.log" | grep -E 'r2\]' | tr '\n' ';' | cut -c1-400)"
}
other=$(docker ps --format '{{.Names}}' | grep -vE '^dsv41-vllm-' || true); [[ -n $other ]] && { log "REFUSE foreign container: $other"; exit 4; }
log "===== V19 A/B START ====="
arm CTL1 ~/pin-hot-experts/hook
arm CAND ~/pin-hot-experts/v19/hook
arm CTL2 ~/pin-hot-experts/hook
stop_all
python3 - "$R" <<'PY' | tee "$R/VERDICT.txt"
import json,sys,glob,os
R=sys.argv[1]
def knee(tag,r):
    p=f"{R}/knee-V19AB-{tag}-{r}.json"
    return {int(a):b for a,b in json.load(open(p))["rows"]} if os.path.exists(p) else {}
def rep(tag):
    out=[]
    for r in ("r1","r2"):
        p=f"{R}/replayc-V19AB-{tag}-{r}-n4.json"
        if os.path.exists(p):
            d=json.load(open(p)); out.append(d.get("agg_tok_s") or d.get("agg") or d)
    return out
print("arm   C1(r1/r2)     C4(r2)   C8(r2)   C16(r2)   replay n=4")
for t in ("CTL1","CAND","CTL2"):
    k1,k2=knee(t,"r1"),knee(t,"r2")
    print(f"{t:5} {k1.get(1,0):6.1f}/{k2.get(1,0):6.1f}  {k2.get(4,0):7.1f}  {k2.get(8,0):7.1f}  {k2.get(16,0):8.1f}   {rep(t)}")
# parity: compare greedy tokens CAND vs CTL1 and CTL1 vs CTL2 (control-vs-control baseline)
def load(t):
    fs=glob.glob(f"{R}/parity/parity-V19AB-{t}.json"); return json.load(open(fs[0]))["rows"] if fs else None
a,b,c=load("CTL1"),load("CAND"),load("CTL2")
def cmp(x,y,label):
    if not x or not y: print(label,"missing"); return
    xs={p["id"]:p for p in x}; ys={p["id"]:p for p in y}
    def key(p): return (p.get("content"), p.get("tool_norm"), p.get("finish"))
    nontool=[k for k in xs if xs[k].get("finish")!="tool_calls"]
    same_nt=sum(1 for k in nontool if k in ys and key(xs[k])==key(ys[k]))
    same_all=sum(1 for k in xs if k in ys and key(xs[k])==key(ys[k]))
    print(f"parity {label}: non-tool identical {same_nt}/{len(nontool)} · all {same_all}/{len(xs)}")
cmp(a,b,"CAND vs CTL1"); cmp(a,c,"CTL1 vs CTL2 (baseline)")
PY
{ echo; echo "## $(date '+%F %H:%M CDT') — v19 fused split_ids A/B"; echo "worker: Milo (hermes milo) · James: go 2026-09-27"; cat "$R/VERDICT.txt"; echo "receipts $R · v18 restored"; } >> "$LEDGER"
drop_caches; docker start "$REF" >/dev/null; wait_bind "$REF"
log "===== V19 A/B END; REF restored ====="
