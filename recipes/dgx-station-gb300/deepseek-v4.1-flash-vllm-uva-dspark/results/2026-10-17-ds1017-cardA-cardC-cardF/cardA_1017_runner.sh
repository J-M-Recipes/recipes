#!/usr/bin/env bash
# Card A — DeepSeek-V4.1-Flash-1017 on the exact v21 lane config. 2026-10-17, Milo (hermes milo) for James.
# James: "you have the station" 2026-10-17 ~07:15 CDT.
# Order: CTL1 (df42c109) -> CAND (1017) -> CTL2 (df42c109), identical launch (0909 image, v19 hook dir = v21,
# rowmap-static-v1, off60/util0.97/1M/seqs24/ksched/cgsizes). Only the /model mount differs.
# Each arm: bind -> facts -> run_window_T1T2 (knee r1/r2, knee6, agent fixture, replay x2) -> gauntlet -> parity capture.
# Stop-and-keep only. Never :30003. Refuses foreign GPU containers. Ends with CAND re-booted as *-LIVE (NOT a promotion;
# v21 stays -BOUND-REF stopped-and-kept; rollback = docker stop LIVE && docker start v21).
set -uo pipefail
D=/home/milo/dsv41/1017; R=$D/results; mkdir -p "$R"
SHA=1bae0e1088993b9b82de08f59de5be5a34e42170
OLD=/models/DeepSeek-V4.1-Flash-df42c109f1defefcbfcedbe7d905718a12266e40
NEW=/models/DeepSeek-V4.1-Flash-1017-$SHA
IMAGE=vllm/vllm-openai:deepseekv41-flash-0909
HOOK=/home/milo/pin-hot-experts/v19/hook
S=/home/milo/pin-hot-experts/scripts; PROMPTS=$S/e5_parity_prompts.json
GDIR=/home/milo/pin-hot-experts/round7/gauntlet
GIMG=sha256:00d577a6a63281e15336029d5bcee4e9a2cf182214a4f20ba6111b1c8e79893d
LEDGER=/home/milo/dsv41/results/ledger.md
V21=dsv41-vllm-v21-fused-split-BOUND-REF
exec >> "$R/runner.log" 2>&1
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
drop_caches(){ sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory' 2>/dev/null || sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; }
stop_all(){ local n; for n in $(docker ps --format '{{.Names}}' | grep -E '^dsv41-vllm-'); do log "stop-and-keep $n"; docker stop "$n" >/dev/null; done; sleep 3; }
wait_bind(){ local n=$1 i
  for ((i=0;i<720;i++)); do
    curl -s -m 2 http://127.0.0.1:30006/v1/models 2>/dev/null | grep -q dsv41-flash-uva && { log "BOUND $n after $((i*10))s"; return 0; }
    docker ps --format '{{.Names}}' | grep -qx "$n" || { log "DIED $n"; return 1; }
    sleep 10
  done; log "TIMEOUT $n"; return 1; }

# ---------- 0. wait for download, verify shard set ----------
log "===== CARD A 1017 START ====="
PIDF=$D/hf-download.pid
if [[ -f $PIDF ]]; then P=$(cat "$PIDF"); while kill -0 "$P" 2>/dev/null; do sleep 30; done; log "hf download pid $P exited"; fi
grep -qiE "error|exception|traceback" "$D/hf-download.log" && log "WARN: hf log has error-ish lines (check)"
n=$(ls "$NEW"/model-*.safetensors 2>/dev/null | wc -l); [[ $n -eq 48 ]] || { log "ABORT: $n/48 shards"; exit 2; }
cmp -s "$OLD/model.safetensors.index.json" "$NEW/model.safetensors.index.json" && log "index.json IDENTICAL to df42c109 (same tensors/shapes -> autotune hash hit expected)" || { log "ABORT: index.json differs — shapes may differ, inspect before boot"; exit 3; }
cmp -s "$OLD/config.json" "$NEW/config.json" && log "config.json IDENTICAL" || { log "ABORT: config.json differs"; exit 3; }
ob=$(du -sb --apparent-size "$OLD" --exclude='.cache' | cut -f1); nb=$(find "$NEW" -path "$NEW/.cache" -prune -o -type f -printf '%s\n' | awk '{s+=$1} END{print s}')
log "bytes old=$ob new=$nb"
# per-shard size must match the old staged shard exactly (same index -> same shard layout)
bad=0; for f in "$NEW"/model-*.safetensors; do b=$(basename "$f"); [[ $(stat -c %s "$f") -eq $(stat -c %s "$OLD/$b") ]] || { log "SIZE MISMATCH $b"; bad=1; }; done
[[ $bad -eq 0 ]] || { log "ABORT: shard size mismatch"; exit 3; }
log "shard sizes all match df42c109 layout"
# tokenizer: 1017 ships tokenizer.json/tokenizer_config.json at top level too — confirm present
[[ -f $NEW/tokenizer.json && -f $NEW/tokenizer_config.json ]] || { log "ABORT: tokenizer files missing"; exit 3; }
cmp -s "$OLD/tokenizer.json" "$NEW/tokenizer.json" && log "tokenizer.json IDENTICAL" || log "NOTE: tokenizer.json differs from staged df42c109 copy"
other=$(docker ps --format '{{.Names}}' | grep -vE '^dsv41-vllm-' || true); [[ -n $other ]] && { log "REFUSE foreign container: $other"; exit 4; }
log "strays: $(ps -eo pid,rss,args --sort=-rss | sed -n 2,4p | tr '\n' '|')"

arm(){ # tag modeldir
  local tag=$1 model=$2 name=dsv41-vllm-1017A-$1
  stop_all; drop_caches
  docker rm -f "$name" >/dev/null 2>&1 || true
  docker run -d --name "$name" --gpus all --ipc host --network host \
    --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK \
    -v "$model":/model:ro -v /mnt/miloark:/mnt/miloark:ro -v /home/milo/dsv41/vllm-cache:/root/.cache/vllm \
    -v "$HOOK/sitecustomize.py":/usr/lib/python3.12/sitecustomize.py:ro -v "$HOOK":/w:ro \
    -v /home/milo/pin-hot-experts/v16:/home/milo/pin-hot-experts/v16 \
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
  local t0; t0=$(date +%s)
  log "launched $name model=$model"
  if ! wait_bind "$name"; then docker logs "$name" > "$R/boot-$tag.log" 2>&1; docker stop "$name" >/dev/null 2>&1; docker rename "$name" "$name-BOOT-FAIL"; return 1; fi
  log "boot->ready $tag $(( $(date +%s) - t0 ))s"
  docker logs "$name" 2>&1 | grep -E "PIN_HOT installed|PIN_HOT rehome done|HBM_expert|pinned_expert|Loaded [0-9]+ configs|GPU KV cache size|Maximum concurrency|MemAvailable|host_avail" > "$R/facts-$tag.txt"
  grep -q "rehome done" "$R/facts-$tag.txt" || { log "ARM $tag: hook did not re-home — invalid"; return 1; }
  grep -q "Loaded [0-9]* configs" "$R/facts-$tag.txt" && log "ARM $tag: autotune HASH HIT" || log "ARM $tag: NO autotune cache load line (live tune?) — check facts"
  bash /home/milo/dsv41/run_window_T1T2.sh "DS1017A-$tag" > "$R/nohup-$tag.log" 2>&1
  mv /home/milo/dsv41/overnight-T1T2/*"DS1017A-$tag"* "$R"/ 2>/dev/null
  rm -rf "$R/gauntlet-$tag"
  (cd "$GDIR" && python3 run_suite.py --base-url http://127.0.0.1:30006/v1 --model dsv41-flash-uva \
     --output "$R/gauntlet-$tag" --tag "ds1017A-$tag" --cache-salt "ds1017A-$tag" --sandbox-image "$GIMG" --workers 4) > "$R/gauntlet-log-$tag.txt" 2>&1 || log "gauntlet $tag nonzero rc"
  python3 "$S/e2c_parity.py" "DS1017A-$tag" "$PROMPTS" "$R/parity" > "$R/parity-log-$tag.txt" 2>&1 || log "parity $tag nonzero"
  docker logs "$name" 2>&1 | grep -cE "Traceback|ERROR" > "$R/errcount-$tag.txt"
  log "ARM $tag: $(grep -h 'conc=' "$R/window-DS1017A-$tag.log" | grep -E 'r2\]' | tr '\n' ';' | cut -c1-400)"
  log "ARM $tag gauntlet: $(python3 -c "import json;d=json.load(open('$R/gauntlet-$tag/summary.json'));print({k:d[k] for k in d if 'count' in k or 'pass' in k or 'total' in k})" 2>/dev/null)"
}
arm CTL1 "$OLD"
arm CAND "$NEW"
arm CTL2 "$OLD"
stop_all
python3 - "$R" <<'PY' | tee "$R/VERDICT.txt"
import json,sys,glob,os,re
R=sys.argv[1]
def knee(tag,r):
    p=f"{R}/knee-DS1017A-{tag}-{r}.json"
    return {int(a):b for a,b in json.load(open(p))["rows"]} if os.path.exists(p) else {}
def rep(tag):
    out=[]
    for r in ("r1","r2"):
        p=f"{R}/replayc-DS1017A-{tag}-{r}-n4.json"
        if os.path.exists(p):
            d=json.load(open(p)); out.append(d.get("agg_tok_s") or d.get("agg") or d)
    return out
def fix(tag):
    p=f"{R}/agentfix-DS1017A-{tag}.json"
    if not os.path.exists(p): return {}
    d=json.load(open(p)); return d if isinstance(d,dict) else {"raw":str(d)[:300]}
def gaunt(tag):
    p=f"{R}/gauntlet-{tag}/summary.json"
    if not os.path.exists(p): return "missing"
    d=json.load(open(p)); return {k:d[k] for k in d if any(s in k for s in ("count","pass","total","fail"))}
print("arm   C1(r1/r2)     C4(r2)   C8(r2)   C16(r2)   replay n=4")
for t in ("CTL1","CAND","CTL2"):
    k1,k2=knee(t,"r1"),knee(t,"r2")
    print(f"{t:5} {k1.get(1,0):6.1f}/{k2.get(1,0):6.1f}  {k2.get(4,0):7.1f}  {k2.get(8,0):7.1f}  {k2.get(16,0):8.1f}   {rep(t)}")
for t in ("CTL1","CAND","CTL2"):
    print(f"agentfix {t}: {json.dumps(fix(t))[:600]}")
    print(f"gauntlet {t}: {gaunt(t)}")
# parity: CAND vs CTL is DIFFERENT WEIGHTS — expected to differ; CTL1 vs CTL2 is the instrument baseline.
def load(t):
    fs=glob.glob(f"{R}/parity/parity-DS1017A-{t}.json"); return json.load(open(fs[0]))["rows"] if fs else None
a,b,c=load("CTL1"),load("CAND"),load("CTL2")
def cmp(x,y,label):
    if not x or not y: print(label,"missing"); return
    xs={p["id"]:p for p in x}; ys={p["id"]:p for p in y}
    def key(p): return (p.get("content"), p.get("tool_norm"), p.get("finish"))
    nontool=[k for k in xs if xs[k].get("finish")!="tool_calls"]
    same_nt=sum(1 for k in nontool if k in ys and key(xs[k])==key(ys[k]))
    same_all=sum(1 for k in xs if k in ys and key(xs[k])==key(ys[k]))
    print(f"parity {label}: non-tool identical {same_nt}/{len(nontool)} · all {same_all}/{len(xs)}")
cmp(a,c,"CTL1 vs CTL2 (instrument baseline)"); cmp(a,b,"CAND vs CTL1 (different weights; informational)")
PY
{ echo; echo "## $(date '+%F %H:%M CDT') — Card A: DeepSeek-V4.1-Flash-1017 ($SHA) on v21 config, CTL1→CAND→CTL2"; echo "worker: Milo (hermes milo) · James: 'you have the station' 2026-10-17 ~07:15"; cat "$R/VERDICT.txt"; echo "receipts $R · dsv41-vllm-1017A-* stopped-and-kept · v21 stopped-and-kept · 1017 re-booted as dsv41-vllm-1017A-LIVE (not promoted)"; } >> "$LEDGER"
# ---------- leave 1017 serving for James (NOT a promotion) ----------
drop_caches
docker rm -f dsv41-vllm-1017A-LIVE >/dev/null 2>&1 || true
docker rename dsv41-vllm-1017A-CAND dsv41-vllm-1017A-CAND-KEPT >/dev/null 2>&1 || true
docker run -d --name dsv41-vllm-1017A-LIVE --gpus all --ipc host --network host \
    --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK \
    -v "$NEW":/model:ro -v /mnt/miloark:/mnt/miloark:ro -v /home/milo/dsv41/vllm-cache:/root/.cache/vllm \
    -v "$HOOK/sitecustomize.py":/usr/lib/python3.12/sitecustomize.py:ro -v "$HOOK":/w:ro \
    -v /home/milo/pin-hot-experts/v16:/home/milo/pin-hot-experts/v16 \
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
wait_bind dsv41-vllm-1017A-LIVE && log "===== CARD A END; 1017 LIVE on :30006 (v21 stopped-and-kept) =====" || log "===== CARD A END; LIVE boot FAILED — nothing serving ====="
