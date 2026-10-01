#!/usr/bin/env bash
# Card N (2026-10-01): big-GLM slot-cache daily, one axis: --max-num-seqs (1 | 2 | 4). Grok idea #4; recipe lists it as the
# untested next axis. Everything else byte-identical to the 2026-09-30 bf16 window argv (launch-window.sh, KV bfloat16).
#   seqs1  : control (= daily)        -> speed_reps C1 + conc_bench C1 + greedy 20 + TF (instrument proof)
#   seqs4  : --max-num-seqs 4         -> conc_bench C1,2,4 + greedy vs seqs1 + TF vs seqs1 + slot-cache HIT/misses per step
#   seqs2  : --max-num-seqs 2         -> same
# Per-stream and aggregate tok/s at C1/C2/C4; the question is whether 2-4 streams amortize per-step overhead or inflate
# C2C miss bytes (more distinct experts per step). Quality gate: greedy vs control + TF vs control (same bars as window 1).
# Stop-and-keep. Big lane is not production (daily is Flash on :30001 and is STOPPED for this card; restarted at the end).
set -uo pipefail
C=$HOME/glm53-big-opt-20260930; T=$C/staging/tools
D=$HOME/glm53-big-opt-20260930/cardN-2026-10-01; mkdir -p "$D"; exec >> "$D/runner.log" 2>&1
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
R=$HOME/recipe-glm53-slotcache
IMAGE=vllm-glm53-uva:v0.28.0-2cf0a691
KEY_FILE=$HOME/.glm_api_key
FLASH_DAILY=glmf-DAILY-rssm-20260929
export MODEL=glm-5.3-big
stop_all(){ for c in $(docker ps --format '{{.Names}}' | grep -E '^(glmf-|glm53-big-)' || true); do log "stop+keep $c"; docker stop "$c" >/dev/null; done; sleep 3; }
wait_hbm(){ local i u; for i in $(seq 1 120); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' '); [ "$u" -lt 2048 ] && { log "HBM free ($u MiB)"; return 0; }; sleep 10; done; log "HBM NOT FREE ($u MiB)"; return 1; }
launch(){ local lane=$1 seqs=$2 NAME="glm53-big-cardN-${lane}-$(date +%Y%m%d)"; mkdir -p "$D/$lane"
  docker container inspect "$NAME" >/dev/null 2>&1 && { log "$NAME exists; refuse"; return 4; }
  docker run -d --name "$NAME" --gpus all --shm-size 32g --network host \
    -v /models/GLM-5.3-NVFP4-big:/model:ro -v $HOME/vllm-cache:/root/.cache/vllm \
    -v "$R/patches/sitecustomize.py:/usr/lib/python3.12/sitecustomize.py:ro" -v "$R:/w:ro" -v "$D/$lane:/wcap" \
    -e VLLM_LOGGING_LEVEL=INFO -e VLLM_AUTOTUNE_CACHE_KEY=slotcache-S112 \
    -e EXACT_PIN=1 -e EXACT_PIN_FILE=/w/patches/exact_pin.py \
    -e SLOT_CACHE=112 -e SLOT_CACHE_PER_LAYER=/w/configs/slots-7360-ctx256k.json \
    -e SLOT_CACHE_HOOK=/w/patches/slot_cache_hook.py -e SLOT_CACHE_SCALAR_FUSE=1 -e SLOT_CACHE_ID_RING=0 \
    -e SLOT_CACHE_CAPTURE=0 -e SLOT_CACHE_ROUTER=ffi -e SLOT_CACHE_UNPACKED=0 -e SLOT_CACHE_LOGIT_RING=0 \
    -e SLOT_CACHE_STATS_SEC=20 -e SLOT_CACHE_BYPASS_TOKENS=16 \
    "$IMAGE" \
    /model --host 0.0.0.0 --port 30001 --api-key <REDACTED> "$KEY_FILE")" \
    --served-model-name glm-5.3-big --trust-remote-code --quantization modelopt --load-format safetensors \
    --offload-backend uva --cpu-offload-gb 420 --cpu-offload-params routed_experts.w13_weight routed_experts.w2_weight \
    --gpu-memory-utilization 0.95 --kv-cache-dtype bfloat16 --kv-cache-memory 25769803776 \
    --max-model-len 262144 --max-num-seqs "$seqs" --max-num-batched-tokens 8192 \
    --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 \
    --compilation-config '{"mode":3,"backend":"eager"}' \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}' >/dev/null || { log "docker run failed $lane"; return 5; }
  echo "$NAME" > "$D/$lane/container.txt"
  setsid nohup bash -c "docker logs -f $NAME 2>&1 | tr '\r' '\n' > $D/$lane/boot.log" >/dev/null 2>&1 < /dev/null &
  local i c; for i in $(seq 1 720); do c=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:30001/v1/models)
    { [ "$c" = 401 ] || [ "$c" = 200 ]; } && { log "READY $lane ($NAME) i=$i"; return 0; }
    docker ps -q -f name="$NAME" | grep -q . || { log "EXITED during boot $lane"; grep -nE "Error|Traceback|OOM" "$D/$lane/boot.log" | tail -5 | cut -c1-250; return 5; }; sleep 10; done
  log "READY_TIMEOUT $lane"; return 5; }
gates(){ local lane=$1 concs=$2 OUT=$D/$lane; export API_KEY="$(cat $KEY_FILE)"; cd "$T"
  grep -E "max_num_seqs|GPU KV cache size|kv_cache_dtype|scaling factor" "$OUT/boot.log" | head -6 | cut -c1-250 > "$OUT/boot-excerpt.txt"
  BASE_URL=http://127.0.0.1:30001 LANE=$lane python3 speed_reps.py "$OUT/$lane-speed.json" > "$OUT/speed.log" 2>&1 || log "SPEED_FAIL $lane"
  BASE_URL=http://127.0.0.1:30001 LANE=$lane CONCS="$concs" python3 "$D/conc_bench.py" "$OUT/$lane-conc.json" > "$OUT/conc.log" 2>&1 || log "CONC_FAIL $lane"
  BASE_URL=http://127.0.0.1:30001/v1 python3 greedy_equiv.py "$OUT/$lane-greedy.json" > "$OUT/greedy.log" 2>&1 || log "GREEDY_FAIL $lane"
  [ "$lane" != seqs1 ] && test -f "$D/seqs1/seqs1-greedy.json" && python3 greedy_equiv.py --compare "$D/seqs1/seqs1-greedy.json" "$OUT/$lane-greedy.json" >> "$OUT/greedy.log" 2>&1
  BASE_URL=http://127.0.0.1:30001 LANE=$lane python3 tf_noninferiority.py "$D/seqs1/seqs1-greedy.json" "$OUT/$lane-tf.json" > "$OUT/tf.log" 2>&1 || log "TF_FAIL $lane"
  [ "$lane" != seqs1 ] && test -f "$D/seqs1/seqs1-tf.json" && python3 tf_noninferiority.py --compare "$D/seqs1/seqs1-tf.json" "$OUT/$lane-tf.json" > "$OUT/$lane-tf-compare.json" 2>>"$OUT/tf.log"
  # slot-cache stats during the conc bench: last 30 STATS windows
  grep "SLOT_CACHE STATS" "$OUT/boot.log" | tail -30 | cut -c1-250 > "$OUT/slotcache-stats.txt"
  unset API_KEY
  log "ARM $lane: speed C1 $(python3 -c "import json;print(round(json.load(open('$OUT/$lane-speed.json'))['summary']['c1_median_tok_s'],1))" 2>/dev/null) | conc $(python3 -c "import json;d=json.load(open('$OUT/$lane-conc.json'))['concs'];print(' '.join(f\"C{k}:agg {v['agg_median']:.1f}/ps {v['per_stream_median']:.1f}/ttft {v['ttft_median']:.2f}/err {v['errors']}\" for k,v in d.items()))" 2>/dev/null) | greedy $(grep -ho 'GREEDY_EQUIV identical=[0-9/]*' $OUT/greedy.log | tail -1) | TF $(grep -oE '"mean": [0-9.e-]+|"p99": [0-9.e-]+|"max": [0-9.e-]+' $OUT/$lane-tf-compare.json 2>/dev/null | tr '\n' ' ') | HIT $(grep -oE 'misses/step/layer=[0-9.]+ .*HIT=[0-9.]+' $OUT/slotcache-stats.txt | tail -1)"; }
arm(){ local lane=$1 seqs=$2 concs=$3; stop_all; wait_hbm || return 1; launch "$lane" "$seqs" || { log "BOOT FAIL $lane"; return 1; }; gates "$lane" "$concs"; }
log "===== CARD N (big-GLM --max-num-seqs 1/4/2) START ====="
other=$(docker ps --format '{{.Names}}' | grep -vE '^(glmf-|glm53-big-)' || true); [[ -n $other ]] && { log "REFUSE foreign container: $other"; exit 4; }
for f in "$T/speed_reps.py" "$T/greedy_equiv.py" "$T/tf_noninferiority.py" "$D/conc_bench.py" "$KEY_FILE" "$R/configs/slots-7360-ctx256k.json"; do test -f "$f" || { log "MISSING $f"; exit 3; }; done
docker image inspect "$IMAGE" >/dev/null 2>&1 || { log "NO IMAGE $IMAGE"; exit 3; }
arm seqs1 1 "1"
arm seqs4 4 "1,2,4"
arm seqs2 2 "1,2"
stop_all
log "restart Flash daily: docker start $FLASH_DAILY"; docker start "$FLASH_DAILY" >/dev/null
for i in $(seq 1 120); do curl -sf --max-time 2 http://127.0.0.1:30001/health >/dev/null 2>&1 && { log "Flash daily back on :30001"; break; }; sleep 5; done
curl -sf --max-time 2 http://127.0.0.1:30001/health >/dev/null 2>&1 || log "DAILY RESTART FAILED — page James"
{ echo; echo "## $(date '+%F %H:%M CDT') — Card N: big-GLM slot-cache --max-num-seqs 1/4/2 (Grok idea #4; recipe 'untested next axis')"
  echo "worker: sonnet55 under Milo (hermes milo) · James: both 2026-10-01 ~15:30"
  grep -h "ARM \|BOOT FAIL\|_FAIL\|REFUSE\|daily back\|DAILY RESTART" "$D/runner.log" | cut -c1-900
  echo "receipts $D · glm53-big-cardN-* stopped-and-kept · Flash daily restarted (same config)"; } >> $HOME/dsv41/results/ledger.md
log "===== CARD N END ====="
