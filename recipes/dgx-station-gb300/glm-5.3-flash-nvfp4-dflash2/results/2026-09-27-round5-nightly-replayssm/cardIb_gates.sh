#!/usr/bin/env bash
# Card I-b: re-run the correctness gates that died on API_KEY, on night-rssm, night, ctl (same window). Also acceptance via /metrics and
# the max-effort C1 x3 to settle the 239-vs-264 question on night-rssm. Stop-and-keep; v21 restored at end.
set -uo pipefail
cd ~/glmf
D=~/glmf/cardI-2026-09-27; exec >> "$D/runner.log" 2>&1
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
DSV=dsv41-vllm-v21-fused-split-BOUND-REF
REL=glmf-sglang:0.5.20-tf5.16.1; NIGHT=glmf-sglang:nightly-20260927-425a1f8f-tf5.16.1
MODEL=/models/GLM-5.3-Flash-NVFP4-nvidia-09b04e5e; DRAFT=/models/GLM-5.3-Flash-DFlash2
BASE_KNOBS="--max-mamba-cache-size 48 --speculative-dflash-block-size 7"; ENV_BASE="-e TORCHINDUCTOR_COMPILE_THREADS=1"
TF=~/window-20260913/scripts/tf_noninferiority.py; CTRL_TF=~/glmf/w5-ctrl/tf-w5-ctrl.json
export BASE_URL=http://127.0.0.1:30001/v1 MODEL_NAME=glm-5.3-flash API_KEY=none MODEL=glm-5.3-flash
stop_glmf(){ for c in $(docker ps --format '{{.Names}}' | grep '^glmf-' || true); do log "stop+keep $c"; docker stop "$c" >/dev/null; done; sleep 3; }
wait_ready(){ local i; for i in $(seq 1 240); do curl -sf --max-time 2 http://127.0.0.1:30001/v1/models >/dev/null 2>&1 && { log "ready $1 i=$i"; return 0; }; docker ps --format '{{.Names}}' | grep -q "^$1$" || { log "CONTAINER_EXITED $1"; return 5; }; sleep 5; done; return 5; }
metrics_accept(){ curl -s http://127.0.0.1:30001/metrics | grep -E "spec_accept|accept_length|spec_verify" | grep -v "^#" | head -6; }
gates(){ local TAG=$1 OUT=$D/$1; mkdir -p "$OUT"
  python3 -c 'import flash_bench as fb; fb.knee(concs=(1,), reps=0)' > /dev/null 2>&1
  python3 greedy_equiv.py "$OUT/greedy-$TAG.json" > "$OUT/greedy-gen.txt" 2>&1 || log "GREEDY_GEN_FAIL $TAG: $(tail -1 $OUT/greedy-gen.txt)"
  test -f "$OUT/greedy-$TAG.json" && python3 greedy_equiv.py --compare ~/glmf/w3/greedy-w2-a.json "$OUT/greedy-$TAG.json" > "$OUT/greedy-vs-w2.txt" 2>&1
  LANE="$TAG" BASE_URL=http://127.0.0.1:30001 python3 "$TF" ~/glmf/w3/greedy-w2-a.json "$OUT/tf-$TAG.json" > "$OUT/tf-gen.txt" 2>&1 || log "TF_GEN_FAIL $TAG: $(tail -1 $OUT/tf-gen.txt)"
  test -f "$OUT/tf-$TAG.json" && python3 "$TF" --compare "$CTRL_TF" "$OUT/tf-$TAG.json" > "$OUT/tf-compare.txt" 2>&1
  TAG="$TAG-max2" EFFORT=max python3 c1_methods.py > "$OUT/c1-max2.txt" 2>&1
  metrics_accept > "$OUT/metrics-accept.txt"
  log "GATES $TAG: $(grep -ho 'GREEDY_EQUIV identical=[0-9/]*' $OUT/greedy-vs-w2.txt) | TF $(tail -1 $OUT/tf-compare.txt | cut -c1-140) | max2 $(grep -h recipe-method $OUT/c1-max2.txt | cut -c30-90) | accept $(tr '\n' ' ' < $OUT/metrics-accept.txt | cut -c1-200)"
}
boot(){ local TAG=$1 IMG=$2 EXTRA=$3; stop_glmf
  TAG="I-$TAG" MODEL=$MODEL DRAFT=$DRAFT IMAGE="$IMG" SPEC=dflash KV=fp8_e4m3 MEM=0.85 CTX=1048576 MAXBS=16 EXTRA="$EXTRA" DOCKER_ENV="$ENV_BASE" bash launch-glmf.sh > /dev/null 2>&1
  wait_ready "glmf-I-$TAG"; }
log "===== CARD I-b (gates rerun) START ====="
docker ps --format '{{.Names}}' | grep -q "^$DSV$" && { log "stop+keep $DSV"; docker stop "$DSV" >/dev/null; }
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
# reuse stopped containers where possible: docker start is a valid same-config re-bind (sglang has no autotune cache issue like vLLM)
for arm in night-rssm night ctl2; do
  stop_glmf; log "start glmf-I-$arm"; docker start "glmf-I-$arm" >/dev/null
  if wait_ready "glmf-I-$arm"; then gates "$arm"; else log "RESTART FAIL $arm"; fi
done
stop_glmf
{ echo; echo "## $(date '+%F %H:%M CDT') — Card I-b gates rerun (API_KEY env was unset in I)"; grep -h "GATES" "$D/runner.log" | tail -3 | cut -c1-500; } >> ~/dsv41/results/ledger.md
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; docker start "$DSV" >/dev/null; t0=$SECONDS
until curl -sf -m 3 http://127.0.0.1:30006/v1/models >/dev/null 2>&1; do sleep 20; [[ $((SECONDS-t0)) -gt 1800 ]] && break; done
log "v21 rebound after $((SECONDS-t0))s"; log "===== CARD I-b END ====="
