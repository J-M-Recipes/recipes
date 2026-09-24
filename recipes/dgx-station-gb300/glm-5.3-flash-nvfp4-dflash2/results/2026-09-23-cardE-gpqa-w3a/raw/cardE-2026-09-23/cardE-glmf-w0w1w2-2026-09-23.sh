#!/usr/bin/env bash
# CARD E (2026-09-23) — GLM-5.3-Flash NVFP4 + DFlash2, recipe pin (v0.5.20-cu130, draft 7d74cdd8, b7, 48 mamba slots, 1M ctx).
# James 2026-09-23: "stop [mimo] for now · W0->W1->W2->W3 sounds good · try port [adaptive verify]".
# This card = W0 (card debt) + W1 (agent claim card) + W2 (strict tool calling, request-side axis) on the pinned config,
# then W0b (long-context variant boot) to test whether the "1M ctx" claim has a KV pool that can hold a 1M prompt.
# Lane :30001, containers glmf-e*. Stop-and-keep. NO restore of any other lane at end (James: no restore-on-done).
# Rules: never :30003/dsfv-*, never glm53-*/dsv41-*/mimo26-* started; STOP file honoured; no secrets printed.
# Written by Milo (Hermes milo profile) for James Meadlock.
set -uo pipefail
export PYTHONUNBUFFERED=1
R=$HOME/glmf/cardE-2026-09-23; mkdir -p "$R"; cd "$R"
G=$HOME/glmf; S=$HOME/pin-hot-experts/scripts; E=$R/scripts
TF=$HOME/window-20260913/scripts/tf_noninferiority.py
BFCL=$HOME/pin-hot-experts/bfcl_data; GPQA=$HOME/pin-hot-experts/gpqa_diamond.csv
STOPF=$R/STOP; LOG=$R/cardE.log
IMG=lmsysorg/sglang:v0.5.20-cu130
MODELDIR=/models/GLM-5.3-Flash-NVFP4-nvidia-09b04e5e; DRAFTDIR=/models/GLM-5.3-Flash-DFlash2
export BASE_URL=http://127.0.0.1:30001/v1 MODEL=glm-5.3-flash API_KEY=""
LOWCTK='{"reasoning_effort":"low"}'
exec >> "$LOG" 2>&1
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
stop_check(){ [[ -f $STOPF ]] && { log "STOP seen"; finish; exit 0; }; return 0; }
stop_glmf(){ local n; for n in $(docker ps --format '{{.Names}}' | grep -E '^glmf-'); do log "stop-and-keep $n"; docker stop "$n" >/dev/null; done; sleep 3; }
finish(){ stop_glmf; log "finish: all glmf-* stopped-and-kept; no other lane restored (by instruction)"; }
power(){ nvidia-smi -i 1 --query-gpu=power.draw --format=csv,noheader,nounits | head -1; }
# refuse if anything else holds the GPU
busy=$(docker ps --format '{{.Names}}' | grep -v -E '^glmf-' || true)
[[ -n $busy ]] && { log "REFUSE: other containers running: $busy"; exit 4; }

boot(){ # TAG MEM EXTRA
  local TAG=$1 MEM=$2 EXTRA=$3 OUT=$R/$1; mkdir -p "$OUT"; stop_check; stop_glmf
  sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
  log "BOOT $TAG mem=$MEM extra=[$EXTRA] (page cache dropped -> cold boot)"
  local t0=$(date +%s)
  TAG="$TAG" MODEL="$MODELDIR" DRAFT="$DRAFTDIR" IMAGE="$IMG" SPEC=dflash KV=fp8_e4m3 MEM="$MEM" CTX=1048576 MAXBS=16 \
    EXTRA="$EXTRA" LOGDIR="$R" bash $G/launch-glmf.sh > "$OUT/launch.txt" 2>&1 || { log "LAUNCH_FAIL $TAG"; cat "$OUT/launch.txt"; return 1; }
  for i in $(seq 1 360); do
    curl -sf --max-time 2 http://127.0.0.1:30001/v1/models >/dev/null 2>&1 && break
    docker ps --format '{{.Names}}' | grep -q "^glmf-$TAG$" || { log "EXITED $TAG"; docker logs --tail 60 "glmf-$TAG" 2>&1 | tr '\r' '\n' | tail -30; return 5; }
    sleep 5; done
  curl -sf --max-time 2 http://127.0.0.1:30001/v1/models >/dev/null 2>&1 || { log "READY_TIMEOUT $TAG"; return 5; }
  local t1=$(date +%s); echo "cold_boot_to_models_s=$((t1-t0))" > "$OUT/boot-time.txt"; log "READY $TAG after $((t1-t0))s"
  docker logs "glmf-$TAG" 2>&1 | tr '\r' '\n' | grep -E "sglang version|Load weight|DFLASH|Mamba Cache|KV Cache is allocated|max_running_requests|max_total_num_tokens|Capture" | grep -v server_args | cut -c1-240 | head -40 > "$OUT/boot-excerpt.txt"
  docker inspect "glmf-$TAG" --format '{{.Image}} {{json .Config.Cmd}}' > "$OUT/inspect.txt"
  return 0; }

warm(){ local OUT=$1; ( cd $G && python3 -c 'import flash_bench as fb; fb.knee(concs=(1,2,4,8), reps=0)' ) > "$OUT/warm.txt" 2>&1; }

log "===== CARD E START (W0 card debt · W1 BFCL claim card · W2 strict tools · W0b long-ctx) ====="
docker image inspect "$IMG" --format '{{.Id}} {{index .RepoDigests 0}}' > "$R/image.txt"
uname -r > "$R/kernel.txt"; nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 >> "$R/kernel.txt"; cat /proc/cmdline >> "$R/kernel.txt"
sha256sum $BFCL/*.json $BFCL/possible_answer/*.json > "$R/BFCL-SHA256SUMS"; sha256sum $E/*.py > "$R/scripts-SHA256SUMS"

# ---------------- E0: recipe pin, daily config ----------------
T=e0; O=$R/$T
if boot $T 0.85 "--max-mamba-cache-size 48 --speculative-dflash-block-size 7"; then
  t_warm0=$(date +%s); warm "$O"; echo "warm_all_shapes_s=$(( $(date +%s)-t_warm0 ))" >> "$O/boot-time.txt"
  ( cd $G && TAG=$T-low EFFORT=low python3 c1_methods.py ) > "$O/c1-low.txt" 2>&1; log "C1 $(grep -m1 'tok/s' $O/c1-low.txt | cut -c1-120)"
  ( cd $G && TAG=$T-low EFFORT=low CONT=glmf-$T python3 accept_probe.py ) > "$O/accept-low.txt" 2>&1
  ( cd $G && EFFORT=low python3 tool_harness.py ) > "$O/tool-1.txt" 2>&1
  ( cd $G && python3 greedy_equiv.py "$O/greedy-$T.json" > "$O/greedy-gen.txt" 2>&1; python3 greedy_equiv.py --compare $HOME/glmf/w3/greedy-w2-a.json "$O/greedy-$T.json" ) > "$O/greedy-vs-w2a.txt" 2>&1
  log "GREEDY $(grep -h GREEDY_EQUIV $O/greedy-vs-w2a.txt | tail -1)"
  LANE=$T BASE_URL=http://127.0.0.1:30001 python3 "$TF" $HOME/glmf/w3/greedy-w2-a.json "$O/tf-$T.json" > "$O/tf-log.txt" 2>&1; log "TF $(tail -1 $O/tf-log.txt | cut -c1-160)"
  stop_check
  # W0: cold prefill on v0.5.20 (pass 1 warms prompt-length classes, pass 2 is the measurement)
  CTK="$LOWCTK" SIZES="8000 32000 64000 128000" N=2 python3 $E/cold_prefill_probe.py > "$O/prefill-pass1-warm.txt" 2>&1
  CTK="$LOWCTK" SIZES="8000 32000 64000 128000" N=3 python3 $E/cold_prefill_probe.py > "$O/prefill.txt" 2>&1; log "PREFILL $(tr '\n' ' ' < $O/prefill.txt | cut -c1-400)"
  # W0: needle ladder under the daily config's KV pool (577K tokens) — top rung kept below the pool
  LANE=$T RUNGS="8000 32000 128000 256000 400000 540000" python3 $E/needle_ladder.py "$O/needle.jsonl" > "$O/needle.log" 2>&1; log "NEEDLE $(grep -E 'RUNG|STOP' $O/needle.log | tail -4 | tr '\n' ' ' | cut -c1-300)"
  stop_check
  # W1: agent claim card — BFCL dev (600) then held-out (1,311), protocol harness/protocol.yaml (effort low, T=0, C8, 2048 tok)
  p0=$(power); CTK="$LOWCTK" SUITE=dev CONC=8 MAXTOK=2048 python3 $E/bfcl_gate.py $T-dev $BFCL "$O" > "$O/bfcl-dev.log" 2>&1; p1=$(power)
  echo "power-pre=$p0 power-post=$p1" >> "$O/bfcl-dev.log"; log "$(grep '^BFCL' $O/bfcl-dev.log)"
  p0=$(power); CTK="$LOWCTK" SUITE=heldout CONC=8 MAXTOK=2048 python3 $E/bfcl_gate.py $T-heldout $BFCL "$O" > "$O/bfcl-heldout.log" 2>&1; p1=$(power)
  echo "power-pre=$p0 power-post=$p1" >> "$O/bfcl-heldout.log"; log "$(grep '^BFCL' $O/bfcl-heldout.log)"
  stop_check
  # W2: strict tool calling (request-side: per-tool strict:true -> xgrammar structural tag). Candidate -> dev decides candidacy.
  p0=$(power); CTK="$LOWCTK" STRICT=1 SUITE=dev CONC=8 MAXTOK=2048 python3 $E/bfcl_gate.py $T-dev-strict $BFCL "$O" > "$O/bfcl-dev-strict.log" 2>&1; p1=$(power)
  echo "power-pre=$p0 power-post=$p1" >> "$O/bfcl-dev-strict.log"; log "$(grep '^BFCL' $O/bfcl-dev-strict.log)"
  docker logs --since 30m "glmf-$T" 2>&1 | grep -i -E "grammar|structural|xgrammar" | grep -v -i "warn.*deprecat" | tail -20 > "$O/strict-grammar-evidence.txt"
  dev_ok=$(python3 -c "import json;print(json.load(open('$O/bfcl-$T-dev-summary.json'))['all']['ok'])")
  st_ok=$(python3 -c "import json;print(json.load(open('$O/bfcl-$T-dev-strict-summary.json'))['all']['ok'])")
  st_ok=${st_ok:-0}; dev_ok=${dev_ok:-999}; log "W2 dev strict $st_ok vs nonstrict $dev_ok"
  if (( st_ok >= dev_ok )); then
    log "W2 candidate passes dev -> held-out once (promotion decision)"
    p0=$(power); CTK="$LOWCTK" STRICT=1 SUITE=heldout CONC=8 MAXTOK=2048 python3 $E/bfcl_gate.py $T-heldout-strict $BFCL "$O" > "$O/bfcl-heldout-strict.log" 2>&1; p1=$(power)
    echo "power-pre=$p0 power-post=$p1" >> "$O/bfcl-heldout-strict.log"; log "$(grep '^BFCL' $O/bfcl-heldout-strict.log)"
  else log "W2 strict fails dev -> no held-out run (rule)"; fi
  # second tool-harness pass (repeat stability); strict speed cost = bfcl wall_s strict vs non-strict
  ( cd $G && EFFORT=low python3 tool_harness.py ) > "$O/tool-2.txt" 2>&1
  stop_check
  # W0: public sanity anchor we run ourselves — GPQA-Diamond 198q, model-default (max) effort, T=0, C8, 32K max tokens
  TAG=$T-gpqa OUT="$O/gpqa-$T.jsonl" BASE_URL=$BASE_URL MODEL=$MODEL API_KEY=none CONC=8 MAXTOK=32768 THINKING=1 \
    python3 $E/gpqa_diamond.py $GPQA > "$O/gpqa.log" 2>&1; log "$(grep 'GPQA-Diamond' $O/gpqa.log)"
  ( cd $G && TAG=$T-low-post EFFORT=low python3 c1_methods.py ) > "$O/c1-low-post.txt" 2>&1
else log "BOOT_FAIL $T"; fi

# ---------------- E0b: long-context variant (does a 1M prompt fit?) ----------------
T=e0b; O=$R/$T
if boot $T 0.90 "--max-mamba-cache-size 16 --speculative-dflash-block-size 7"; then
  pool=$(grep -o 'max_total_num_tokens=[0-9]*' "$O/boot-excerpt.txt" | head -1 | cut -d= -f2); log "E0b pool=$pool"
  warm "$O"
  ( cd $G && TAG=$T-low EFFORT=low python3 c1_methods.py ) > "$O/c1-low.txt" 2>&1; log "C1 e0b $(grep -m1 'tok/s' $O/c1-low.txt | cut -c1-120)"
  rungs="540000 768000"; [[ -n $pool ]] && (( pool >= 1030000 )) && rungs="540000 768000 1000000"
  LANE=$T RUNGS="$rungs" python3 $E/needle_ladder.py "$O/needle.jsonl" > "$O/needle.log" 2>&1; log "NEEDLE e0b $(grep -E 'RUNG|STOP' $O/needle.log | tail -6 | tr '\n' ' ' | cut -c1-400)"
  ( cd $G && EFFORT=low python3 tool_harness.py ) > "$O/tool-1.txt" 2>&1
else log "BOOT_FAIL $T"; docker logs --tail 40 glmf-$T 2>&1 | tr '\r' '\n' | grep -iE "error|memory|Traceback" | tail -5; fi

finish
log "===== CARD E END ====="
echo CARDE_DETACHED_END
