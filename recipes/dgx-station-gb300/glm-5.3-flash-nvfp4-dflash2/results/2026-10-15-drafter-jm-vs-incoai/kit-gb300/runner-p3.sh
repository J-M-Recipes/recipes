#!/usr/bin/env bash
# P3 runner on gb300. One boot per arm, clean teardown between. Fixtures per arm:
#   a) weekend E1 matrix + probe (diag_matrix_gb300.py) x3 passes; G1 regimes x1
#   b) canada-quant holdout kit: c16 x500, c6 x500, c1 x100 (thinking ON, T=1.0/top_p 0.95, 1024 tok)
#   c) P1 holdout 300 stratified (thinking OFF greedy, 300 tok)
# Usage: setsid nohup bash runner-p3.sh ARM1 ARM2 ... > runner.log 2>&1 &
# Touch $P3/STOP to stop after the current step.
set -Euo pipefail
P3=/models/drafter-train/p3; K=$P3/kit-gb300; RAW=$P3/raw
MODEL=/models/GLM-5.3-Flash-NVFP4/aa28e1f54130286c95fee10d0705c74ce8743734/original
TEMPLATE=/models/drafter-train/p1/build/chat_template_mm.jinja
URL=http://127.0.0.1:30021
log() { echo "[$(date -Is)] $*"; }
hbm() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1; }
memav() { awk '/MemAvailable:/ {print $2}' /proc/meminfo; }
stopchk() { [[ -f $P3/STOP ]] && { log "STOP file seen"; return 0; }; return 1; }

declare -A DRAFT SHA KK
DRAFT[JM_K4]=/models/drafter-train/ckpt-final;            SHA[JM_K4]=bfdf92d2d92527ca5e6a8202457d0ae6c1b7948e4102e201fb45f6b52b3adf31; KK[JM_K4]=4
DRAFT[JM_K7]=/models/drafter-train/ckpt-final;            SHA[JM_K7]=bfdf92d2d92527ca5e6a8202457d0ae6c1b7948e4102e201fb45f6b52b3adf31; KK[JM_K7]=7
DRAFT[INCOAI_K4]=/models/GLM-5.3-Flash-DFlash2;           SHA[INCOAI_K4]=8931dc522be0aa31760a7463f8d2f8044fa3e6d40be2e87aa08e9fd17bfd6683; KK[INCOAI_K4]=4
DRAFT[INCOAI_K7]=/models/GLM-5.3-Flash-DFlash2;           SHA[INCOAI_K7]=8931dc522be0aa31760a7463f8d2f8044fa3e6d40be2e87aa08e9fd17bfd6683; KK[INCOAI_K7]=7
DRAFT[INCOAI_bf58_K4]=/models/GLM-5.3-Flash-DFlash2-bf582e4e; SHA[INCOAI_bf58_K4]=b038e1d9d1e7833fa3880c2c0135ba9b673013f03da1b29fb831931584759dac; KK[INCOAI_bf58_K4]=4
CFGSHA=c4aeac0101196a6e26705b34c45230bcd0c7c68ee2d2d1efdb242087f3712573

log "RUNNER_START arms=$* host=$(hostname) kernel=$(uname -r)"
for ARM in "$@"; do
  stopchk && break
  D=${DRAFT[$ARM]}; S=${SHA[$ARM]}; KV=${KK[$ARM]}; O=$RAW/$ARM; NAME="p3-${ARM,,}-30021"
  mkdir -p "$O/boot"
  # preflight
  up=$(docker ps --format '{{.Names}}')
  if [[ -n "$up" ]]; then log "REFUSE_RUNNING $ARM: $up"; exit 3; fi
  h=$(hbm); if (( h > 2048 )); then log "REFUSE_HBM $ARM hbm=${h}MiB"; exit 3; fi
  if ! printf '%s  %s\n%s  %s\n' "$S" "$D/model.safetensors" "$CFGSHA" "$D/config.json" | sha256sum -c - > "$O/boot/sha-check.txt" 2>&1; then
    log "SHA_FAIL $ARM"; cat "$O/boot/sha-check.txt"; exit 4; fi
  log "PREFLIGHT_OK $ARM hbm=${h}MiB memavail_kB=$(memav) sha=$(cut -c1-16 <<<"$S")"
  t0=$(date +%s); log "BOOT_BEGIN $ARM k=$KV draft=$D"
  bash "$K/launch-p3.sh" "$ARM" "$D" "$KV" > "$O/boot/launch.out" 2>&1 || { log "LAUNCH_FAIL $ARM"; exit 5; }
  docker inspect "$NAME" --format '{{json .Args}}' > "$O/boot/docker-args.json"
  (docker logs -f "$NAME" > "$O/boot/server.log" 2>&1 &)
  ok=0
  for i in $(seq 1 480); do
    curl -sf --max-time 2 $URL/v1/models >/dev/null 2>&1 && { ok=1; break; }
    [[ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" == true ]] || break
    sleep 5
  done
  t1=$(date +%s)
  echo "{\"arm\":\"$ARM\",\"boot_epoch\":$t0,\"ready_epoch\":$t1,\"boot_to_ready_s\":$((t1-t0)),\"ready\":$ok}" > "$O/boot/timing.json"
  if (( ok != 1 )); then
    log "BOOT_FAIL $ARM after $((t1-t0))s"; docker logs --tail 300 "$NAME" > "$O/boot/fail-tail.log" 2>&1
    docker stop "$NAME" >/dev/null 2>&1; docker rm "$NAME" >/dev/null 2>&1; continue
  fi
  log "READY $ARM boot_to_ready=$((t1-t0))s hbm=$(hbm)MiB memavail_kB=$(memav)"
  sleep 2
  grep -E "KV cache|GPU KV|Maximum concurrency|dflash|DFlash|target_layer_ids|auxiliary|CUDA graph|Graph capturing|Model loading took|Application startup" "$O/boot/server.log" | head -60 > "$O/boot/boot-excerpt.txt"
  curl -fsS $URL/metrics | grep -E "per_pos" > "$O/boot/perpos-at-boot.txt" || true
  echo "{\"after_ready_hbm_MiB\":$(hbm),\"after_ready_memavail_kB\":$(memav)}" > "$O/boot/mem-after-ready.json"

  for m in 1 2 3; do
    stopchk && break
    log "MATRIX_m${m}_BEGIN $ARM"
    python3 "$K/diag_matrix_gb300.py" --base-url $URL --mode dflash --out "$O/matrix-m$m" > "$O/matrix-m$m.log" 2>&1; rc=$?
    log "MATRIX_m${m}_DONE $ARM rc=$rc $(grep -o '"c1_prose_accept_len_median": [0-9.]*' "$O/matrix-m$m/summary.json" 2>/dev/null)"
  done
  stopchk && { docker stop "$NAME"; docker rm "$NAME"; break; }
  log "P1HOLDOUT_BEGIN $ARM"
  docker exec "$NAME" python3 /models/drafter-train/p3/kit-gb300/p1_holdout_eval.py --url $URL \
    --holdout /models/drafter-train/data/holdout.jsonl --template $TEMPLATE --model-dir $MODEL \
    --out /p3raw/$ARM/p1-holdout-300.json --label "$ARM" > "$O/p1-holdout-300.log" 2>&1; rc=$?
  log "P1HOLDOUT_DONE $ARM rc=$rc $(grep RESULT "$O/p1-holdout-300.log" | cut -c1-300)"
  for spec in "16 500" "6 500" "1 100"; do
    stopchk && break
    read -r c n <<<"$spec"
    log "CQ_c${c}x${n}_BEGIN $ARM"
    docker exec "$NAME" python3 /models/drafter-train/p3/cq-eval/holdout_client.py --url $URL \
      --prompts /models/drafter-train/p3/cq-eval/prompts_holdout_500.jsonl --limit $n --concurrency $c --max-tokens 1024 \
      --model-dir $MODEL --served-name glm-5.3-flash --out /p3raw/$ARM/cq-c${c}x${n}.json --label "${ARM}_c${c}x${n}" \
      > "$O/cq-c${c}x${n}.log" 2>&1; rc=$?
    log "CQ_c${c}x${n}_DONE $ARM rc=$rc $(grep RESULT "$O/cq-c${c}x${n}.log" | cut -c1-200) hbm=$(hbm)MiB memavail_kB=$(memav)"
  done
  if ! stopchk; then
    log "G1_BEGIN $ARM"
    BASE_URL=$URL python3 "$K/g1_regimes_gb300.py" "$O/g1" > "$O/g1.log" 2>&1; rc=$?
    log "G1_DONE $ARM rc=$rc $(grep SUMMARY "$O/g1.log" | cut -c1-300)"
  fi
  echo "{\"before_teardown_hbm_MiB\":$(hbm),\"before_teardown_memavail_kB\":$(memav)}" > "$O/boot/mem-before-teardown.json"
  docker stop "$NAME" >/dev/null; docker rm "$NAME" >/dev/null
  sleep 10
  log "TEARDOWN $ARM hbm=$(hbm)MiB memavail_kB=$(memav)"
done
log "PIN_END incoai7d74=$(sha256sum /models/GLM-5.3-Flash-DFlash2/model.safetensors | cut -c1-16) jm=$(sha256sum /models/drafter-train/ckpt-final/model.safetensors | cut -c1-16)"
log "RUNNER_DONE"
