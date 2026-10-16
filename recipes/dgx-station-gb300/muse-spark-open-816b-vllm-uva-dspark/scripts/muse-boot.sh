#!/usr/bin/env bash
# Muse S1 boot 1 (2026-10-15, Milo): Muse-Spark-Open-816B-A42B-NVFP4 @4239e4db on ONE GB300, vLLM day-0 image, UVA expert offload.
# Safety: exact-size host memory (VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1; torch pin_memory rounds to pow2 = ~2x host),
# eager, 128K, 4 seqs, no spec; host-memory watchdog kills the container if MemAvailable < FLOOR_GIB. Stop-and-keep.
set -uo pipefail
R=/models/Muse-Spark-Open-816B-A42B-NVFP4; D=$R/runs/${RUN:-s1-boot1}; mkdir -p "$D"; exec >> "$D/runner.log" 2>&1
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
IMG='vllm/vllm-openai@sha256:c91025b5be1039a93ca975d1d44f352b0e770bb6c8b142b5056cf92204ca137c'
GPU='"device=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55"'
M=$R/4239e4db19376904f2e82c167e78f70abb5455fe/original
NAME=${NAME:-muse-s1-o300}; PORT=18100; OFF=${OFF:-300}; FLOOR_GIB=${FLOOR_GIB:-24}
log "===== MUSE S1 BOOT1 START (offload $OFF GiB) ====="
for i in $(seq 1 120); do [ -f $R/STAGE_DONE ] && break; sleep 30; done
[ -f $R/STAGE_DONE ] || { log "REFUSE: STAGE_DONE missing (sha verify not passed)"; exit 3; }
docker ps --format '{{.Names}}' | grep -q . && { log "REFUSE: something is Up: $(docker ps --format '{{.Names}}' | tr '\n' ' ')"; exit 4; }
hbm=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits); [ "$hbm" -lt 2048 ] || { log "REFUSE: GPU1 HBM $hbm MiB"; exit 4; }
sync; echo 1 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null && log "dropped page cache"
grep -E 'MemTotal|MemAvailable' /proc/meminfo | tee "$D/mem-before.txt"
docker rm -f "$NAME" >/dev/null 2>&1
ARGV=(docker run -d --name "$NAME" --gpus "$GPU" --ipc=host --network=host --ulimit memlock=-1 --ulimit stack=67108864
  -v /models:/models:ro -v $R/cache:/root/.cache -e VLLM_LOGGING_LEVEL=INFO -e VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -v $R/sitecustomize.py:/usr/lib/python3.12/sitecustomize.py:ro -e VLLM_AUTOTUNE_CACHE_KEY=31af09c977cda6b179644fc1d98d8b1d4047394cebae776617d7076a379e2a70 "$IMG"
  "$M" --served-model-name muse --host 127.0.0.1 --port $PORT --tensor-parallel-size 1
  --offload-backend uva --cpu-offload-gb "$OFF" --cpu-offload-params experts
  --max-model-len 131072 --max-num-seqs 4 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.92
  ${EAGER:---enforce-eager} ${EXTRA:-} --reasoning-parser muse --tool-call-parser muse --enable-auto-tool-choice)
printf '%q ' "${ARGV[@]}" > "$D/docker-run-argv.txt"; echo >> "$D/docker-run-argv.txt"
mkdir -p $R/cache; t0=$(date +%s)
"${ARGV[@]}" > "$D/container-id.txt" || { log "docker run failed"; exit 5; }
docker logs -f "$NAME" > "$D/container.log" 2>&1 &
# watchdog + ready wait (2 s cadence)
minav=999999; ok=0
for i in $(seq 1 ${WAITN:-1800}); do
  av=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); [ "$av" -lt "$minav" ] && minav=$av
  if [ $((i % 15)) = 0 ]; then echo "$(date +%T) avail=${av}GiB hbm=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)MiB" >> "$D/mem-watch.log"; fi
  if [ "$av" -lt "$FLOOR_GIB" ]; then log "WATCHDOG: MemAvailable ${av} GiB < ${FLOOR_GIB}; docker kill"; docker kill "$NAME"; echo "$minav" > "$D/min-avail.txt"; exit 9; fi
  docker ps --format '{{.Names}}' | grep -qx "$NAME" || { log "EXITED during boot"; echo "$minav" > "$D/min-avail.txt"; tail -40 "$D/container.log" > "$D/boot-failed.log"; exit 6; }
  if [ $((i % 5)) = 0 ] && curl -sf --max-time 2 http://127.0.0.1:$PORT/health >/dev/null; then ok=1; break; fi
  sleep 2
done
echo "$minav" > "$D/min-avail.txt"
[ $ok = 1 ] || { log "TIMEOUT"; docker stop "$NAME"; exit 7; }
log "READY after $(( $(date +%s) - t0 )) s; min MemAvailable during boot ${minav} GiB"
grep -E 'Total CPU offloaded|Model loading took|KV cache|Maximum concurrency|kv_cache_dtype|scaling factor|MoE backend|Using .* backend|ERROR|Error' "$D/container.log" | head -40 > "$D/boot-excerpt.txt"
grep -E 'MemTotal|MemAvailable' /proc/meminfo > "$D/mem-ready.txt"; nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader > "$D/hbm-ready.txt"
python3 /models/Muse-Spark-Open-816B-A42B-NVFP4/muse_gates.py http://127.0.0.1:$PORT/v1 "$D" > "$D/gates.log" 2>&1; log "gates rc=$?"
tail -25 "$D/gates.log"
curl -s http://127.0.0.1:$PORT/metrics | grep -E "^vllm:spec_decode" > "$D/spec-metrics.txt"
[ -n "${POST:-}" ] && { log "POST start"; bash -c "$POST"; log "POST rc=$?"; }
docker stop "$NAME" >/dev/null; log "stop+keep $NAME"
log "===== MUSE S1 BOOT1 END ====="
