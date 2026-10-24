#!/usr/bin/env bash
# launch_ar.sh — v26ar adaptive-residency boot = the exact v23 launch
# (launch_v23.sh, promoted 2026-10-18) with ONLY these differences:
#   hook dir  v19/hook -> v26ar/hook   (pin_hot_experts_hook.py + e4_adaptive.py patched:
#                                       PIN_COUNTER=atomic, guarded swaps, ctrl/telemetry)
#   PIN_MODE  split -> adaptive        (+ execute_model swap barrier)
#   +env      PIN_COUNTER=atomic PIN_ADAPTIVE_FREEZE=1 (boot frozen: count only)
#             PIN_ADAPTIVE_GO / PIN_CTRL / PIN_TELEMETRY / PIN_ADAPTIVE_LOG -> /ar (rw)
#             PIN_COUNT_TMAX=256 PIN_SWAPS_PER_BARRIER=8
#   +mount    $RUN -> /ar (rw)
# Same image, model, args, rowmap (rowmap-static-v1.json), autotune cache.
set -euo pipefail
NEW=/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170
IMAGE=vllm/vllm-openai:deepseekv41-flash-0909
CACHE=/home/milo/dsv41/vllm-cache
HOOK=${HOOK:-/home/milo/pin-hot-experts/v26ar/hook}
NAME=${NAME:-dsv41-vllm-1017-v26ar-AR}
RUN=${RUN:-/home/milo/dsv41/ar/run}
mkdir -p "$RUN"
sudo -n sh -c "sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory" 2>/dev/null || true
docker run -d --name "$NAME" --gpus all --ipc host --network host --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK \
  -v "$NEW":/model:ro -v /mnt/miloark:/mnt/miloark:ro -v "$CACHE":/root/.cache/vllm \
  -v "$HOOK/sitecustomize.py":/usr/lib/python3.12/sitecustomize.py:ro -v "$HOOK":/w:ro -v /home/milo/pin-hot-experts/v16:/home/milo/pin-hot-experts/v16 \
  -v "$RUN":/ar \
  -e PIN_MODE=adaptive -e PIN_COUNTER=atomic -e PIN_ADAPTIVE_FREEZE=1 \
  -e PIN_ADAPTIVE_GO=/ar/GO -e PIN_CTRL=/ar/ctrl.json -e PIN_TELEMETRY=/ar/telemetry.jsonl -e PIN_ADAPTIVE_LOG=/ar/swaps.jsonl \
  -e PIN_COUNT_TMAX=256 -e PIN_SWAPS_PER_BARRIER=8 \
  -e PIN_HOOK=/w/pin_hot_experts_hook.py -e PIN_ROWMAP=/w/rowmap-static-v1.json -e PYTORCH_CUDA_ALLOC_CONF=pinned_max_cached_size_mb:1024 -e VLLM_LOGGING_LEVEL=INFO -e VLLM_ENGINE_READY_TIMEOUT_S=7200 \
  "$IMAGE" \
  --model /model --served-model-name dsv41-flash-uva --trust-remote-code --tensor-parallel-size 1 \
  --offload-backend uva --cpu-offload-gb 60 --cpu-offload-params routed_experts.w13_weight routed_experts.w2_weight \
  --engram-config '{"cpu_offload": true}' --max-model-len 1048576 --max-num-seqs 24 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.97 \
  --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 --enable-auto-tool-choice --long-prefill-token-threshold 6144 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":[[1,8,5],[9,24,1]]}' \
  --cudagraph-capture-sizes 1 2 4 6 8 12 16 18 24 32 40 48 64 96 128 \
  --host 0.0.0.0 --port 30006 >/dev/null
echo "launched $NAME (run dir $RUN)"
