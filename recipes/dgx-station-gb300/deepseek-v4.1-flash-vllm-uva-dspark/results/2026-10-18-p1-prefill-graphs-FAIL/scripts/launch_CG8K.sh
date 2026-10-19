#!/usr/bin/env bash
# launch_v22.sh — the exact v22 launch (promoted 2026-10-17 16:55): v21 config with k-schedule [[1,8,5],[9,24,1]]. Same autotune hash as v21 (num_speculative_tokens unchanged).
# Rollback: docker stop dsv41-vllm-1017-v22-ksched1to8-BOUND-REF && docker start dsv41-vllm-1017-v21cfg-RETIRED-REF
set -euo pipefail
NEW=/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170
IMAGE=vllm/vllm-openai:deepseekv41-flash-0909
CACHE=$HOME/dsv41/vllm-cache
HOOK=$HOME/pin-hot-experts/v19/hook
NAME=${1:-dsv41-vllm-1017-v22-ksched1to8-BOUND-REF}
sudo -n sh -c "sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory" 2>/dev/null || true
docker run -d --name "$NAME" --gpus all --ipc host --network host --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK \
  -v "$NEW":/model:ro -v /mnt/miloark:/mnt/miloark:ro -v "$CACHE":/root/.cache/vllm \
  -v "$HOOK/sitecustomize.py":/usr/lib/python3.12/sitecustomize.py:ro -v "$HOOK":/w:ro -v $HOME/pin-hot-experts/v16:$HOME/pin-hot-experts/v16 \
  -e PIN_MODE=split -e PIN_HOOK=/w/pin_hot_experts_hook.py -e PIN_ROWMAP=/w/rowmap-static-v1.json -e VLLM_LOGGING_LEVEL=INFO -e VLLM_ENGINE_READY_TIMEOUT_S=7200 \
  "$IMAGE" \
  --model /model --served-model-name dsv41-flash-uva --trust-remote-code --tensor-parallel-size 1 \
  --offload-backend uva --cpu-offload-gb 60 --cpu-offload-params routed_experts.w13_weight routed_experts.w2_weight \
  --engram-config '{"cpu_offload": true}' --max-model-len 1048576 --max-num-seqs 24 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.97 \
  --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 --enable-auto-tool-choice --long-prefill-token-threshold 6144 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":[[1,8,5],[9,24,1]]}' \
  --cudagraph-capture-sizes 1 2 4 6 8 12 16 18 24 32 40 48 64 96 128 256 512 1024 2048 4096 8192 \
  --host 0.0.0.0 --port 30006 >/dev/null
echo "launched $NAME"
