#!/usr/bin/env bash
set -euo pipefail

readonly ROOT=<lane-root>
readonly LANE="$ROOT/v22"
readonly MODEL_CACHE=<hf-home>/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017
readonly MODEL="$MODEL_CACHE/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170"
readonly IMAGE='vllm/vllm-openai@sha256:91c875065792e9b79dfec5019e5017f96b1511eed292e7ec15182444b23efe66'
readonly NAME='dsv41-recipe-v22'

[[ "$(systemctl --user is-active vllm-current.service 2>/dev/null || true)" != active ]] || { echo 'vllm-current.service is active' >&2; exit 1; }
[[ -e "$MODEL/config.json" ]] || { echo 'Pinned 1017 snapshot (1bae0e10) is missing' >&2; exit 1; }
[[ "$(docker ps -q --filter name='^dsv41-recipe-v18-exact$')" == "" ]] || { echo 'V18 rollback is still running' >&2; exit 1; }
docker image inspect "$IMAGE" >/dev/null
sha256sum -c "$LANE/results/hook-artifacts.sha256" >/dev/null

if docker container inspect "$NAME" >/dev/null 2>&1; then
  exec docker start "$NAME"
fi

exec docker run -d --name "$NAME" \
  --gpus all --ipc host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK --cap-add SYS_NICE \
  --security-opt label=disable \
  -v "$MODEL":/model:ro \
  -v "$MODEL_CACHE/blobs":/blobs:ro \
  -v "$LANE/cache":/root/.cache/vllm \
  -v "$LANE/hook/sitecustomize.py":/usr/lib/python3.12/sitecustomize.py:ro \
  -v "$LANE/hook":/w:ro \
  -v "$LANE/dump":/results \
  -e VLLM_LOGGING_LEVEL=INFO \
  -e PIN_MODE=split \
  -e PIN_HOOK=/w/pin_hot_experts_hook.py \
  -e PIN_ROWMAP=/w/rowmap-static-v1.json \
  -e PIN_DUMP_DIR=/results \
  --entrypoint /usr/bin/numactl \
  "$IMAGE" \
  --membind=0 /usr/local/bin/vllm serve \
  --model /model --served-model-name current --trust-remote-code \
  --tensor-parallel-size 1 \
  --offload-backend uva --cpu-offload-gb 54 \
  --kv-cache-dtype fp8_ds_mla \
  --cpu-offload-params routed_experts.w13_weight routed_experts.w2_weight \
  --engram-config '{"cpu_offload": true}' \
  --max-model-len 1048576 --max-num-seqs 24 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.97 \
  --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 --enable-auto-tool-choice \
  --long-prefill-token-threshold 6144 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":[[1,4,5],[5,24,1]]}' \
  --cudagraph-capture-sizes 1 2 4 6 8 12 16 18 24 32 40 48 64 96 128 \
  --host 0.0.0.0 --port 8000
