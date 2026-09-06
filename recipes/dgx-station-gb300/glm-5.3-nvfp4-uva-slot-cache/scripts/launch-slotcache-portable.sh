#!/usr/bin/env bash
# Portable slot-cache launcher for the recipe tree.
# Not live-tested from this packaged path; it preserves the measured sc13g flags.
set -euo pipefail
if [ "$#" -lt 2 ]; then
  echo "usage: $0 <run-name> <slots> [extra vllm args...]" >&2
  exit 2
fi
RUN="$1"; SLOTS="$2"; shift 2
HERE="$(cd "$(dirname "$0")" && pwd)"
RECIPE_DIR="$(cd "$HERE/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-/home/exx/models/GLM-5.3-NVFP4-big}"
CACHE_DIR="${CACHE_DIR:-$HOME/vllm-cache}"
API_KEY_FILE="${API_KEY_FILE:-$HOME/.glm_api_key}"
CAPTURE_DIR="${CAPTURE_DIR:-$RECIPE_DIR/capture}"
SLOT_CACHE_PER_LAYER="${SLOT_CACHE_PER_LAYER:-/w/configs/slots-8400.json}"
AT_KEY="${AT_KEY:-slotcache-S$SLOTS}"
IMAGE="${IMAGE:-vllm-glm53-uva:v0.28.0-2cf0a691}"
CONTAINER_NAME="${CONTAINER_NAME:-glm53-big-$RUN}"
COMPILATION_CONFIG="${COMPILATION_CONFIG:-{\"mode\":3,\"backend\":\"eager\"}}"

if [ ! -r "$API_KEY_FILE" ]; then
  echo "missing readable API_KEY_FILE: $API_KEY_FILE" >&2
  exit 2
fi
API_KEY="$(tr -d '\r\n' < "$API_KEY_FILE")"
if [ -z "$API_KEY" ]; then
  echo "API_KEY_FILE is empty: $API_KEY_FILE" >&2
  exit 2
fi

mkdir -p "$CAPTURE_DIR"

vllm_args=(
  /model --host 0.0.0.0 --port 30001
  --served-model-name glm-5.3-big
  --trust-remote-code
  --quantization modelopt
  --load-format safetensors
  --offload-backend uva
  --cpu-offload-gb 420
  --cpu-offload-params routed_experts.w13_weight routed_experts.w2_weight
  --gpu-memory-utilization 0.95
  --kv-cache-dtype bfloat16
  --kv-cache-memory 8589934592
  --max-model-len 65536
  --max-num-seqs 4
  --max-num-batched-tokens 8192
  --enable-auto-tool-choice
  --tool-call-parser glm47
  --reasoning-parser glm45
)

has_compilation_config=0
for arg in "$@"; do
  if [ "$arg" = "--compilation-config" ]; then
    has_compilation_config=1
    break
  fi
done
if [ "$has_compilation_config" = "0" ]; then
  vllm_args+=(--compilation-config "$COMPILATION_CONFIG")
fi
vllm_args+=("$@")

docker run -d --name "$CONTAINER_NAME" --gpus all --shm-size 32g --network host \
  -v "$MODEL_DIR:/model:ro" \
  -v "$CACHE_DIR:/root/.cache/vllm" \
  -v "$RECIPE_DIR:/w:ro" \
  -v "$CAPTURE_DIR:/wcap" \
  -v "$RECIPE_DIR/patches/sitecustomize.py:/usr/lib/python3.12/sitecustomize.py:ro" \
  -e VLLM_LOGGING_LEVEL=INFO \
  -e "VLLM_AUTOTUNE_CACHE_KEY=$AT_KEY" \
  -e EXACT_PIN=1 \
  -e EXACT_PIN_FILE=/w/patches/exact_pin.py \
  -e "SLOT_CACHE=$SLOTS" \
  -e SLOT_CACHE_HOOK=/w/patches/slot_cache_hook.py \
  -e SLOT_CACHE_CAPTURE="${CAPTURE:-0}" \
  -e SLOT_CACHE_ROUTER="${ROUTER:-ffi}" \
  -e SLOT_CACHE_PER_LAYER="$SLOT_CACHE_PER_LAYER" \
  -e SLOT_CACHE_UNPACKED="${UNPACKED:-0}" \
  -e SLOT_CACHE_LOGIT_RING="${LOGIT_RING:-0}" \
  -e SLOT_CACHE_STATS_SEC="${STATS_SEC:-20}" \
  -e SLOT_CACHE_BYPASS_TOKENS="${BYPASS:-16}" \
  -e "VLLM_API_KEY=$API_KEY" \
  "$IMAGE" \
  "${vllm_args[@]}"

echo "launched $CONTAINER_NAME SLOT_CACHE=$SLOTS offload=420 per_layer=$SLOT_CACHE_PER_LAYER"
