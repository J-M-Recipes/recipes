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
DRAFT_MODEL_DIR="${DRAFT_MODEL_DIR:-}"
CACHE_DIR="${CACHE_DIR:-$HOME/vllm-cache}"
API_KEY_FILE="${API_KEY_FILE:-$HOME/.glm_api_key}"
CAPTURE_DIR="${CAPTURE_DIR:-$RECIPE_DIR/capture}"
SLOT_CACHE_PER_LAYER="${SLOT_CACHE_PER_LAYER:-/w/configs/slots-8400.json}"
AT_KEY="${AT_KEY:-slotcache-S$SLOTS}"
IMAGE="${IMAGE:-vllm-glm53-uva:v0.28.0-2cf0a691}"
CONTAINER_NAME="${CONTAINER_NAME:-glm53-big-$RUN}"
COMPILATION_CONFIG="${COMPILATION_CONFIG:-{\"mode\":3,\"backend\":\"eager\"}}"
# Context-profile overrides. Defaults preserve the measured sc13g flags (8 GiB KV / 65k).
# The live-tested 512K daily profile (see results/2026-09-07-ctx512k-live/) is launched with:
#   KV_CACHE_MEMORY=51539607552 MAX_MODEL_LEN=524288 MAX_NUM_SEQS=1 \
#   SLOT_CACHE_PER_LAYER=/w/configs/slots-5792-ctx512k.json bash scripts/launch-slotcache-portable.sh sc13g-mtp-ctx512k 112 ...
KV_CACHE_MEMORY="${KV_CACHE_MEMORY:-8589934592}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"

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
  --kv-cache-memory "$KV_CACHE_MEMORY"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
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

docker_mounts=(
  -v "$MODEL_DIR:/model:ro"
  -v "$CACHE_DIR:/root/.cache/vllm"
  -v "$RECIPE_DIR:/w:ro"
  -v "$CAPTURE_DIR:/wcap"
  -v "$RECIPE_DIR/patches/sitecustomize.py:/usr/lib/python3.12/sitecustomize.py:ro"
)
if [ -n "$DRAFT_MODEL_DIR" ]; then
  if [ ! -d "$DRAFT_MODEL_DIR" ]; then
    echo "missing DRAFT_MODEL_DIR: $DRAFT_MODEL_DIR" >&2
    exit 2
  fi
  docker_mounts+=( -v "$DRAFT_MODEL_DIR:/draft:ro" )
fi

docker run -d --name "$CONTAINER_NAME" --gpus all --shm-size 32g --network host \
  "${docker_mounts[@]}" \
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
