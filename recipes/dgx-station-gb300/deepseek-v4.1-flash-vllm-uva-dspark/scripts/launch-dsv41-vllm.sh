#!/usr/bin/env bash
# launch-dsv41-vllm.sh — DeepSeek-V4.1-Flash on one GB300 via vLLM dsv41-feat, UVA selective expert offload + Engram host.
# Experiment lane :30006. NOT production. No --restart.
set -euo pipefail
TAG="${TAG:-v1}"
OFFGB="${OFFGB:-105}"
UTIL="${UTIL:-0.92}"
SEQS="${SEQS:-8}"
SPEC="${SPEC:-}"   # e.g. SPEC=dspark:5
CTX="${CTX:-131072}"
EXTRA="${EXTRA:-}"   # extra vllm args, e.g. partial-prefill knobs
DOCKER_ENV="${DOCKER_ENV:-}"   # extra container env, e.g. DOCKER_ENV="-e VLLM_USE_RUST_FRONTEND=1 -e VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1"
PARAMS="${PARAMS:-routed_experts.w13_weight routed_experts.w2_weight}"
IMAGE=vllm/vllm-openai:deepseekv41-flash-0909
SHA=df42c109f1defefcbfcedbe7d905718a12266e40
MODEL="${MODEL:?set MODEL to the local checkpoint dir (revision $SHA)}"
NAME=dsv41-vllm-$TAG
LOG="${LOGDIR:-$HOME/dsv41}/$NAME.log"; mkdir -p "$(dirname "$LOG")"
CACHE="${CACHE:-$HOME/vllm-cache}"; mkdir -p "$CACHE"   # persistent: FlashInfer autotune cache lives here
test -f "$MODEL/config.json" || { echo MODEL MISSING; exit 3; }
docker ps --format '{{.Names}}' | grep -qE '^(dsfv-|dsv41-dryrun)' && { echo "another GPU container running; refuse"; docker ps --format '{{.Names}}'; exit 4; }
docker rm -f "$NAME" 2>/dev/null || true
docker run -d --name "$NAME" --gpus all --ipc host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK \
  -v "$MODEL":/model:ro -v $CACHE:/root/.cache/vllm \
  -e VLLM_LOGGING_LEVEL=INFO $DOCKER_ENV \
  "$IMAGE" \
  --model /model --served-model-name dsv41-flash-uva --trust-remote-code \
  --tensor-parallel-size 1 \
  --offload-backend uva --cpu-offload-gb "$OFFGB" --cpu-offload-params $PARAMS \
  --engram-config '{"cpu_offload": true}' \
  --max-model-len "$CTX" --max-num-seqs "$SEQS" --max-num-batched-tokens 8192 \
  --gpu-memory-utilization "$UTIL" \
  ${SPEC:+--speculative-config "{\"method\":\"${SPEC%%:*}\",\"num_speculative_tokens\":${SPEC##*:}}"} \
  --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 --enable-auto-tool-choice \
  $EXTRA \
  --host 0.0.0.0 --port 30006
echo "launched $NAME offload=${OFFGB}GB util=$UTIL seqs=$SEQS ctx=$CTX spec=[$SPEC] extra=[$EXTRA] env=[$DOCKER_ENV]"
nohup docker logs -f "$NAME" > "$LOG" 2>&1 &
disown
echo "log: $LOG"
