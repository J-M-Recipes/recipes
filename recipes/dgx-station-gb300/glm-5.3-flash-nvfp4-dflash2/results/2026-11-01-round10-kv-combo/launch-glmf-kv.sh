#!/usr/bin/env bash
# launch-glmf-kv.sh — copy of launch-glmf.sh for Card KV1 (2026-11-01). Two changes, both commented:
#  (1) --gpus "device=<GB300 UUID>" instead of --gpus all: the RTX PRO 6000 is now a visible compute GPU (index 0).
#  (2) DRAFT_BACKEND env (default fa4 = daily) feeds --speculative-draft-attention-backend.
set -euo pipefail
TAG="${TAG:?}"; MODEL="${MODEL:?local checkpoint dir}"; IMAGE="${IMAGE:?}"
DRAFT="${DRAFT:-/models/GLM-5.3-Flash-DFlash2}"
SPEC="${SPEC:-dflash}"; KV="${KV:-fp8_e4m3}"; MEM="${MEM:-0.85}"; CTX="${CTX:-131072}"; MAXBS="${MAXBS:-32}"
PORT="${PORT:-30001}"; EXTRA="${EXTRA:-}"; DOCKER_ENV="${DOCKER_ENV:--e TORCHINDUCTOR_COMPILE_THREADS=1}"
DRAFT_BACKEND="${DRAFT_BACKEND:-fa4}"                                   # (2)
GPU_UUID="${GPU_UUID:?set to the GB300 UUID from nvidia-smi -L}"        # (1) GB300
NAME=glmf-$TAG; LOGDIR=~/glmf; mkdir -p $LOGDIR; LOG=$LOGDIR/$NAME.log
test -f "$MODEL/config.json" || { echo MODEL MISSING; exit 3; }
docker ps --format '{{.Names}}' | grep -qE '^(dsfv-|dsv41-vllm-v11|ds41)' && { echo "prod/ref lane running; refuse"; exit 4; }
case "$SPEC" in
  dflash) SPECARGS="--speculative-algorithm DFLASH --speculative-draft-model-path /draft --speculative-draft-attention-backend $DRAFT_BACKEND"; DRAFTMOUNT="-v $DRAFT:/draft:ro" ;;
  none)   SPECARGS=""; DRAFTMOUNT="" ;;
  *) echo "bad SPEC"; exit 2 ;;
esac
KVARGS=""; [ "$KV" != "auto" ] && KVARGS="--kv-cache-dtype $KV"
CGFLAG="${CGFLAG:---cuda-graph-max-bs-decode}"
docker rm -f "$NAME" 2>/dev/null || true
docker run -d --name "$NAME" --gpus "device=$GPU_UUID" --shm-size 32g --network host $DOCKER_ENV \
  -v "$MODEL":/model:ro $DRAFTMOUNT \
  "$IMAGE" \
  python3 -m sglang.launch_server --model-path /model --host 0.0.0.0 --port $PORT \
    --quantization modelopt_fp4 --trust-remote-code --served-model-name glm-5.3-flash \
    --tool-call-parser glm47 --reasoning-parser glm45 \
    --context-length $CTX --mem-fraction-static $MEM \
    $CGFLAG $MAXBS --max-running-requests $MAXBS \
    --max-prefill-tokens 8192 --chunked-prefill-size 8192 \
    $KVARGS $SPECARGS $EXTRA > /dev/null
nohup docker logs -f "$NAME" > "$LOG" 2>&1 &
echo "launched $NAME image=$IMAGE spec=$SPEC draft_backend=$DRAFT_BACKEND kv=$KV mem=$MEM ctx=$CTX bs=$MAXBS gpu=$GPU_UUID extra=[$EXTRA]"; echo "log: $LOG"
