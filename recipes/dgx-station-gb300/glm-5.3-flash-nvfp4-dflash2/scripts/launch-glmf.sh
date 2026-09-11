#!/usr/bin/env bash
# GLM-5.3-Flash experiment launcher on GB300 — lane :30001, containers glmf-<TAG>. No --restart (stop+keep).
# Knobs: TAG MODEL DRAFT IMAGE SPEC(dflash|eagle|none) KV(fp8_e4m3|auto) MEM CTX MAXBS EXTRA
set -euo pipefail
TAG="${TAG:?}"; MODEL="${MODEL:?local checkpoint dir}"; IMAGE="${IMAGE:?}"
DRAFT="${DRAFT:-/models/GLM-5.3-Flash-DFlash2}"
SPEC="${SPEC:-dflash}"; KV="${KV:-fp8_e4m3}"; MEM="${MEM:-0.85}"; CTX="${CTX:-131072}"; MAXBS="${MAXBS:-32}"
PORT="${PORT:-30001}"; EXTRA="${EXTRA:-}"; DOCKER_ENV="${DOCKER_ENV:--e TORCHINDUCTOR_COMPILE_THREADS=1}"
NAME=glmf-$TAG; LOGDIR="${LOGDIR:-$HOME/glmf}"; mkdir -p $LOGDIR; LOG=$LOGDIR/$NAME.log
test -f "$MODEL/config.json" || { echo MODEL MISSING; exit 3; }
case "$SPEC" in
  dflash) SPECARGS="--speculative-algorithm DFLASH --speculative-draft-model-path /draft --speculative-draft-attention-backend fa4"; DRAFTMOUNT="-v $DRAFT:/draft:ro" ;;
  eagle)  SPECARGS="--speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 --speculative-adaptive"; DRAFTMOUNT="" ;;
  none)   SPECARGS=""; DRAFTMOUNT="" ;;
  *) echo "bad SPEC"; exit 2 ;;
esac
KVARGS=""; [ "$KV" != "auto" ] && KVARGS="--kv-cache-dtype $KV"
CGFLAG="${CGFLAG:---cuda-graph-max-bs-decode}"   # old frozen v2 image: CGFLAG=--cuda-graph-max-bs
docker rm -f "$NAME" 2>/dev/null || true
docker run -d --name "$NAME" --gpus all --shm-size 32g --network host $DOCKER_ENV \
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
echo "launched $NAME image=$IMAGE spec=$SPEC kv=$KV mem=$MEM ctx=$CTX bs=$MAXBS extra=[$EXTRA] env=[$DOCKER_ENV]"; echo "log: $LOG"
