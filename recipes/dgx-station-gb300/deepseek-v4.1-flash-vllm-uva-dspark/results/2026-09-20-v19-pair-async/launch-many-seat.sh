#!/usr/bin/env bash
# Launch v16-sched candidate: v15 (pin-hot-experts static-v1) + scheduler-only changes.
# ONE AXIS = scheduler. Hook, rowmap, offload, image, util, batched-tokens all unchanged.
# Changes vs v15:  --max-num-seqs 16 -> 32
#                  --long-prefill-token-threshold 6144 -> 2048  (per-request per-step chunk cap; V1 scheduler
#                     serves `running` FCFS, so 2 long prefills at 6144 starve every later request of budget)
# NOTE: max-num-partial-prefills / max-long-partial-prefills do NOT exist in this build (checked EngineArgs).
#                  k-schedule end 16 -> 32  ([[1,4,5],[5,32,1]])
# Never rm. Drop caches first (caller). Hash may MISS (seqs change graph capture sizes) — budget 80 min.
set -euo pipefail
set -f
NAME=${NAME:-dsv41-vllm-v18-cgsizes-EXP}  # v18 defaults (promoted 2026-09-18): seqs 24, lpt 6144, token capture sizes
IMAGE=vllm/vllm-openai:deepseekv41-flash-0909
MODEL=/models/DeepSeek-V4.1-Flash-df42c109f1defefcbfcedbe7d905718a12266e40
CACHE=/home/milo/dsv41/vllm-cache
HOOK=/home/milo/pin-hot-experts/hook
DUMP=/home/milo/pin-hot-experts/v16
LOG=/home/milo/dsv41/$NAME.log
SEQS=${SEQS:-24}
LPT=${LPT:-6144}
KSCHED=${KSCHED:-"[[1,4,5],[5,$SEQS,1]]"}
CGSIZES=${CGSIZES:-1 2 4 6 8 12 16 18 24 32 40 48 64 96 128}   # v18 default; CGSIZES="" for vLLM defaults (v17). TOKEN counts, space-separated, e.g. "1 2 4 6 8 12 16 18 24 32 40 48 64 96 128"
CGFLAG=(); if [ -n "$CGSIZES" ]; then CGFLAG=(--cudagraph-capture-sizes $CGSIZES); fi
EXTRA=${EXTRA:-}   # 2026-09-20: extra vllm args passthrough (e.g. --async-scheduling); empty = v18 launch unchanged
EXTRAFLAG=(); if [ -n "$EXTRA" ]; then EXTRAFLAG=($EXTRA); fi

if docker ps --format '{{.Names}}' | grep -q .; then
  echo "REFUSE: a container is running:"; docker ps --format '{{.Names}}'
  exit 4
fi
if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "REFUSE: $NAME already exists (stop+rename, never rm)"; exit 5
fi
test -f "$MODEL/config.json" || { echo MODEL MISSING; exit 3; }
test -f "$HOOK/sitecustomize.py" && test -f "$HOOK/pin_hot_experts_hook.py" || { echo HOOK MISSING; exit 3; }
test -f "$HOOK/rowmap-static-v1.json" || { echo ROWMAP MISSING; exit 3; }
mkdir -p "$DUMP" "$CACHE"

docker run -d --name "$NAME" --gpus all --ipc host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK \
  --security-opt label=disable \
  -v "$MODEL":/model:ro \
  -v /mnt/miloark:/mnt/miloark:ro \
  -v "$CACHE":/root/.cache/vllm \
  -v "$HOOK/sitecustomize.py":/usr/lib/python3.12/sitecustomize.py:ro \
  -v "$HOOK":/w:ro \
  -v "$DUMP":/home/milo/pin-hot-experts/v16 \
  -e VLLM_LOGGING_LEVEL=INFO \
  -e PIN_MODE=split \
  -e PIN_HOOK=/w/pin_hot_experts_hook.py \
  -e PIN_ROWMAP=/w/rowmap-static-v1.json \
  "$IMAGE" \
  --model /model --served-model-name dsv41-flash-uva --trust-remote-code \
  --tensor-parallel-size 1 \
  --offload-backend uva --cpu-offload-gb 60 \
  --cpu-offload-params routed_experts.w13_weight routed_experts.w2_weight \
  --engram-config '{"cpu_offload": true}' \
  --max-model-len 1048576 --max-num-seqs "$SEQS" --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.97 \
  --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 --enable-auto-tool-choice \
  --long-prefill-token-threshold "$LPT" \
  --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":5,\"num_speculative_tokens_per_batch_size\":$KSCHED}" \
  ${CGFLAG[@]+"${CGFLAG[@]}"} \
  ${EXTRAFLAG[@]+"${EXTRAFLAG[@]}"} \
  --host 0.0.0.0 --port 30006

echo "launched $NAME  seqs=$SEQS lpt=$LPT ksched=$KSCHED cgsizes=${CGSIZES:-default} extra=[${EXTRA}]"
: > "$LOG"
nohup docker logs -f --since 1s "$NAME" >> "$LOG" 2>&1 &
disown
echo "log: $LOG"
