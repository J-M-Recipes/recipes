#!/usr/bin/env bash
# T4 (2026-09-18 overnight): E4b D1 adaptive counter on **v18** flags, gated at C8/C16.
# = launch-many-seat.sh (v18: seqs 24, ksched [[1,4,5],[5,24,1]], lpt 6144, token capture sizes)
#   + PIN_MODE=adaptive PIN_COUNTER=d1 (fixed-slice snap, no ring), drain parked on a GO file.
# Why: E4b measured the D1 tax at C1 -1.68 / C8 -1.26 / C16 -0.94 %. The verdict for adaptive
# placement belongs at C8/C16 where the per-step tax is smallest and off-profile traffic (JP/DE) is
# where swaps can earn it back without a per-domain reboot. Never rm. Never :30003. Never GLM.
set -euo pipefail
set -f
NAME=${NAME:-dsv41-vllm-T4-adaptive-d1-unfrozen-C8gate-EXP}
IMAGE=vllm/vllm-openai:deepseekv41-flash-0909
MODEL=/models/DeepSeek-V4.1-Flash-df42c109f1defefcbfcedbe7d905718a12266e40
CACHE=$BOX_HOME/dsv41/vllm-cache
HOOK=$BOX_HOME/pin-hot-experts/hook
DUMP=$BOX_HOME/pin-hot-experts/t4
LOG=$BOX_HOME/dsv41/${NAME}.log
FREEZE=${PIN_ADAPTIVE_FREEZE:-0}      # T4 runs UNFROZEN (swaps live) after the GO file is dropped
COUNTER=${PIN_COUNTER:-d1}
SEQS=24
KSCHED='[[1,4,5],[5,24,1]]'
CGSIZES="1 2 4 6 8 12 16 18 24 32 40 48 64 96 128"

if docker ps --format '{{.Names}}' | grep -q .; then
  echo "REFUSE: a container is running:"; docker ps --format '{{.Names}}'; exit 4
fi
if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "REFUSE: $NAME already exists (stop+rename, never rm)"; exit 5
fi
test -f "$MODEL/config.json" || { echo MODEL MISSING; exit 3; }
test -f "$HOOK/sitecustomize.py" && test -f "$HOOK/pin_hot_experts_hook.py" || { echo HOOK MISSING; exit 3; }
test -f "$HOOK/e4_adaptive.py" || { echo E4 ADAPTIVE MISSING; exit 3; }
test -f "$HOOK/rowmap-static-v1.json" || { echo ROWMAP MISSING; exit 3; }
mkdir -p "$DUMP" "$CACHE"
# Drain must stay parked through graph capture. Never inherit an older GO file.
rm -f "$DUMP/GO" $BOX_HOME/pin-hot-experts/e4/GO $BOX_HOME/pin-hot-experts/e4b/GO
: > "$DUMP/swaps.jsonl"

docker run -d --name "$NAME" --gpus all --ipc host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK \
  --security-opt label=disable \
  -v "$MODEL":/model:ro \
  -v /mnt/miloark:/mnt/miloark:ro \
  -v "$CACHE":/root/.cache/vllm \
  -v "$HOOK/sitecustomize.py":/usr/lib/python3.12/sitecustomize.py:ro \
  -v "$HOOK":/w:ro \
  -v "$DUMP":$BOX_HOME/pin-hot-experts/t4 \
  -e VLLM_LOGGING_LEVEL=INFO \
  -e PIN_MODE=adaptive \
  -e PIN_HOOK=/w/pin_hot_experts_hook.py \
  -e PIN_ROWMAP=/w/rowmap-static-v1.json \
  -e PIN_COUNTER="$COUNTER" \
  -e PIN_ADAPTIVE_RING=0 \
  -e PIN_ADAPTIVE_FREEZE="$FREEZE" \
  -e PIN_ADAPTIVE_LOG=$BOX_HOME/pin-hot-experts/t4/swaps.jsonl \
  -e PIN_ADAPTIVE_GO=$BOX_HOME/pin-hot-experts/t4/GO \
  -e PIN_SWAPS_PER_LAYER_PER_DRAIN=2 \
  -e PIN_EWMA_HALFLIFE=500 \
  -e PIN_DRAIN_N=64 \
  "$IMAGE" \
  --model /model --served-model-name dsv41-flash-uva --trust-remote-code \
  --tensor-parallel-size 1 \
  --offload-backend uva --cpu-offload-gb 60 \
  --cpu-offload-params routed_experts.w13_weight routed_experts.w2_weight \
  --engram-config '{"cpu_offload": true}' \
  --max-model-len 1048576 --max-num-seqs "$SEQS" --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.97 \
  --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 --enable-auto-tool-choice \
  --long-prefill-token-threshold 6144 \
  --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":5,\"num_speculative_tokens_per_batch_size\":$KSCHED}" \
  --cudagraph-capture-sizes $CGSIZES \
  --host 0.0.0.0 --port 30006

echo "launched $NAME freeze=$FREEZE counter=$COUNTER seqs=$SEQS"
: > "$LOG"
nohup docker logs -f --since 1s "$NAME" >> "$LOG" 2>&1 &
disown
echo "log: $LOG"
