#!/usr/bin/env bash
# T3 (2026-09-18 overnight): nightly image swap, v18 flags, hook OFF (step 1) or ON (step 2 / T3b).
# Image pre-pulled by Milo with James's approval: vllm/vllm-openai:nightly-dee37d89115db4c94a820a79a78a7828e141c910
# (2026-09-18, vLLM main dee37d89; carries #56935 FlashMLA mega + NVFP4 KV, #56962 Mega-mHC, #56266 Mega-Gate,
#  #56464 DeepSelect, #56568/#57204 MegaMoE, #56512 Engram prefetch). All v18 flag names verified present on main.
# New image = new autotune hash: seed_autotune.sh MUST be armed before launch; 74-min tune is allowed tonight.
# HOOK=0 -> positional offload, no sitecustomize. HOOK=1 -> v15 hook mounted (only after the class check passes).
# Never rm. Never :30003. Never GLM. No further docker pull.
set -euo pipefail
set -f
HOOK_ON=${HOOK:-0}
IMAGE=vllm/vllm-openai:nightly-dee37d89115db4c94a820a79a78a7828e141c910
if [ "$HOOK_ON" = "1" ]; then NAME=${NAME:-dsv41-vllm-T3b-nightly-dee37d89-hook-EXP}; else NAME=${NAME:-dsv41-vllm-T3-nightly-dee37d89-nohook-EXP}; fi
MODEL=/models/DeepSeek-V4.1-Flash-df42c109f1defefcbfcedbe7d905718a12266e40
CACHE=$BOX_HOME/dsv41/vllm-cache
HOOK=$BOX_HOME/pin-hot-experts/hook
DUMP=$BOX_HOME/pin-hot-experts/t3
LOG=$BOX_HOME/dsv41/${NAME}.log
SEQS=24
KSCHED='[[1,4,5],[5,24,1]]'
CGSIZES="1 2 4 6 8 12 16 18 24 32 40 48 64 96 128"

if docker ps --format '{{.Names}}' | grep -q .; then
  echo "REFUSE: a container is running:"; docker ps --format '{{.Names}}'; exit 4
fi
if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "REFUSE: $NAME already exists (stop+rename, never rm)"; exit 5
fi
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "IMAGE NOT PRESENT: $IMAGE (do NOT pull; report)"; exit 3; }
test -f "$MODEL/config.json" || { echo MODEL MISSING; exit 3; }
mkdir -p "$DUMP" "$CACHE"

HOOKMOUNTS=()
HOOKENV=()
if [ "$HOOK_ON" = "1" ]; then
  test -f "$HOOK/sitecustomize.py" && test -f "$HOOK/pin_hot_experts_hook.py" || { echo HOOK MISSING; exit 3; }
  HOOKMOUNTS=(-v "$HOOK/sitecustomize.py":/usr/lib/python3.12/sitecustomize.py:ro -v "$HOOK":/w:ro)
  HOOKENV=(-e PIN_MODE=split -e PIN_HOOK=/w/pin_hot_experts_hook.py -e PIN_ROWMAP=/w/rowmap-static-v1.json)
fi

docker run -d --name "$NAME" --gpus all --ipc host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK \
  --security-opt label=disable \
  -v "$MODEL":/model:ro \
  -v /mnt/miloark:/mnt/miloark:ro \
  -v "$CACHE":/root/.cache/vllm \
  "${HOOKMOUNTS[@]}" \
  -v "$DUMP":$BOX_HOME/pin-hot-experts/t3 \
  -e VLLM_LOGGING_LEVEL=INFO \
  "${HOOKENV[@]}" \
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

echo "launched $NAME image=$IMAGE hook=$HOOK_ON"
: > "$LOG"
nohup docker logs -f --since 1s "$NAME" >> "$LOG" 2>&1 &
disown
echo "log: $LOG"
