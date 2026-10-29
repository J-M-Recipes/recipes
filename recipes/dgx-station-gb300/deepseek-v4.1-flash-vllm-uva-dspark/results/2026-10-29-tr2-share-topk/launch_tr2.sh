#!/usr/bin/env bash
# launch_tr2.sh — Card TR2 boot (2026-10-29, Milo). Clone of launch_br3.sh (sha1 8066045c, the v28-prob
# launcher used by Cards BR4-BR6) with ONE addition: a second sitecustomize that installs /l2/share_topk.py
# (L2 top-k table sharing) on top of the pin-hot hook. Every other arg, mount, env var and the autotune cache
# are byte-identical, so a python-only overlay keeps the FlashInfer autotune hash (hash-hit boot expected).
#   A / A2 : L2_SHARE=0  -> the patch installs and reports, every kernel still runs (behaviour unchanged)
#   B      : L2_SHARE=2  -> relaxed sharing: skip the op for layers that are not the index source
#   C0     : L2_SHARE=0 with L2_STATS_EVERY=1 -> instrument boot, mechanism only
set -euo pipefail
NEW=/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170
IMAGE=vllm/vllm-openai:deepseekv41-flash-0909
CACHE=~/dsv41/vllm-cache
HOOK=${HOOK:-~/pin-hot-experts/v26ar2/hook}
L2=${L2:-~/dsv41/m3/tr2}
NAME=${NAME:?NAME required}
RUN=${RUN:?NAME required}
ROWMAP=${ROWMAP:-rowmap-mixed-ar-v1.json}
PIN_STEP_RING=${PIN_STEP_RING:-65536}
PIN_SWAP_CUDA_EVENTS=${PIN_SWAP_CUDA_EVENTS:-1}
SPEC_EXTRA=${SPEC_EXTRA:-}
L2_SHARE=${L2_SHARE:-2}
L2_VERIFY=${L2_VERIFY:-0}
L2_STATS_EVERY=${L2_STATS_EVERY:-64}
[[ -s $HOOK/$ROWMAP ]] || { echo "launch_tr2: $HOOK/$ROWMAP missing" >&2; exit 3; }
for f in pin_hot_experts_hook.py e4_adaptive.py sitecustomize.py; do [[ -s $HOOK/$f ]] || { echo "launch_tr2: $HOOK/$f missing" >&2; exit 3; }; done
for f in share_topk.py sitecustomize.py; do [[ -s $L2/$f ]] || { echo "launch_tr2: $L2/$f missing" >&2; exit 3; }; done
mkdir -p "$RUN"
sudo -n sh -c "sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory" 2>/dev/null || true
docker run -d --name "$NAME" --gpus all --ipc host --network host --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK \
  -v "$NEW":/model:ro -v /mnt/miloark:/mnt/miloark:ro -v "$CACHE":/root/.cache/vllm \
  -v "$L2/sitecustomize.py":/usr/lib/python3.12/sitecustomize.py:ro -v "$L2":/l2:ro \
  -v "$HOOK":/w:ro -v "$HOME/pin-hot-experts/v16:$HOME/pin-hot-experts/v16" \
  -v "$RUN":/ar \
  -e PIN_MODE=adaptive -e PIN_COUNTER=atomic -e PIN_ADAPTIVE_FREEZE=1 \
  -e PIN_ADAPTIVE_GO=/ar/GO -e PIN_CTRL=/ar/ctrl.json -e PIN_TELEMETRY=/ar/telemetry.jsonl -e PIN_ADAPTIVE_LOG=/ar/swaps.jsonl \
  -e PIN_COUNT_TMAX=256 -e PIN_SWAPS_PER_BARRIER=8 \
  -e PIN_STEP_RING="$PIN_STEP_RING" -e PIN_SWAP_CUDA_EVENTS="$PIN_SWAP_CUDA_EVENTS" \
  -e PIN_HOOK=/w/pin_hot_experts_hook.py -e PIN_ROWMAP="/w/$ROWMAP" \
  -e L2_SHARE="$L2_SHARE" -e L2_VERIFY="$L2_VERIFY" -e L2_STATS_EVERY="$L2_STATS_EVERY" -e L2_HOOK=/l2/share_topk.py \
  -e PYTORCH_CUDA_ALLOC_CONF=pinned_max_cached_size_mb:1024 -e VLLM_LOGGING_LEVEL=INFO -e VLLM_ENGINE_READY_TIMEOUT_S=7200 \
  "$IMAGE" \
  --model /model --served-model-name dsv41-flash-uva --trust-remote-code --tensor-parallel-size 1 \
  --offload-backend uva --cpu-offload-gb 60 --cpu-offload-params routed_experts.w13_weight routed_experts.w2_weight \
  --engram-config '{"cpu_offload": true}' --max-model-len 1048576 --max-num-seqs 24 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.97 \
  --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 --enable-auto-tool-choice --long-prefill-token-threshold 6144 \
  --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":5,\"num_speculative_tokens_per_batch_size\":[[1,8,5],[9,24,1]]$SPEC_EXTRA}" \
  --cudagraph-capture-sizes 1 2 4 6 8 12 16 18 24 32 40 48 64 96 128 \
  --host 0.0.0.0 --port 30006 >/dev/null
echo "launched $NAME (run $RUN, rowmap $ROWMAP, L2_SHARE=$L2_SHARE verify=$L2_VERIFY stats=$L2_STATS_EVERY)"