#!/usr/bin/env bash
# launch-muse.sh — Muse Spark Open 816B-A42B NVFP4 on ONE DGX Station GB300 (recipe lane, 2026-10-16).
# vLLM day-0 image, UVA expert offload of 310 GiB to Grace, CUDA graphs on, FlashInfer autotune OFF, DSpark k=2, 128K.
# This is the exact argv of run q-k2 (results/2026-10-16-muse-s1/runs/q-k2/docker-run-argv.txt) with paths as variables.
# scripts/muse-boot.sh is the measured runner (refuse-if-busy, host-memory watchdog, gates, stop+keep) around the same argv.
set -euo pipefail
MODEL=${MODEL:-/models/Muse-Spark-Open-816B-A42B-NVFP4/4239e4db19376904f2e82c167e78f70abb5455fe/original}   # meta-models/...-NVFP4 @4239e4db
DSPARK=${DSPARK:-/models/Muse-Spark-Open-816B-A42B-NVFP4/dspark/ffc3eb50}                                    # Inferact/Muse-Spark-Open-DSpark @ffc3eb50
CACHE=${CACHE:-/models/Muse-Spark-Open-816B-A42B-NVFP4/cache}
GPU=${GPU:-device=1}            # the GB300; on DGX OS index 0 is the display GPU. Do NOT set CUDA_VISIBLE_DEVICES.
NAME=${NAME:-muse}; PORT=${PORT:-18100}; OFF=${OFF:-310}; K=${K:-2}
IMG=vllm/vllm-openai@sha256:c91025b5be1039a93ca975d1d44f352b0e770bb6c8b142b5056cf92204ca137c   # vllm/vllm-openai:muse-spark-open
HOOK=${HOOK:-}   # measured runs bind-mounted scripts/sitecustomize.py (autotune-cache-path hook; inert with autotune off). Optional.
docker run -d --name "$NAME" --gpus "\"$GPU\"" --ipc=host --network=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /models:/models:ro -v "$CACHE":/root/.cache \
  -e VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  ${HOOK:+-v "$HOOK":/usr/lib/python3.12/sitecustomize.py:ro} \
  "$IMG" "$MODEL" --served-model-name muse --host 127.0.0.1 --port "$PORT" --tensor-parallel-size 1 \
  --offload-backend uva --cpu-offload-gb "$OFF" --cpu-offload-params experts \
  --max-model-len 131072 --max-num-seqs 4 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.92 \
  --kernel-config '{"enable_flashinfer_autotune":false}' \
  --speculative-config "{\"method\":\"dspark\",\"model\":\"$DSPARK\",\"num_speculative_tokens\":$K}" \
  --reasoning-parser muse --tool-call-parser muse --enable-auto-tool-choice
echo "launched $NAME; expect /health 200 in ~6 min (366 s measured). Watch MemAvailable: lowest recorded during boot 116-142 GiB."
