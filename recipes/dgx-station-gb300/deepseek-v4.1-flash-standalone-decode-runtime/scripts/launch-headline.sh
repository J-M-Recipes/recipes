#!/usr/bin/env bash
# launch-headline.sh - run the DSV4.1 standalone decode runtime's headline instrument on ONE DGX Station GB300.
#
# This is a research runtime, not a server: the container runs bench/headline.py once (boot ~60 s, then 8 fixture
# prompts x 256 new tokens, two columns), writes a JSON receipt and exits. No port is opened (--network none).
#
# Required env
#   MODEL_DIR   deepseek-ai/DeepSeek-V4.1-Flash-1017 @ 1bae0e1088993b9b82de08f59de5be5a34e42170, on local NVMe (never CIFS)
#   WORK        checkout of https://github.com/jmeadlock/gb300-dsv41-decode at the pinned commit (mounted at /work)
#   GPU         CDI device name for the GB300, e.g. the UUID from `nvidia-ctk cdi list`. Index 0 may be a display GPU.
# Optional env
#   SCRATCH     receipt/tune-cache dir (default $WORK/scratch)            JIT_DIR   JIT caches (default $HOME/.cache/dsv41-jit)
#   NAME        container name (default dsv41-decode)                    OUT       receipt path in the container (default scratch/headline.json)
#   IMAGE       default vllm/vllm-openai:v0.31.0 (used as a CUDA/torch/DeepGEMM/FlashMLA/TileLang library image; vllm is never imported)
#   ALLOW_OTHER_COMMIT=1 / ALLOW_IMAGE_MISMATCH=1 / ALLOW_OTHER_CHECKPOINT=1   override the three pin checks below
# Extra arguments are appended to bench/headline.py.
#
# NOT executed end to end by the recipe writer (no box access): passes `bash -n` and a stub-docker render only.
set -euo pipefail

PIN_COMMIT=ed6582e051b9b743f1a68b88750c627d2a63ca03
PIN_IMAGE_DIGEST=sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3
PIN_CONFIG_SHA256=8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879   # sha256 of the checkpoint's config.json, from the receipt

: "${MODEL_DIR:?set MODEL_DIR to the 1bae0e10 checkpoint directory}"
: "${WORK:?set WORK to a checkout of gb300-dsv41-decode at ${PIN_COMMIT:0:7}}"
: "${GPU:?set GPU to the CDI device name of the GB300 (nvidia.com/gpu=\$GPU)}"
SCRATCH=${SCRATCH:-$WORK/scratch}
JIT_DIR=${JIT_DIR:-$HOME/.cache/dsv41-jit}
NAME=${NAME:-dsv41-decode}
OUT=${OUT:-scratch/headline.json}
IMAGE=${IMAGE:-vllm/vllm-openai:v0.31.0}

sha256_of() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

# --- pin checks (each can be overridden, but a receipt from an overridden run is not this recipe's number) ---
head_sha=$(git -C "$WORK" rev-parse HEAD 2>/dev/null || echo none)
if [[ "$head_sha" != "$PIN_COMMIT" && "${ALLOW_OTHER_COMMIT:-0}" != 1 ]]; then
  echo "WORK is at $head_sha, recipe pins $PIN_COMMIT (ALLOW_OTHER_COMMIT=1 to override)" >&2; exit 2
fi
cfg_sha=$(sha256_of "$MODEL_DIR/config.json")
if [[ "$cfg_sha" != "$PIN_CONFIG_SHA256" && "${ALLOW_OTHER_CHECKPOINT:-0}" != 1 ]]; then
  echo "MODEL_DIR/config.json sha256 $cfg_sha != pinned $PIN_CONFIG_SHA256 (ALLOW_OTHER_CHECKPOINT=1 to override)" >&2; exit 2
fi
if ! docker image inspect "$IMAGE" --format '{{.Id}} {{json .RepoDigests}}' | grep -q "$PIN_IMAGE_DIGEST"; then
  if [[ "${ALLOW_IMAGE_MISMATCH:-0}" != 1 ]]; then
    echo "image $IMAGE does not report $PIN_IMAGE_DIGEST as its id or a repo digest (ALLOW_IMAGE_MISMATCH=1 to override)" >&2; exit 2
  fi
fi

mkdir -p "$SCRATCH" "$JIT_DIR"/{tilelang,triton,nv,dg}

# The headline command line is the one every receipt in results/2026-11-08-v1-recipe/raw/ records in its `argv`.
# DSV41_PREFILL / DSV41_PREFILL_STREAMS are the defaults at the pinned commit; they are passed explicitly because
# bench/box/m4p2.sh (the run behind headline-m4p2-20261108.json) passed them explicitly.
exec docker run --rm --name "$NAME" --network none \
  --device "nvidia.com/gpu=$GPU" --ipc=host --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  -v "$WORK":/work -w /work \
  -v "$MODEL_DIR":/ref:ro \
  -v "$SCRATCH":/work/scratch \
  -v "$JIT_DIR/tilelang":/root/.tilelang -v "$JIT_DIR/triton":/root/.triton -v "$JIT_DIR/nv":/root/.nv -v "$JIT_DIR/dg":/root/.deep_gemm \
  -e DG_JIT_CACHE_DIR=/root/.deep_gemm \
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  -e DSV41_MODEL_DIR=/ref \
  -e DSV41_GEMV_CACHE=/work/scratch/gemv-tune.json \
  -e DSV41_PREFILL=grouped -e DSV41_PREFILL_STREAMS=4 \
  --entrypoint python3 "$IMAGE" \
  bench/headline.py --model-dir /ref --moe-impl dg-inplace --fp8-impl gemv --glue-impl fused --attn-impl flashmla \
  --fp32-impl fixed --moe-rowmatch on --spare-slots 16 --residency-drain 64 --max-new 256 --json "$OUT" "$@"
