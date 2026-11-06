#!/usr/bin/env bash
# build.sh — engine + weights for the Kimi-K3 Q0.0086 joke build.
# Step 2 downloads 56.10 GB of a 594 GB checkpoint (9.4%) and never asks for the rest.
set -euo pipefail

LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
WORKDIR="${WORKDIR:-$HOME/k3-01bit}"
LAYERS="${LAYERS:-12}"
TOP_K="${TOP_K:-2}"
JOBS="${JOBS:-8}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$WORKDIR"
cd "$WORKDIR"

# 1. engine with K3 (KDA + attention residuals + latent MoE) support
if [[ ! -d "$LLAMA_DIR" ]]; then
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
fi
cmake -B "$LLAMA_DIR/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 \
      -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$LLAMA_DIR/build" -j "$(nproc)" --target llama-cli llama-server llama-quantize llama-bench

# 2. range-fetch 16 of 896 expert slabs per layer, everything else whole
python3 "$HERE/k3_slice.py" plan --dir UD-IQ1_S --shards 14
python3 "$HERE/k3_slice.py" build k3-iq1s-sliced.gguf --dir UD-IQ1_S --shards 14 --jobs "$JOBS"

# 3. depth + active-expert pruning, then stock Q1_0
python3 "$HERE/local_subset.py" k3-iq1s-sliced.gguf "k3-${LAYERS}l.gguf" --layers "$LAYERS" --expert-used "$TOP_K"
"$LLAMA_DIR/build/bin/llama-quantize" --allow-requantize "k3-${LAYERS}l.gguf" "k3-q01-L${LAYERS}.gguf" Q1_0

# 4. prove the arithmetic (declared vs real bits per parameter)
python3 "$HERE/receipts.py" "k3-q01-L${LAYERS}.gguf"
