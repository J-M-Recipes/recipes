#!/bin/bash
# Build the worker Python env on each Spark and on the Station (for the RTX worker): FlashInfer 0.7.0.post1 in a venv
# on top of the NGC PyTorch 26.07 image, plus the SiTU activation patch (Kimi-K3's expert activation is SiTU-GLU,
# 4*tanh(g/4)*sigmoid(g) * 25*tanh(u/25); stock FlashInfer B12x MoE kernels support silu/gelu/relu2 only).
#   K3_DIR=$HOME/k3pod bash setup-worker-venv.sh        (needs patches/b12x-situ.patch next to this script's dir)
set -euo pipefail
K3_DIR=${K3_DIR:-$HOME/k3pod}
HERE=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$K3_DIR/situ"
cp "$HERE/patches/b12x-situ.patch" "$K3_DIR/situ/"
docker run --rm -v "$K3_DIR:/w" --entrypoint bash \
  nvcr.io/nvidia/pytorch:26.07-py3@sha256:3b0a5c3aa754aff7c86bc0ced07cb064ebce1048f6c12d717e2fa55766276521 -c '
set -e
python3 -m venv --system-site-packages /w/situ/venv
/w/situ/venv/bin/pip install -q flashinfer-python==0.7.0.post1
D=/w/situ/venv/lib/python3.12/site-packages/flashinfer/fused_moe/cute_dsl
patch -p1 -d "$D" --forward < /w/situ/b12x-situ.patch
grep -q "\"situ\"" "$D/blackwell_sm12x/moe_activation.py" && echo "SiTU patch applied"
/w/situ/venv/bin/python -c "import flashinfer; print(\"flashinfer\", flashinfer.__version__)"'
# Reference venv (2026-10-30): flashinfer_python 0.7.0.post1, nvidia_cutlass_dsl 4.8.0, cuda_python 13.4.1,
# apache_tvm_ffi 0.1.14.post1, cuda_tile 1.6.0, nvidia_cudnn_frontend 1.30.0, nccl4py 0.6.0.
