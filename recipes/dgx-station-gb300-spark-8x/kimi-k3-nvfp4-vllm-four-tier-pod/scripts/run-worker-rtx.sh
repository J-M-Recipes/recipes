#!/bin/bash
# The 9th expert worker on the Station's RTX PRO 6000 (sm_120): plan target rtx0, second Station NIC.
# The RTX is VGA-class and hidden from --gpus by DGX OS: expose it with a CDI spec and pin it by UUID.
#   RTX_UUID=GPU-...  bash run-worker-rtx.sh
: "${RTX_UUID:?set RTX_UUID}"
K3_DIR=${K3_DIR:-$HOME/k3pod}; RTX_BANK_DIR=${RTX_BANK_DIR:-/models/kimi-k3/banks/rtx0}; CACHE=${CACHE:-$K3_DIR/fi-cache}
mkdir -p "$CACHE"
docker rm -f k3-worker-rtx >/dev/null 2>&1
docker run -d --name k3-worker-rtx --device nvidia.com/gpu=$RTX_UUID \
  --ipc=host --network host --ulimit memlock=-1:-1 \
  --device /dev/infiniband/uverbs0 --device /dev/infiniband/uverbs1 \
  -v "$K3_DIR:/w" -v "$RTX_BANK_DIR:/bank:ro" -v "$CACHE:/root/.cache" \
  -e BANK=/bank/k3bank-rtx0.ldb -e PLAN=/w/a3/plan-4tier-v1.json -e TARGET=rtx0 \
  -e BIND_IP=${BIND:-10.0.1.9} -e PORT=${PORT:-19601} -e DEV=${DEV:-mlx5_1} -e PYTHONUNBUFFERED=1 -e K3_WGRAPH=${K3_WGRAPH:-1} \
  nvcr.io/nvidia/pytorch:26.07-py3@sha256:3b0a5c3aa754aff7c86bc0ced07cb064ebce1048f6c12d717e2fa55766276521 \
  /w/situ/venv/bin/python /w/a3/worker.py
