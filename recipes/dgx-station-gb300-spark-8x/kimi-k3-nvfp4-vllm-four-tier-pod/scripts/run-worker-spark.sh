#!/bin/bash
# One Spark expert worker. Usage: run-worker-spark.sh TARGET FABRIC_IP [PORT] [RDMA_DEV]
#   TARGET = spark0..spark7 (plan target), FABRIC_IP = this Spark's RoCE address.
# Needs: K3_DIR/a3/{worker.py,k3proto.py,libk3hop.so,plan-uniform-62.json}, K3_DIR/situ/venv (setup-worker-venv.sh),
#        BANK_DIR/k3bank-$TARGET.ldb (scripts/build/k3_bank_build.py; 105,977,358,016 bytes, 98.7 GiB)
T=$1; IP=$2; PORT=${3:-19600}; DEV=${4:-rocep1s0f0}
K3_DIR=${K3_DIR:-$HOME/k3pod}; BANK_DIR=${BANK_DIR:-$HOME/k3/banks}; CACHE=${CACHE:-$K3_DIR/fi-cache}
mkdir -p "$CACHE"
docker rm -f k3-worker >/dev/null 2>&1
docker run -d --name k3-worker --gpus all --ipc=host --network host --ulimit memlock=-1:-1 \
  --device /dev/infiniband/rdma_cm --device /dev/infiniband/uverbs0 --device /dev/infiniband/uverbs1 \
  --device /dev/infiniband/uverbs2 --device /dev/infiniband/uverbs3 \
  -v "$K3_DIR:/w" -v "$BANK_DIR:/bank:ro" -v "$CACHE:/root/.cache" \
  -e BANK=/bank/k3bank-$T.ldb -e PLAN=/w/a3/plan-uniform-62.json -e TARGET=$T \
  -e BIND_IP=$IP -e PORT=$PORT -e DEV=$DEV -e PYTHONUNBUFFERED=1 -e K3_WGRAPH=${K3_WGRAPH:-1} \
  nvcr.io/nvidia/pytorch:26.07-py3@sha256:3b0a5c3aa754aff7c86bc0ced07cb064ebce1048f6c12d717e2fa55766276521 \
  /w/situ/venv/bin/python /w/a3/worker.py
