#!/bin/bash
# run.sh <TAG> "<extra -e flags>"  — 8 ranks, v19 image (LD_PRELOAD nccl 2.30.4 from image ENV), launcher NCCL env
TAG=$1; EXTRA=$2; PORT=$((29600 + RANDOM % 300)); IMG=ghcr.io/ciprianveg/gb10-glm-5.2:v19-vision
cmd(){ echo "docker rm -f nccl-lat >/dev/null 2>&1; docker run --rm --name nccl-lat --gpus all --network host --ipc host --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1:-1 -v \$HOME/glm53-tp8:/w --entrypoint /opt/venv/bin/python \
 -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
 -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 -e NCCL_CROSS_NIC=1 -e NCCL_NVLS_ENABLE=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 \
 -e NCCL_BUFFSIZE=16777216 -e NCCL_MAX_NCHANNELS=4 -e NCCL_MIN_NCHANNELS=4 -e NCCL_DEBUG=WARN $EXTRA \
 -e TAG=$TAG -e MASTER_ADDR=10.0.0.1 -e MASTER_PORT=$PORT -e RANK=$1 -e WORLD_SIZE=8 $IMG /w/nccl_lat.py"; }
read -r -a H <<<"${SPARK_HOSTS:?}"; for r in 7 6 5 4 3 2 1; do ssh -f -o BatchMode=yes ${H[$r]} "timeout 300 bash -c '$(cmd $r)' > ~/glm53-tp8/lat-rank$r.log 2>&1"; done
ssh -o BatchMode=yes ${H[0]} "timeout 300 bash -c '$(cmd 0)' 2>&1" | grep -E "bytes=|Error|error" 
