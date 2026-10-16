#!/bin/bash
# Usage: SPARK_HOSTS="h1 h2 ..." run_bench.sh "1 2" [tag]  -> allreduce_bench across those Spark numbers (1-based into SPARK_HOSTS, fabric IP 10.0.0.N); first = rank 0.
# Runs from the M4. One container per node. Rank-0 output to stdout; others to ~/ghettorak/bench-rank<r>.log on each node.
set -u
read -r -a H <<<"${SPARK_HOSTS:?}"; NODES=($1); TAG=${2:-run}; WS=${#NODES[@]}; M=${NODES[0]}; PORT=$((29500 + RANDOM % 400))
IMG=nvcr.io/nvidia/pytorch:25.09-py3
for n in "${NODES[@]}"; do scp -q -o BatchMode=yes "$(dirname "$0")/allreduce_bench.py" ${H[$((n-1))]}:ghettorak/ 2>/dev/null || { ssh -o BatchMode=yes ${H[$((n-1))]} 'mkdir -p ~/ghettorak'; scp -q -o BatchMode=yes "$(dirname "$0")/allreduce_bench.py" ${H[$((n-1))]}:ghettorak/; }; done
dockercmd() { # rank node
  echo "docker rm -f nccl-bench >/dev/null 2>&1; docker run --rm --name nccl-bench --gpus all --network host --ipc host \
   --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1:-1 --ulimit stack=67108864 -v \$HOME/ghettorak:/w \
   -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
   -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET \
   -e MASTER_ADDR=10.0.0.$M -e MASTER_PORT=$PORT -e RANK=$1 -e WORLD_SIZE=$WS $IMG python /w/allreduce_bench.py"
}
r=1
for n in "${NODES[@]:1}"; do
  ssh -f -o BatchMode=yes ${H[$((n-1))]} "timeout 600 bash -c '$(dockercmd $r)' > ~/ghettorak/bench-$TAG-rank$r.log 2>&1"
  r=$((r+1))
done
ssh -o BatchMode=yes ${H[$((M-1))]} "timeout 600 bash -c '$(dockercmd 0)' 2>&1"
