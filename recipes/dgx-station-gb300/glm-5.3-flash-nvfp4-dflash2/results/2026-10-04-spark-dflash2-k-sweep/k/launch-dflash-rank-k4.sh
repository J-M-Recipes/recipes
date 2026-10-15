#!/usr/bin/env bash
set -Eeuo pipefail

rank="${1:?usage: launch-dflash-rank.sh <0|1>}"
case "$rank" in
  0) host_ip=<rdma-peer-1>; headless=() ;;
  1) host_ip=<rdma-peer-2>; headless=(--headless) ;;
  *) echo "rank must be 0 or 1" >&2; exit 2 ;;
esac

image="funland/glm53-dflash2:ccc5a78e"
expected_image="sha256:5be2a57fe832d1701cc5a3d6c90edfc3509bc6f931344690180ee90ca188b07c"
target_host="~/models/GLM-5.3-Flash-NVFP4"
draft_host="~/models/GLM-5.3-Flash-DFlash2-7d74cdd"
template_host="~/glm53-tony-dflash2-ccc5a78e/chat_template_mm.jinja"
cache_host="~/.cache/glm53-k-vllm"
name="glm53-k-rank${rank}"

actual_image=$(docker image inspect "$image" --format '{{.Id}}')
[[ "$actual_image" == "$expected_image" ]]
[[ -f "$target_host/config.json" ]]
[[ -f "$draft_host/config.json" ]]
[[ -f "$draft_host/model.safetensors" ]]
[[ -f "$template_host" ]]
printf '%s  %s\n' 8931dc522be0aa31760a7463f8d2f8044fa3e6d40be2e87aa08e9fd17bfd6683 "$draft_host/model.safetensors" | sha256sum -c -
printf '%s  %s\n' c4aeac0101196a6e26705b34c45230bcd0c7c68ee2d2d1efdb242087f3712573 "$draft_host/config.json" | sha256sum -c -
mkdir -p "$cache_host"
docker rm -f "$name" >/dev/null 2>&1 || true

docker run --gpus all -d \
  --name "$name" --restart no \
  --network host --ipc host --shm-size 64g \
  --ulimit memlock=-1:-1 --ulimit stack=67108864 \
  --cap-add IPC_LOCK \
  --device /dev/infiniband:/dev/infiniband \
  -v "$target_host:/models/glm53-target:ro" \
  -v "$draft_host:/models/dflash2-draft:ro" \
  -v "$template_host:/opt/glm53/chat-template-mm.jinja:ro" \
  -v "$cache_host:/cache" \
  -e VLLM_HOST_IP="$host_ip" \
  -e HF_HOME=/cache/huggingface \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA=rocep1s0f0,rocep1s0f1 -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
  -e NCCL_IB_ADDR_RANGE=<rdma-peer-0>/24 \
  -e NCCL_SOCKET_IFNAME=bond0 -e GLOO_SOCKET_IFNAME=bond0 \
  -e TP_SOCKET_IFNAME=bond0 -e MN_IF_NAME=bond0 \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=1 \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e NCCL_MAX_NCHANNELS=4 -e NCCL_MIN_NCHANNELS=4 \
  -e NCCL_DEBUG=WARN -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  "$image" \
  /models/glm53-target \
  --served-model-name glm-5.3-flash \
  --host 0.0.0.0 --port 8888 \
  --trust-remote-code --load-format safetensors \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 262144 \
  --max-num-seqs 6 \
  --block-size 2304 \
  --moe-backend marlin \
  --speculative-config '{"method":"dflash","model":"/models/dflash2-draft","num_speculative_tokens":4}' \
  --kv-cache-dtype fp8_e4m3 \
  --kv-cache-memory 3221225472 \
  --enforce-eager \
  --chat-template /opt/glm53/chat-template-mm.jinja \
  --tool-call-parser glm47 --enable-auto-tool-choice \
  --reasoning-parser glm45 \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --distributed-executor-backend mp \
  --nnodes 2 --node-rank "$rank" \
  --master-addr <rdma-peer-1> --master-port 29521 \
  "${headless[@]}"

sleep 3
[[ $(docker inspect "$name" --format '{{.State.Running}}') == true ]]
echo "DFLASH_RANK_LAUNCHED rank=$rank image=$actual_image"
