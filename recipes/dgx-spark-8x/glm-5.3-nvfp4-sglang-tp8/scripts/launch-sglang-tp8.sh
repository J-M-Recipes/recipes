#!/usr/bin/env bash
# SGLang GLM-5.3 (RadixArk NVFP4) TP8 across 8x DGX Spark. rank r = Spark r+1, fabric 10.0.0.(r+1). API head :8891.
# usage: [env overrides] ./launch-sglang-tp8.sh <rank>
#   SG_IMAGE   image (default lmsysorg/sglang:v0.5.22-cu130)
#   SG_CTX     --context-length           (16384)
#   SG_MAXRUN  --max-running-requests     (1)
#   SG_MEMFRAC --mem-fraction-static      (0.75)
#   SG_CHUNK   --chunked-prefill-size     (8192)
#   SG_KVDT    --kv-cache-dtype           (fp8_e4m3; "auto" lets the engine pick)
#   SG_SPEC    EAGLE spec: "steps,topk,draft" e.g. 3,1,4 ; empty = no speculative decoding
#   SG_GRAPH   1 = CUDA graphs on with --cuda-graph-max-bs-decode $SG_GBS ; 0 = --disable-cuda-graph
#   SG_GBS     cuda graph max decode bs (4)
#   SG_AUTOTUNE 1 = allow FlashInfer autotune; 0 = --disable-flashinfer-autotune (default 0; 1 hung multi-node on 10-16)
#   SG_EXTRA   any extra server args, verbatim
#   SG_ENV     extra "-e K=V -e K2=V2" docker env, verbatim
set -euo pipefail
NODE_RANK=${1:?rank 0-7}
NAME=glm53_sglang_tp8
IMAGE=${SG_IMAGE:-lmsysorg/sglang:v0.5.22-cu130}
MODEL_HOST=${MODEL_HOST:-$HOME/models/GLM-5.3-NVFP4-radixark-6e389189}
CACHE_HOST=/var/tmp/glm53-sglang-cache
HOST_IP=10.0.0.$((NODE_RANK+1))
GIDX=3
mkdir -p "$CACHE_HOST/root-cache"
docker rm -f "$NAME" >/dev/null 2>&1 || true
n=$(ls "$MODEL_HOST"/*.safetensors 2>/dev/null | wc -l); [ "$n" -eq 47 ] || { echo "MISSING: $n/47 safetensors" >&2; exit 3; }

ARGS=(--model-path /models/glm-5.3-nvfp4 --served-model-name glm-5.3-nvfp4-sglang --trust-remote-code
  --tp 8 --nnodes 8 --node-rank "$NODE_RANK" --dist-init-addr 10.0.0.1:20000
  --quantization modelopt_fp4 --reasoning-parser glm45 --tool-call-parser glm47
  --context-length "${SG_CTX:-16384}" --max-running-requests "${SG_MAXRUN:-1}"
  --mem-fraction-static "${SG_MEMFRAC:-0.75}" --chunked-prefill-size "${SG_CHUNK:-8192}"
  --host 0.0.0.0 --port 8891)
[ "${SG_SEF:-0}" = 1 ] || ARGS+=(--disable-shared-experts-fusion)   # SG_SEF=1 → shared-experts fusion ON (SGLang default); 0 = disabled (lane default so far)
KVDT=${SG_KVDT:-fp8_e4m3}; [ "$KVDT" = auto ] || ARGS+=(--kv-cache-dtype "$KVDT")
ARGS+=(--dsa-prefill-backend "${SG_DSA_PREFILL:-flashinfer_sparse_mla}" --dsa-decode-backend "${SG_DSA_DECODE:-flashinfer_sparse_mla}")
if [ -n "${SG_SPEC:-}" ]; then IFS=, read -r S T D <<<"$SG_SPEC"; ARGS+=(--speculative-algorithm EAGLE --speculative-num-steps "$S" --speculative-eagle-topk "$T" --speculative-num-draft-tokens "$D"); fi
if [ "${SG_GRAPH:-1}" = 1 ]; then ARGS+=(--cuda-graph-max-bs-decode "${SG_GBS:-4}"); else ARGS+=(--disable-cuda-graph); fi
[ "${SG_AUTOTUNE:-0}" = 1 ] || ARGS+=(--disable-flashinfer-autotune)
# shellcheck disable=SC2206
[ -z "${SG_EXTRA:-}" ] || ARGS+=(${SG_EXTRA})
KEY_ARGS=()
if [ "$NODE_RANK" = 0 ]; then . "$HOME/glm53-tp8/.env.glm53"; KEY_ARGS=(--api-key "$GLM53_API_KEY"); fi

# shellcheck disable=SC2086
docker run -d --name "$NAME" --network host --ipc host --gpus all --privileged \
  --ulimit memlock=-1 --ulimit stack=67108864 --shm-size 32g \
  -v "$MODEL_HOST":/models/glm-5.3-nvfp4:ro -v "$CACHE_HOST":/cache -v "$CACHE_HOST/root-cache":/root/.cache \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e TRITON_CACHE_DIR=/cache/triton -e FLASHINFER_WORKSPACE_BASE=/cache/flashinfer -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 -e NCCL_IB_GID_INDEX=$GIDX \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 -e TP_SOCKET_IFNAME=enp1s0f0np0 \
  -e NCCL_CROSS_NIC=1 -e NCCL_NVLS_ENABLE=0 -e NCCL_CUMEM_ENABLE=0 \
  -e SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ${SG_ENV:-} \
  --entrypoint python3 "$IMAGE" -m sglang.launch_server "${ARGS[@]}" "${KEY_ARGS[@]}" >/dev/null
echo "launched $NAME rank=$NODE_RANK ip=$HOST_IP ctx=${SG_CTX:-16384} run=${SG_MAXRUN:-1} mem=${SG_MEMFRAC:-0.75} kv=$KVDT spec=${SG_SPEC:-off} graph=${SG_GRAPH:-1} autotune=${SG_AUTOTUNE:-0} extra='${SG_EXTRA:-}'"
