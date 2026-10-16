#!/bin/bash
# GLM-5.3 743B official FP8 (zai-org/GLM-5.3 @ba32f8c1) · TP=8 · GhettoRak 8x DGX Spark GB10
#
# usage: ./launch-fp8-tp8.sh <node-rank 0-7>      (rank r = Spark r+1, fabric 10.0.0.(r+1))
#   rank 0 = Spark1 (.51) head, serves the API on :8890. Ranks 1-7 = --headless workers.
#   Launch WORKERS FIRST (7..1) then the head (0). Driver: ./cluster.sh up
#
# Source: im0xMagnus/glm-5.3-uncensored-8x-dgx-spark @dd5fff7a launchers/launch-fp8-tp8.sh
#   (itself derived from ciprianveg/gb10-vllm glm-5.3/v19). Deltas marked "MILO:".
set -u

NODE_RANK="${1:-}"
case "$NODE_RANK" in 0|1|2|3|4|5|6|7) ;; *) echo "rank must be 0-7" >&2; exit 2 ;; esac

# registry manifest sha256:7ac0031c… → config/Id sha256:ba79eedf36272341… on overlay2 nodes
IMAGE="ghcr.io/ciprianveg/gb10-glm-5.2:v19-vision"
IMAGE_ID_PREFIX="sha256:ba79eedf36272341"
NAME="glm53_fp8_tp8"
MODEL_HOST=${MODEL_HOST:-$HOME/models/GLM-5.3-FP8-zai-ba32f8c1}   # MILO: local sha256-verified copy on every node
MODEL_PATH=/models/glm-5.3-fp8
CACHE_HOST=/var/tmp/glm53-fp8-vllm-cache
HEAD_IP=10.0.0.1
MPORT=29531
PORT=8890                                                # MILO: not 8888 (spark-ds4 route lives there)
ENVF=${ENVF:-$HOME/glm53-tp8/.env.glm53}                  # head only: GLM53_API_KEY=... (never printed)

HOST_IP="10.0.0.$((NODE_RANK + 1))"
[ "$NODE_RANK" = "0" ] && HEADLESS="" || HEADLESS="--headless"

GIDX=3
for i in 0 1 2 3 4 5 6 7; do
  t=$(cat /sys/class/infiniband/rocep1s0f0/ports/1/gid_attrs/types/$i 2>/dev/null)
  g=$(cat /sys/class/infiniband/rocep1s0f0/ports/1/gids/$i 2>/dev/null)
  case "$t" in *"RoCE v2"*) case "$g" in *ffff*) GIDX=$i; break ;; esac ;; esac
done
echo "rank $NODE_RANK  ip=$HOST_IP  NCCL_IB_GID_INDEX=$GIDX"

ip -br a show enp1s0f0np0 | grep -q "$HOST_IP/" || { echo "fabric IP $HOST_IP not on enp1s0f0np0" >&2; exit 3; }
test -f "$MODEL_HOST/config.json" || { echo "MISSING: $MODEL_HOST/config.json" >&2; exit 3; }
n=$(ls "$MODEL_HOST"/*.safetensors 2>/dev/null | wc -l)
[ "$n" -eq 141 ] || { echo "MISSING: only $n/141 safetensors in $MODEL_HOST" >&2; exit 3; }
# weights must be sha256-verified against the Hub LFS manifest before first launch (see README "Launch")
id=$(docker image inspect "$IMAGE" --format '{{.Id}}' 2>/dev/null)
case "$id" in "$IMAGE_ID_PREFIX"*) ;; *) echo "image id mismatch: $id" >&2; exit 3 ;; esac

APIKEY_ARGS=()
if [ "$NODE_RANK" = "0" ]; then
  # shellcheck disable=SC1090
  . "$ENVF" || { echo "missing $ENVF" >&2; exit 3; }
  [ -n "${GLM53_API_KEY:-}" ] || { echo "GLM53_API_KEY empty" >&2; exit 3; }
  APIKEY_ARGS=(-e "VLLM_API_KEY=$GLM53_API_KEY")
fi

mkdir -p "$CACHE_HOST"
docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run --gpus all -d --name "$NAME" --restart no \
  --entrypoint /opt/venv/bin/vllm \
  --network host --ipc host --shm-size 32g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  --device /dev/infiniband:/dev/infiniband \
  -v "$MODEL_HOST:$MODEL_PATH:ro" \
  -v "$CACHE_HOST:/cache" \
  "${APIKEY_ARGS[@]}" \
  -e VLLM_HOST_IP=$HOST_IP \
  -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=7200 \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e CUTE_DSL_ARCH=sm_121a \
  -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_USE_B12X_SPARSE_INDEXER=1 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
  -e DETERMINISTIC_MOE_ALIGN=1 -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e VLLM_DCP_QUERY_SPLIT=1 -e VLLM_B12X_MLA_CKG_GATHER=1 \
  -e OMP_NUM_THREADS=8 \
  -e VLLM_ADAPTIVE_SPEC_DEPTHS=2,4,5 -e VLLM_MTP_INSTRUMENT=1 -e VLLM_MTP_INSTRUMENT_WINDOW=32 \
  -e VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE=1 -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256 \
  -e GLM52_BIND_HOST_TRITON=0 \
  -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor-cache \
  -e TRITON_CACHE_DIR=/cache/triton-cache -e XDG_CACHE_HOME=/cache \
  -e B12X_CUTE_COMPILE_CACHE_DIR=/cache/b12x/cute_compile \
  -e B12X_LOG_CUTE_COMPILES_AFTER_ENGINE_START=0 \
  `# both PCIe halves of QSFP port 0 (GhettoRak Phase 1/2: 8-node busbw 20.1 GB/s @1G)` \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 -e NCCL_IB_GID_INDEX=$GIDX \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
  `# MILO: no NCCL_IB_ADDR_RANGE — PF1 lives on 10.0.1.0/24 here` \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
  -e TP_SOCKET_IFNAME=enp1s0f0np0 -e MN_IF_NAME=enp1s0f0np0 \
  -e NCCL_CROSS_NIC=1 -e NCCL_NVLS_ENABLE=0 -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_BUFFSIZE=16777216 \
  -e NCCL_MAX_NCHANNELS=4 -e NCCL_MIN_NCHANNELS=4 \
  -e NCCL_DEBUG=WARN -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  "$IMAGE" \
    serve "$MODEL_PATH" \
    --served-model-name glm-5.3-fp8 \
    --host 0.0.0.0 --port "$PORT" \
    --trust-remote-code \
    --tensor-parallel-size 8 --pipeline-parallel-size 1 \
    --decode-context-parallel-size 1 --dcp-kv-cache-interleave-size 1 \
    --attention-backend B12X_MLA_SPARSE \
    --hf-overrides '{"index_topk_pattern":"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"}' \
    `# Magnus measured: model+MTP drafter 90.96 GiB/rank; 0.85 → 5.6–6.3 GiB KV; launch free ~105.8 → ceiling ~0.869` \
    --gpu-memory-utilization ${GMU:-0.86} \
    --max-model-len ${MAXLEN:-131072} \
    --max-num-seqs 4 --max-num-batched-tokens 4096 \
    --enable-chunked-prefill --enable-prefix-caching --async-scheduling \
    --long-prefill-token-threshold 2048 \
    --enable-decode-aware-prefill \
    --decode-prefill-token-budget 1024 \
    --idle-prefill-token-budget 16384 \
    --max-long-prefills-per-step 1 \
    --kv-cache-dtype fp8_ds_mla \
    `# MILO: MTP re-enabled (Magnus README: 22.8 tok/s with MTP vs 15.2–15.6 without). ciprianveg spec config, quantization fp8.` \
    --speculative-config '{"model":"/models/glm-5.3-fp8","draft_attention_backend":"B12X_MLA_SPARSE","method":"mtp","quantization":"fp8","num_speculative_tokens":5,"draft_sample_method":"probabilistic","adaptive_speculative_tokens_window":32,"draft_tensor_parallel_size":1}' \
    --compilation-config '{"cudagraph_mode":"FULL","max_cudagraph_capture_size":30}' \
    --disable-custom-all-reduce \
    `# MILO 2026-10-16 opt1: 8-thread safetensors load (boot-time only; head took 956 s single-threaded)` \
    --model-loader-extra-config "${LOADER_CFG:-{\"enable_multithread_load\":true,\"num_threads\":8}}" \
    --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \
    --generation-config vllm \
    `# template honours only low/high; anything else = max (FP8 at max/high thinks to the budget)` \
    --default-chat-template-kwargs '{"reasoning_effort":"low"}' \
    --distributed-executor-backend mp \
    --nnodes 8 --node-rank "$NODE_RANK" \
    --master-addr "$HEAD_IP" --master-port "$MPORT" \
    --seed 42 \
    $HEADLESS

sleep 3
docker ps --format '{{.Names}} {{.Status}}' | grep -q "$NAME" \
  && echo "launched $NAME rank=$NODE_RANK" \
  || { echo "$NAME exited immediately — docker logs $NAME" >&2; exit 1; }
