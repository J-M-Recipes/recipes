#!/bin/bash
# Kimi-K3 NVFP4 four-tier pod — engine on the DGX Station GB300 (vLLM 0.31.0 + the k3pod hook).
# Start the nine expert workers first (start-workers.sh); each accepts ONE engine connection per start.
#
#   GB300_UUID=GPU-...  MODEL_DIR=/models/kimi-k3/nvidia-nvfp4-b2428a0b  bash launch-engine.sh
#
# Env (defaults in brackets):
#   MODEL_DIR       NVIDIA Kimi-K3-NVFP4 checkpoint @ b2428a0b, on local NVMe        (required)
#   VIEW_ROOT       where truncated-layer views are written                          [$(dirname $MODEL_DIR)]
#   K3_ENGINE_DIR   dir holding scripts/engine/{k3pod.py,sitecustomize.py}           [$HOME/k3-engine]
#   K3_PROTO_DIR    dir holding libk3hop.so, k3proto.py, plan-4tier-v1.json          [$HOME/k3pod/a3]
#   K3_CACHE_DIR    persistent /root/.cache (vLLM + FlashInfer autotune)             [$HOME/k3-cache]
#   K3_OUT_DIR      logs / shadow / record output                                    [$HOME/k3-out]
#   GB300_UUID      the GB300's GPU UUID (on DGX Station, device=1 may not be it)    (required)
#   N               decoder layers to load (93 = full model; <93 = truncated view)   [93]
#   PORT            OpenAI API port (bound to 127.0.0.1)                             [30023]
#   K3_OFFLOAD_GB   pinned-Grace budget for the Station's packed expert weights      [401 when N=93]
#   K3_GPU_UTIL     --gpu-memory-utilization                                         [0.92 when N=93]
#   K3POD_PROXY     1 = graph-resident expert step (C proxy + in-graph wait)         [1]
#   K3POD_SHADOW    1 = shadow parity mode (needs every expert of N layers in HBM)    [0]
#   K3POD_DUMP_DIR  record mode for scripts/check/pod_ref_check.py (needs EAGER=1)   []
#   EAGER           1 = --enforce-eager (no CUDA graphs)                             [0]
#   K3_MAX_SEQS     concurrent requests (--max-num-seqs); multi-token decode goes through the proxy up to 8  [8]
#   K3_MAX_BATCHED  prefill chunk (--max-num-batched-tokens); <= 1024 (k3proto.BATCH_MAX_M, 8 MiB slots)  [1024]
set -euo pipefail
: "${MODEL_DIR:?set MODEL_DIR}" "${GB300_UUID:?set GB300_UUID}"
N=${N:-93}; PORT=${PORT:-30023}
VIEW_ROOT=${VIEW_ROOT:-$(dirname "$MODEL_DIR")}
K3_ENGINE_DIR=${K3_ENGINE_DIR:-$HOME/k3-engine}; K3_PROTO_DIR=${K3_PROTO_DIR:-$HOME/k3pod/a3}
K3_CACHE_DIR=${K3_CACHE_DIR:-$HOME/k3-cache}; K3_OUT_DIR=${K3_OUT_DIR:-$HOME/k3-out}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$K3_OUT_DIR" "$K3_CACHE_DIR"
NAME=k3-pod-n$N
VIEW=$VIEW_ROOT/k3-view-n$N
# symlink view + pruned index + num_hidden_layers=N (no weights copied); the hook drops layers >= N at load
python3 "$HERE/build/k3_truncated_view.py" "$N" "$VIEW" | tee "$K3_OUT_DIR/$NAME.view.txt"
OFF=()
if [ "$N" = 93 ]; then
  OFF=(--cpu-offload-gb "${K3_OFFLOAD_GB:-401}" --cpu-offload-params w13_weight w2_weight)
  UTIL=${K3_GPU_UTIL:-0.92}
else
  [ -n "${K3_OFFLOAD_GB:-}" ] && OFF=(--cpu-offload-gb "$K3_OFFLOAD_GB" --cpu-offload-params w13_weight w2_weight)
  UTIL=${K3_GPU_UTIL:-0.85}
fi
GRAPHS=(-cc '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,2,4,8]}')
[ "${EAGER:-0}" = 1 ] && GRAPHS=(--enforce-eager)
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus "\"device=$GB300_UUID\"" --ipc=host --network=host --shm-size=32g \
  --device /dev/infiniband/uverbs0 --device /dev/infiniband/uverbs1 --ulimit memlock=-1:-1 \
  -v "$VIEW_ROOT:$VIEW_ROOT:ro" -v "$MODEL_DIR:$MODEL_DIR:ro" \
  -v "$K3_CACHE_DIR:/root/.cache" -v "$K3_ENGINE_DIR:/w:ro" -v "$K3_PROTO_DIR:/k3a3:ro" -v "$K3_OUT_DIR:/pod" \
  -e PYTHONPATH=/w -e PYTHONUNBUFFERED=1 -e K3_MAX_LAYER="$N" \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 -e VLLM_DISABLE_SHARED_EXPERTS_STREAM=1 \
  -e K3POD_PLAN=/k3a3/plan-4tier-v1.json -e K3POD_PROTO_DIR=/k3a3 \
  -e K3POD_PROXY="${K3POD_PROXY:-1}" -e K3POD_SHADOW="${K3POD_SHADOW:-0}" -e K3POD_SHADOW_OUT="/pod/shadow-n$N.json" \
  -e K3POD_DUMP_DIR="${K3POD_DUMP_DIR:-}" \
  -e K3POD_MAX_REMOTE_M=$(( ${K3_MAX_BATCHED:-1024} > 512 ? ${K3_MAX_BATCHED:-1024} : 512 )) \
  --entrypoint vllm vllm/vllm-openai:v0.31.0@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3 \
  serve "$VIEW" --served-model-name k3 --trust-remote-code --language-model-only \
  --max-model-len 4096 --max-num-seqs "${K3_MAX_SEQS:-8}" --max-num-batched-tokens "${K3_MAX_BATCHED:-1024}" --no-enable-prefix-caching \
  --gpu-memory-utilization "$UTIL" "${GRAPHS[@]}" "${OFF[@]}" \
  --host 127.0.0.1 --port "$PORT" > "$K3_OUT_DIR/$NAME.cid"
echo "started $NAME on 127.0.0.1:$PORT (load ~7 min + FlashInfer autotune ~12 min at N=93)"
