#!/usr/bin/env bash
# launch_m3g.sh — el8 "M3" hook in GRACE-ONLY mode (MEGA_PEER=0, MEGA_COLD_TRT=1) on James's Station.
# Derived from original-el8/dgx-station-gb300-research deepseek-v4.1-flash/m3/launch-m3.sh @87af662 (Apache-2.0),
# with our paths/UUID, the 1017 weights, and the vllm#58132 replay overlay mounted (REPLAY_OVERLAY=0 to drop it).
# Card M3G, 2026-10-17 (Milo, hermes milo). James: "the station is yours" ~22:58 CDT.
set -euo pipefail
NAME=${NAME:-dsv41-vllm-M3G-EXP}
IMAGE=${IMAGE:-vllm/vllm-openai@sha256:3af6a45b060d681aa6254200f87859d1bb3c1f962afb3d15540b4b31d4a41063}   # = nightly-af7f9488 (tag rotated out of Hub)
MODEL=${MODEL:-/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170}
W=$HOME/dsv41/m3
HOOK=$W/el8/deepseek-v4.1-flash/m3/hook
CACHE=$HOME/dsv41/cache-a
GB300=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55
SEQS=${SEQS:-32}
KSCHED=${KSCHED:-"[[1,4,5],[5,8,3],[9,$SEQS,3]]"}
CG_SIZES=${CG_SIZES:-"1 2 4 6 8 12 16 18 20 24 28 32 36 40 48 56 64 72 80 96 128"}
ROWMAP=${ROWMAP:-rowmap-mix-v1.json}
GPU_UTIL=${GPU_UTIL:-0.97}
MAXLEN=${MAXLEN:-1048576}
EXTRA=${EXTRA:-}
# vllm#58132 overlay (five files) — tree built per el8 overlay-58132/README.md
MOUNT_ARGS=()
if [ "${REPLAY_OVERLAY:-1}" = 1 ]; then
  O=$W/overlay-58132/tree/vllm; D=/usr/local/lib/python3.12/dist-packages/vllm
  for f in config/cache.py models/deepseek_v41/decoder_replay_layers.py models/deepseek_v41/nvidia/model.py \
           models/deepseek_v41/nvidia/model_state.py models/deepseek_v41/nvidia/vl_model.py; do
    [ -f "$O/$f" ] || { echo "overlay file missing: $O/$f" >&2; exit 2; }
    MOUNT_ARGS+=(-v "$O/$f:$D/$f:ro")
  done
fi
mkdir -p "$CACHE"/{vllm,tilelang,triton,nv,dj,fi} $W/prof
sudo -n sh -c "sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory" 2>/dev/null || true
docker run -d --name "$NAME" --gpus "\"device=$GB300\"" \
  --cap-add SYS_NICE --ipc host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK --security-opt label=disable \
  -v "$MODEL":/model:ro -v "$CACHE/vllm":/root/.cache/vllm -v "$CACHE/fi":/root/.cache/flashinfer \
  -v "$CACHE/tilelang":/root/.tilelang -v "$CACHE/triton":/root/.triton -v "$CACHE/nv":/root/.nv -v "$CACHE/dj":/root/.dj \
  -v $W/prof:/prof \
  -v "$HOOK/sitecustomize.py":/usr/lib/python3.12/sitecustomize.py:ro -v "$HOOK":/w:ro \
  -v /usr/bin/numactl:/usr/local/bin/numactl:ro --entrypoint /usr/local/bin/numactl \
  "${MOUNT_ARGS[@]}" \
  -e VLLM_LOGGING_LEVEL=INFO -e VLLM_ENGINE_READY_TIMEOUT_S=7200 -e CUDA_LOG_FILE=stderr -e HF_HUB_OFFLINE=1 \
  -e MEGA_HOOK=/w/mega_peer_hook.py -e MEGA_ROWMAP=/w/$ROWMAP \
  -e MEGA_PEER=0 -e MEGA_COLD_TRT=1 -e MEGA_FUSED_SEND=0 -e MEGA_PEER_CHECK=0 -e MEGA_COUNT=${MEGA_COUNT:-} \
  -e MEGA_MHC_OVERLAP=1 -e MEGA_LL_GEMM=${MEGA_LL_GEMM:-1} -e MEGA_FI_TUNE_FILE=${MEGA_FI_TUNE_FILE:-} -e MEGA_NO_MEGA_MHC=0 \
  "$IMAGE" --membind=0 vllm serve \
  --model /model --served-model-name dsv41-flash-uva --trust-remote-code --tensor-parallel-size 1 \
  --moe-backend deep_gemm_mega_moe --enable-expert-parallel \
  --engram-config '{"cpu_offload": true}' \
  --max-model-len "$MAXLEN" --max-num-seqs "$SEQS" --max-num-batched-tokens 8192 \
  --gpu-memory-utilization "$GPU_UTIL" --kv-cache-dtype fp8_ds_mla \
  --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 --enable-auto-tool-choice \
  --long-prefill-token-threshold 6144 --enable-prefix-caching \
  --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":5,\"num_speculative_tokens_per_batch_size\":$KSCHED,\"draft_sample_method\":\"probabilistic\"}" \
  --cudagraph-capture-sizes $CG_SIZES \
  $EXTRA --host 0.0.0.0 --port 30006 >/dev/null
echo "launched $NAME rowmap=$ROWMAP seqs=$SEQS util=$GPU_UTIL overlay=${REPLAY_OVERLAY:-1}"
