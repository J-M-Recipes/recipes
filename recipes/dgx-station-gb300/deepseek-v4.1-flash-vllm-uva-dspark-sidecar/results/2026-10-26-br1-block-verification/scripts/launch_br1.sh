#!/usr/bin/env bash
# launch_br1.sh — Card BR1 variant of launch_sidecar.sh (2026-10-25, Milo desk): identical except SPEC_EXTRA, spliced into the
# --speculative-config JSON before its closing brace. Empty = byte-identical production args; arm B: SPEC_EXTRA=',"rejection_sample_method":"block"'.
# launch_sidecar.sh — DeepSeek-V4.1-Flash-1017 on ONE DGX Station GB300: hot experts in GB300 HBM (MegaMoE, el8 hook),
# cold experts COMPUTED on the RTX PRO 6000 Max-Q via original-el8's /dev/shm "peer tier 2" sidecar (b12x kernels).
# Derived from ~/dsv41/m3/launch_m3g.sh (our Grace-only M3G lane, Card M3G 2026-10-17) and RonanLabsAI
# dgx-station-frontier recipes/dsv41-flash-sidecar/launch/launch-dsv41-sidecar.sh (Apache-2.0). Card SC0, 2026-10-21 (Milo).
# Usage:  launch_sidecar.sh sidecar     # starts the RTX side, waits for "serving"
#         launch_sidecar.sh server      # starts the GB300 vLLM server (requires sidecar serving)
# Bring-up gate: MEGA_PEER_CHECK=100000 MEGA_COLD_TRT=1 launch_sidecar.sh server   (every layer cos >= 0.999 on real prompts)
# PEER_SPINS (default 2e9 since 2026-10-22; el8 default 2e7): GB300 spin limit while waiting on the sidecar. On expiry the
# hook SKIPS the cold add (wrong tokens), so production keeps it effectively unbounded: a dead sidecar hangs, the watchdog rolls back.
# RTX is reached through CDI (nvidia.com/gpu=<UUID>); legacy --gpus cannot see it on DGX OS (dgx-desktop-docker-gpus policy).
set -euo pipefail
STEP=${1:?usage: launch_sidecar.sh sidecar|server}
NAME=${NAME:-dsv41-vllm-SC0-sidecar-EXP}
SCNAME=${SCNAME:-${NAME}-rtx}
IMAGE=${IMAGE:-vllm/vllm-openai@sha256:3af6a45b060d681aa6254200f87859d1bb3c1f962afb3d15540b4b31d4a41063}   # = nightly-af7f9488 (backup: /models/ark-stage/images)
MODEL=${MODEL:-/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170}
W=~/dsv41/m3
HOOK=$W/el8/deepseek-v4.1-flash/m3/hook
B12X=~/dsv41/b12x            # local-inference-lab/b12x @2cc7f66a
CACHE=~/dsv41/cache-a
GB300=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55
RTX=GPU-d07eb6e6-7053-42a7-6811-7bb833266182
SEQS=${SEQS:-32}
KSCHED=${KSCHED:-"[[1,4,5],[5,8,3],[9,$SEQS,3]]"}
CG_SIZES=${CG_SIZES:-"1 2 4 6 8 12 16 18 20 24 28 32 36 40 48 56 64 72 80 96 128"}
ROWMAP=${ROWMAP:-rowmap-static-v1.json}   # ours: uniform 295 hot / 89 cold. Positive control: rowmap-mix-v1.json (285/99)
GPU_UTIL=${GPU_UTIL:-0.97}
MAXLEN=${MAXLEN:-1048576}
PORT=${PORT:-30006}
EXTRA=${EXTRA:-}
SPEC_EXTRA=${SPEC_EXTRA:-}
LOGDIR=$W/logs
mkdir -p "$CACHE"/{vllm,tilelang,triton,nv,dj,fi,b12x} "$W/prof" "$LOGDIR" "$W/local"
test -f "$MODEL/config.json" || { echo "MODEL MISSING: $MODEL"; exit 3; }
test -f "$HOOK/$ROWMAP" || { echo "ROWMAP MISSING: $HOOK/$ROWMAP"; exit 3; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "IMAGE MISSING: $IMAGE"; exit 3; }

if [ "$STEP" = sidecar ]; then
  USED=$(nvidia-smi --id="$RTX" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [ "${USED:-99999}" -lt 2048 ] || { echo "RTX busy (${USED} MiB): stop whatever runs on it first"; exit 4; }
  test -f "$B12X/b12x/moe/fused_moe/api.py" || { echo "b12x checkout missing at $B12X"; exit 3; }
  test -f "$W/local/peer_server2_local.py" || { echo "peer_server2_local.py missing"; exit 3; }
  rm -f /dev/shm/vllm_peer_tier2 2>/dev/null || true   # stale buffer would be opened without O_CREAT by the GB300 side
  docker rm -f "$SCNAME" 2>/dev/null || true
  docker run -d --name "$SCNAME" --device nvidia.com/gpu="$RTX" --ipc host --network host --cap-add SYS_NICE \
    --ulimit memlock=-1 --cap-add IPC_LOCK --security-opt label=disable \
    -v "$MODEL":/model:ro -v "$HOOK":/w:ro -v "$B12X":/b12x:ro -v "$W/local":/sc:ro -v "$LOGDIR":/logs \
    -v "$CACHE/b12x":/root/.cache -v /usr/bin/numactl:/usr/local/bin/numactl:ro \
    -e PYTHONPATH=/b12x -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e HF_HUB_OFFLINE=1 --entrypoint bash "$IMAGE" -c \
    "python3 -c 'import b12x.moe.fused_moe as fm; print(\"b12x ok\", fm.PackedSourceFormat(\"fp4_e8m0_k32\"))' && \
     exec numactl --membind=0 python3 /sc/peer_server2_local.py --rowmap /w/$ROWMAP --model /model" >/dev/null
  nohup docker logs -f "$SCNAME" > "$LOGDIR/peer_server2.log" 2>&1 &
  echo "sidecar $SCNAME starting on RTX $RTX rowmap=$ROWMAP; waiting for 'serving' (Ronan: 40 layers in 45 s, 560 graphs, ~76 GB at 99 cold)"
  for i in $(seq 1 120); do
    if grep -q "^serving" "$LOGDIR/peer_server2.log" 2>/dev/null; then
      tail -3 "$LOGDIR/peer_server2.log"; ls -la /dev/shm/vllm_peer_tier2
      nvidia-smi --id="$RTX" --query-gpu=memory.used,power.draw --format=csv,noheader; exit 0
    fi
    docker ps -q -f name="^${SCNAME}$" | grep -q . || { echo "sidecar died"; tail -30 "$LOGDIR/peer_server2.log"; exit 1; }
    sleep 5
  done
  echo "sidecar not serving after 10 min"; tail -30 "$LOGDIR/peer_server2.log"; exit 1
fi

[ "$STEP" = server ] || { echo "usage: sidecar|server"; exit 2; }
grep -q "^serving" "$LOGDIR/peer_server2.log" 2>/dev/null || { echo "sidecar not serving; run: $0 sidecar"; exit 4; }
docker ps -q -f name="^${SCNAME}$" | grep -q . || { echo "sidecar container $SCNAME not running"; exit 4; }
USED=$(nvidia-smi --id="$GB300" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
[ "${USED:-99999}" -lt 8192 ] || { echo "GB300 busy (${USED} MiB)"; exit 4; }
# vllm#58132 overlay (five files) — same tree as launch_m3g.sh; REPLAY_OVERLAY=0 drops it
MOUNT_ARGS=()
if [ "${REPLAY_OVERLAY:-1}" = 1 ]; then
  O=$W/overlay-58132/tree/vllm; D=/usr/local/lib/python3.12/dist-packages/vllm
  for f in config/cache.py models/deepseek_v41/decoder_replay_layers.py models/deepseek_v41/nvidia/model.py \
           models/deepseek_v41/nvidia/model_state.py models/deepseek_v41/nvidia/vl_model.py; do
    [ -f "$O/$f" ] || { echo "overlay file missing: $O/$f" >&2; exit 2; }
    MOUNT_ARGS+=(-v "$O/$f:$D/$f:ro")
  done
fi
sudo -n sh -c "sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory" 2>/dev/null || true
docker rm -f "$NAME" 2>/dev/null || true
docker run -d --name "$NAME" --gpus "\"device=$GB300\"" \
  --cap-add SYS_NICE --ipc host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK --security-opt label=disable \
  -v "$MODEL":/model:ro -v "$CACHE/vllm":/root/.cache/vllm -v "$CACHE/fi":/root/.cache/flashinfer \
  -v "$CACHE/tilelang":/root/.tilelang -v "$CACHE/triton":/root/.triton -v "$CACHE/nv":/root/.nv -v "$CACHE/dj":/root/.dj \
  -v "$W/prof":/prof -v "$LOGDIR":/logs \
  -v "$HOOK/sitecustomize.py":/usr/lib/python3.12/sitecustomize.py:ro -v "$HOOK":/w:ro \
  -v /usr/bin/numactl:/usr/local/bin/numactl:ro --entrypoint /usr/local/bin/numactl \
  "${MOUNT_ARGS[@]}" \
  -e VLLM_LOGGING_LEVEL=INFO -e VLLM_ENGINE_READY_TIMEOUT_S=7200 -e CUDA_LOG_FILE=stderr -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e HF_HUB_OFFLINE=1 \
  -e MEGA_HOOK=/w/mega_peer_hook.py -e MEGA_ROWMAP=/w/$ROWMAP \
  -e VLLM_EXP_PEER_SPINS=${PEER_SPINS:-2000000000} \
  -e MEGA_PEER=2 -e MEGA_PEER_MIN_TOKENS=1 -e VLLM_EXP_PEER2_SMALL=64 -e MEGA_FUSED_SEND=2 \
  -e MEGA_PEER_CHECK=${MEGA_PEER_CHECK:-0} -e MEGA_PEER_CHECK_MIN_T=65 -e MEGA_COLD_TRT=${MEGA_COLD_TRT:-0} -e MEGA_COUNT=${MEGA_COUNT:-} \
  -e MEGA_MHC_OVERLAP=1 -e MEGA_LL_GEMM=${MEGA_LL_GEMM:-1} -e MEGA_FI_TUNE_FILE=${MEGA_FI_TUNE_FILE:-} -e MEGA_NO_MEGA_MHC=0 \
  "$IMAGE" --membind=0 vllm serve \
  --model /model --served-model-name dsv41-flash-uva --trust-remote-code --tensor-parallel-size 1 \
  --moe-backend deep_gemm_mega_moe --enable-expert-parallel \
  --engram-config '{"cpu_offload": true}' \
  --max-model-len "$MAXLEN" --max-num-seqs "$SEQS" --max-num-batched-tokens 8192 \
  --gpu-memory-utilization "$GPU_UTIL" --kv-cache-dtype fp8_ds_mla \
  --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 --enable-auto-tool-choice \
  --long-prefill-token-threshold 6144 --enable-prefix-caching \
  --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":5,\"num_speculative_tokens_per_batch_size\":$KSCHED,\"draft_sample_method\":\"probabilistic\"$SPEC_EXTRA}" \
  --cudagraph-capture-sizes $CG_SIZES \
  $EXTRA --host 0.0.0.0 --port "$PORT" >/dev/null
nohup docker logs -f "$NAME" > "$LOGDIR/$NAME.log" 2>&1 &
echo "launched $NAME (peer v2, rowmap=$ROWMAP seqs=$SEQS util=$GPU_UTIL maxlen=$MAXLEN overlay=${REPLAY_OVERLAY:-1} PEER_CHECK=${MEGA_PEER_CHECK:-0} COLD_TRT=${MEGA_COLD_TRT:-0}) log $LOGDIR/$NAME.log"
