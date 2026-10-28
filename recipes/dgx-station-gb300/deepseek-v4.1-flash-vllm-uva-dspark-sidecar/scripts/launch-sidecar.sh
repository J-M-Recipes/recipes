#!/usr/bin/env bash
# launch-sidecar.sh — DeepSeek-V4.1-Flash-1017 on ONE DGX Station GB300 with an RTX PRO 6000 Max-Q sidecar.
#
# The GB300 keeps the hot experts in HBM (DeepGEMM MegaMoE, original-el8's hook) and the RTX card
# *computes* each layer's cold-expert contribution; only activations cross PCIe, through one /dev/shm
# buffer with a device-side sequence counter (original-el8's "peer tier 2" protocol, b12x kernels).
#
#   launch-sidecar.sh sidecar   # start the RTX side, wait for "serving"
#   launch-sidecar.sh server    # start the GB300 vLLM server (requires the sidecar serving)
#   launch-sidecar.sh stop      # stop the server then the sidecar (reverse order)
#
# Bring-up gate (do this before serving anything — a timed-out sidecar silently produces wrong tokens):
#   MEGA_PEER_CHECK=100000 MEGA_COLD_TRT=1 launch-sidecar.sh server
#   python3 scripts/gate_cosine.py <server log>          # bar: 40/40 layers, >=32 cold rows, min cos >= 0.999
#
# Derived from RonanLabsAI/dgx-station-frontier recipes/dsv41-flash-sidecar (Apache-2.0) and
# original-el8/dgx-station-gb300-research @87af662 (Apache-2.0). Both are credited in the README.
#
# Required env (no defaults — they are machine-specific):
#   GB300_UUID, RTX_UUID   nvidia-smi -L shows both; legacy `--gpus` cannot see the VGA-class RTX on DGX OS,
#                          so the RTX is reached through a CDI spec (nvidia-ctk cdi generate) and pinned by UUID.
# Optional env:
#   M3_DIR (default $HOME/dsv41/m3), B12X_DIR ($HOME/dsv41/b12x), MODEL_DIR, IMAGE (digest), CACHE_DIR,
#   NAME (default dsv41-vllm-sidecar), SEQS (32), KSCHED, CG_SIZES, ROWMAP (rowmap-static-v1.json),
#   GPU_UTIL (0.97), MAXLEN (1048576), PORT (30006), PEER_SPINS (2000000000), REPLAY_OVERLAY (1),
#   MEGA_PEER_CHECK (0), MEGA_COLD_TRT (0), MEGA_FI_TUNE_FILE (empty), EXTRA (extra serve flags),
#   SPEC_EXTRA (empty; spliced into the --speculative-config JSON before its closing brace — the v27 serving
#   release sets SPEC_EXTRA=',"rejection_sample_method":"block"', as Card BR1's launch_br1.sh did)
# h1/h2 additions (2026-10-28; every block is OFF by default, so v27's docker argv is unchanged — scripts/start-h2ref.sh
# sets them for the h2-REF serving release). Each mirrors an added-only block of the box launcher that served the
# numbers (results/2026-10-28-h2-splitkv-ap2/launch/launch_att2d.sh.as-run):
#   HOOK_DIR             hook directory (default: the el8 hook). h1/h2 = a copy of it with patches/hook-h1-l2-key-sharing.patch
#   MEGA_SHARE_TOPK      set (h2: 2) to pass the L2 key-sharing env: MEGA_SHARE_TOPK, MEGA_POS_I32_ONCE (h2: 1),
#                        MEGA_CAND_COOP (0), MEGA_L2_VERIFY (h2: 1), MEGA_L2_STATS_EVERY (h2: 2000). Unset = not passed.
#   OVERLAY_61161=1      bind-mount the vllm#61161 backport (3 files) from O61 (default $W/overlay-61161/tree/vllm); the
#                        rejection_sampler_utils.py there must also carry the vllm#60945 guard (sha256 2d43d513…) —
#                        see README "Software"; the launcher refuses any other hash unless O61_ALLOW_ANY=1
#   SPLITKV=4            split-KV decode attention: mounts the SIDE extension _flashmla_splitkv_C.abi3.so (next to the
#                        image's _flashmla_C, never over it) + the 2-file vllm#57711-derived overlay from SPLITKV_DIR
#                        (default $W/splitkv: build/out/_flashmla_splitkv_C.abi3.so, overlay/vllm/...), with the quarantine
#                        env SPLITKV_MAX_TOKENS (8) SPLITKV_MAX_REQS (h2: 1) SPLITKV_SQ (h2: 6) SPLITKV_TRACE (h2: 1)
#                        SPLITKV_GUARD (h2: 0) — the last three are passed only when set, as on the box
#   NOCORE=1             --ulimit core=0 + CUDA GPU/CPU coredump-on-exception = 0 (a 602 GB core once filled the disk)
set -euo pipefail
STEP=${1:?usage: launch-sidecar.sh sidecar|server|stop}
NAME=${NAME:-dsv41-vllm-sidecar}
SCNAME=${SCNAME:-${NAME}-rtx}
IMAGE=${IMAGE:-vllm/vllm-openai@sha256:3af6a45b060d681aa6254200f87859d1bb3c1f962afb3d15540b4b31d4a41063}  # nightly-af7f9488 (tag rotated off Hub — keep a `docker save` copy)
MODEL=${MODEL_DIR:-/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170}
W=${M3_DIR:-$HOME/dsv41/m3}
HOOK=${HOOK_DIR:-$W/el8/deepseek-v4.1-flash/m3/hook}   # original-el8 @87af662 (see README "Software"); h1/h2: hook-h1 copy
B12X=${B12X_DIR:-$HOME/dsv41/b12x}           # local-inference-lab/b12x @2cc7f66a
CACHE=${CACHE_DIR:-$HOME/dsv41/cache-a}      # persistent JIT caches: FlashInfer, DeepGEMM, tilelang, triton, b12x
GB300=${GB300_UUID:?set GB300_UUID (nvidia-smi -L)}
RTX=${RTX_UUID:?set RTX_UUID (nvidia-smi -L) — the VGA-class card needs CDI, not --gpus}
SEQS=${SEQS:-32}
KSCHED=${KSCHED:-"[[1,4,5],[5,8,3],[9,$SEQS,3]]"}
CG_SIZES=${CG_SIZES:-"1 2 4 6 8 12 16 18 20 24 28 32 36 40 48 56 64 72 80 96 128"}
ROWMAP=${ROWMAP:-rowmap-static-v1.json}      # ours: uniform 295 hot / 89 cold. Positive control: rowmap-mix-v1.json (285/99)
GPU_UTIL=${GPU_UTIL:-0.97}
MAXLEN=${MAXLEN:-1048576}
PORT=${PORT:-30006}
EXTRA=${EXTRA:-}
SPEC_EXTRA=${SPEC_EXTRA:-}
LOGDIR=$W/logs
mkdir -p "$CACHE"/{vllm,tilelang,triton,nv,dj,fi,b12x} "$W/prof" "$LOGDIR" "$W/local"

if [ "$STEP" = stop ]; then
  docker stop "$NAME" 2>/dev/null || true
  docker stop "$SCNAME" 2>/dev/null || true
  echo "stopped $NAME and $SCNAME (containers kept — rollback is 'docker start', not a relaunch)"
  exit 0
fi

test -f "$MODEL/config.json" || { echo "MODEL MISSING: $MODEL"; exit 3; }
test -f "$HOOK/$ROWMAP" || { echo "ROWMAP MISSING: $HOOK/$ROWMAP"; exit 3; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "IMAGE MISSING: $IMAGE — pull by digest or docker load"; exit 3; }

if [ "$STEP" = sidecar ]; then
  USED=$(nvidia-smi --id="$RTX" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [ "${USED:-99999}" -lt 2048 ] || { echo "RTX busy (${USED} MiB): stop whatever runs on it first (it also drives the display)"; exit 4; }
  test -f "$B12X/b12x/moe/fused_moe/api.py" || { echo "b12x checkout missing at $B12X"; exit 3; }
  test -f "$W/local/peer_server2_local.py" || { echo "peer_server2_local.py missing — apply patches/peer-server2-local.patch (README)"; exit 3; }
  rm -f /dev/shm/vllm_peer_tier2 2>/dev/null || true   # the GB300 side opens it without O_CREAT
  docker rm -f "$SCNAME" 2>/dev/null || true
  docker run -d --name "$SCNAME" --device nvidia.com/gpu="$RTX" --ipc host --network host --cap-add SYS_NICE \
    --ulimit memlock=-1 --cap-add IPC_LOCK --security-opt label=disable \
    -v "$MODEL":/model:ro -v "$HOOK":/w:ro -v "$B12X":/b12x:ro -v "$W/local":/sc:ro -v "$LOGDIR":/logs \
    -v "$CACHE/b12x":/root/.cache -v /usr/bin/numactl:/usr/local/bin/numactl:ro \
    -e PYTHONPATH=/b12x -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e HF_HUB_OFFLINE=1 --entrypoint bash "$IMAGE" -c \
    "python3 -c 'import b12x.moe.fused_moe as fm; print(\"b12x ok\", fm.PackedSourceFormat(\"fp4_e8m0_k32\"))' && \
     exec numactl --membind=0 python3 /sc/peer_server2_local.py --rowmap /w/$ROWMAP --model /model" >/dev/null
  nohup docker logs -f "$SCNAME" > "$LOGDIR/peer_server2.log" 2>&1 &
  echo "sidecar $SCNAME starting on RTX $RTX rowmap=$ROWMAP; waiting for 'serving' (40 layers ~45 s, 560 graphs, ~76 GB at 99 cold)"
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

[ "$STEP" = server ] || { echo "usage: sidecar|server|stop"; exit 2; }
grep -q "^serving" "$LOGDIR/peer_server2.log" 2>/dev/null || { echo "sidecar not serving; run: $0 sidecar"; exit 4; }
docker ps -q -f name="^${SCNAME}$" | grep -q . || { echo "sidecar container $SCNAME not running"; exit 4; }
USED=$(nvidia-smi --id="$GB300" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
[ "${USED:-99999}" -lt 8192 ] || { echo "GB300 busy (${USED} MiB)"; exit 4; }

# vllm#58132 decoder bounded replay, 5 files bind-mounted. REPLAY_OVERLAY=0 drops it (stock nightly-af7f9488
# still runs the KV-side replay; the flag --no-swa-bounded-replay is the other half — see README results).
MOUNT_ARGS=()
if [ "${REPLAY_OVERLAY:-1}" = 1 ]; then
  O=$W/overlay-58132/tree/vllm; D=/usr/local/lib/python3.12/dist-packages/vllm
  for f in config/cache.py models/deepseek_v41/decoder_replay_layers.py models/deepseek_v41/nvidia/model.py \
           models/deepseek_v41/nvidia/model_state.py models/deepseek_v41/nvidia/vl_model.py; do
    [ -f "$O/$f" ] || { echo "overlay file missing: $O/$f" >&2; exit 2; }
    MOUNT_ARGS+=(-v "$O/$f:$D/$f:ro")
  done
fi
D=/usr/local/lib/python3.12/dist-packages/vllm
# vllm#61161 backport (block-verification fresh draws) + the vllm#60945 rejection-sampler guard in the same overlay file.
if [ "${OVERLAY_61161:-0}" = 1 ]; then
  O61=${O61:-$W/overlay-61161/tree/vllm}
  for pair in "v1/worker/gpu/sample/gumbel.py ee5072d245ea850ff4efa60ec548c172484019415f5d687b805d2d6af650ade1" \
              "v1/worker/gpu/spec_decode/rejection_sampler_utils.py 2d43d5133c1fb96399e186ed10c5a26b54c12e6a019d9d163cbff1adbcfbcc39" \
              "v1/worker/gpu/spec_decode/dflash2/speculator.py 4aad3ed14897174c299f01e9360bb75d7450dee40c91ef19707cff58d4c577c5"; do
    set -- $pair
    [ -f "$O61/$1" ] || { echo "overlay-61161 file missing: $O61/$1" >&2; exit 2; }
    got=$(sha256sum "$O61/$1" | cut -d' ' -f1)
    [ "$got" = "$2" ] || [ "${O61_ALLOW_ANY:-0}" = 1 ] || { echo "overlay-61161 $1 sha256 $got != $2 (42c5e8a0… = #61161 WITHOUT the #60945 guard: the latent OOB read)" >&2; exit 2; }
    MOUNT_ARGS+=(-v "$O61/$1:$D/$1:ro")
  done
fi
# split-KV MegaAttn decode (vllm-project/FlashMLA#25 built as a SIDE extension + vllm#57711-derived overlay), quarantined.
if [ -n "${SPLITKV:-}" ] && [ "${SPLITKV}" != 0 ]; then
  SKD=${SPLITKV_DIR:-$W/splitkv}
  SIDE=$SKD/build/out/_flashmla_splitkv_C.abi3.so
  [ -f "$SIDE" ] || { echo "split-KV side .so missing: $SIDE (README 'Software': build it)" >&2; exit 2; }
  got=$(sha256sum "$SIDE" | cut -d' ' -f1)
  [ "$got" = 744a9de89060730eecbb7c4c249dedc36f3dbfb3854de17299780b66cb90e983 ] || \
    echo "WARNING: side .so sha256 $got is not the measured build 744a9de8… — run the kernel tests before serving" >&2
  MOUNT_ARGS+=(-v "$SIDE:$D/_flashmla_splitkv_C.abi3.so:ro")
  for pair in "models/deepseek_v41/nvidia/flash_mla_mega_attn.py 9946e6eef6c1f8e342261c7383e4d2ae69a18111c244f5efc7258cfe87d56d14" \
              "models/deepseek_v41/nvidia/mega_splitkv_combine.py 97c67176a1b8f036de6d122a6dc2525a3e981e476a9584a13c571b158cbe0b56"; do
    set -- $pair
    got=$(sha256sum "$SKD/overlay/vllm/$1" 2>/dev/null | cut -d' ' -f1)
    [ "$got" = "$2" ] || { echo "split-KV overlay $1 sha256 '${got}' != $2" >&2; exit 2; }
    MOUNT_ARGS+=(-v "$SKD/overlay/vllm/$1:$D/$1:ro")
  done
  MOUNT_ARGS+=(-e VLLM_DSV41_MEGA_SPLITKV="$SPLITKV" -e VLLM_DSV41_MEGA_SPLITKV_MAX_TOKENS="${SPLITKV_MAX_TOKENS:-8}"
               -e VLLM_DSV41_MEGA_SPLITKV_SWA="${SPLITKV_SWA:-0}" -e VLLM_DSV41_MEGA_SPLITKV_STRICT=1
               -e VLLM_DSV41_MEGA_SPLITKV_SO="$D/_flashmla_splitkv_C.abi3.so"
               -e VLLM_DSV41_MEGA_SPLITKV_MAX_REQS="${SPLITKV_MAX_REQS:-2}" -e VLLM_DSV41_MEGA_SPLITKV_PRELOAD=1)
  if [ -n "${SPLITKV_SQ+x}" ]; then MOUNT_ARGS+=(-e VLLM_DSV41_MEGA_SPLITKV_SQ="$SPLITKV_SQ"); fi
  if [ -n "${SPLITKV_TRACE:-}" ]; then MOUNT_ARGS+=(-e VLLM_DSV41_MEGA_SPLITKV_TRACE="$SPLITKV_TRACE"); fi
  if [ -n "${SPLITKV_GUARD:-}" ]; then MOUNT_ARGS+=(-e VLLM_DSV41_MEGA_SPLITKV_GUARD="$SPLITKV_GUARD"); fi
fi
if [ -n "${MEGA_SHARE_TOPK:-}" ]; then   # L2 attention/indexer key-sharing (hook-h1); needs HOOK_DIR = the patched hook
  [ -f "$HOOK/attn_share.py" ] || { echo "MEGA_SHARE_TOPK set but $HOOK/attn_share.py missing (apply patches/hook-h1-l2-key-sharing.patch)" >&2; exit 2; }
  MOUNT_ARGS+=(-e MEGA_SHARE_TOPK="$MEGA_SHARE_TOPK" -e MEGA_POS_I32_ONCE="${MEGA_POS_I32_ONCE:-0}" -e MEGA_CAND_COOP="${MEGA_CAND_COOP:-0}"
               -e MEGA_L2_VERIFY="${MEGA_L2_VERIFY:-0}" -e MEGA_L2_STATS_EVERY="${MEGA_L2_STATS_EVERY:-0}")
fi
if [ "${NOCORE:-0}" = 1 ]; then
  MOUNT_ARGS+=(--ulimit core=0 -e CUDA_ENABLE_COREDUMP_ON_EXCEPTION=0 -e CUDA_ENABLE_CPU_COREDUMP_ON_EXCEPTION=0)
fi
sudo -n sh -c "sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory" 2>/dev/null || true
docker rm -f "$NAME" 2>/dev/null || true
docker run -d --name "$NAME" --gpus "device=$GB300" \
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
echo "launched $NAME (peer v2, rowmap=$ROWMAP seqs=$SEQS util=$GPU_UTIL maxlen=$MAXLEN overlay=${REPLAY_OVERLAY:-1} ov61161=${OVERLAY_61161:-0} hook=$HOOK l2=${MEGA_SHARE_TOPK:-0}/${MEGA_POS_I32_ONCE:-0} splitkv=${SPLITKV:-0} PEER_CHECK=${MEGA_PEER_CHECK:-0} COLD_TRT=${MEGA_COLD_TRT:-0}) log $LOGDIR/$NAME.log"
