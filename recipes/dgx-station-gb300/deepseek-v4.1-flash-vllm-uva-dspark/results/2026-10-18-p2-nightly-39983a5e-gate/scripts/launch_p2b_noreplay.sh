#!/usr/bin/env bash
# launch_p2b_noreplay.sh — Card P2b ARM C (2026-10-18, claude-opus-5-5 subagent for Milo) = launch_p2_nightly39983a5e.sh + ONE flag:
#   --no-swa-bounded-replay  (CacheConfig.swa_bounded_replay=False; verified CPU-only on the image by argparse: default True, flag -> False).
#   Same image digest, same cache mount vllm-cache-p2-39983a5e, same FI managed store. Everything else byte-identical.
# ---- original header follows ----
# launch_p2_nightly39983a5e.sh — Card P2 CAND launch (armed 2026-10-18 ~10:45 CDT, claude-opus-5-5 subagent for Milo).
# = ~/dsv41/1017/launch_v23.sh translated to vLLM nightly 39983a5e (pinned by DIGEST a3bc7f09…; tag rotates).
# Brief: Mac ~/.hermes/profiles/milo/work/dsv41-optimization/el8-cards-2026-10-18/P2/BRIEF-P2-prep.md
# Deltas vs launch_v23.sh (the image rebase is the axis; everything else equal in meaning):
#   IMAGE  deepseekv41-flash-0909 -> vllm/vllm-openai@sha256:a3bc7f09…2623 (vllm 0.32.0rc1.dev329+g39983a5e7, flashinfer 0.7.2 — preflight 10-18)
#   CACHE  $HOME/dsv41/vllm-cache -> $HOME/dsv41/vllm-cache-p2-39983a5e (fresh)
#   -e VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/root/.cache/vllm/fi-autotune-v2   (#55671 managed store; else lands outside the mount)
#   --tokenizer-mode deepseek_v41                 explicit (auto resolves the same; = Card R)
#   --attention-config FLASHMLA_MEGA_ATTN_DSV41   explicit (= SM100 default on this image; = Card R2b)
#   --kv-cache-dtype fp8_ds_mla                   REQUIRED (MegaAttn otherwise resolves to nvfp4_ds_mla = second axis)
#   --compilation-config '{"cudagraph_capture_sizes":[…]}'  same 15 sizes as v23 (JSON form = Card R; both forms valid)
# KEPT from v23: -e PYTORCH_CUDA_ALLOC_CONF=pinned_max_cached_size_mb:1024, off60 (OFFGB env; 54 only as the host-guard retry),
#   util 0.97, 1M, seqs 24, mnbt 8192, lpt 6144, DSpark k5 + ksched [[1,8,5],[9,24,1]], parsers, Engram cpu_offload,
#   hook dir $HOME/pin-hot-experts/v19/hook (sha b5e4f4f8), PIN_* env, port 30006, served name.
# Native on this image: #58132 decoder SWA bounded replay (default on), #59532 replay CUDA graphs (default on).
set -euo pipefail
NEW=/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170
IMAGE=vllm/vllm-openai@sha256:a3bc7f09c1e2e72e068021feb75118c92e98c538706caa0c5d1acc0bb9252623
CACHE=$HOME/dsv41/vllm-cache-p2-39983a5e
HOOK=$HOME/pin-hot-experts/v19/hook
OFFGB=${OFFGB:-60}
NAME=${1:-dsv41-vllm-1017-P2b-n39983a5e-noreplay-EXP}
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "image $IMAGE not present"; exit 3; }
[[ "$(sha256sum "$HOOK/pin_hot_experts_hook.py" | cut -c1-8)" == "b5e4f4f8" ]] || { echo "hook sha mismatch — refusing"; exit 3; }
mkdir -p "$CACHE"
sudo -n sh -c "sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory" 2>/dev/null || true
docker run -d --name "$NAME" --gpus all --ipc host --network host --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK \
  -v "$NEW":/model:ro -v /mnt/miloark:/mnt/miloark:ro -v "$CACHE":/root/.cache/vllm \
  -v "$HOOK/sitecustomize.py":/usr/lib/python3.12/sitecustomize.py:ro -v "$HOOK":/w:ro -v $HOME/pin-hot-experts/v16:$HOME/pin-hot-experts/v16 \
  -e PIN_MODE=split -e PIN_HOOK=/w/pin_hot_experts_hook.py -e PIN_ROWMAP=/w/rowmap-static-v1.json -e PYTORCH_CUDA_ALLOC_CONF=pinned_max_cached_size_mb:1024 -e VLLM_LOGGING_LEVEL=INFO -e VLLM_ENGINE_READY_TIMEOUT_S=7200 \
  -e VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/root/.cache/vllm/fi-autotune-v2 \
  "$IMAGE" \
  --model /model --served-model-name dsv41-flash-uva --trust-remote-code --tensor-parallel-size 1 --tokenizer-mode deepseek_v41 \
  --offload-backend uva --cpu-offload-gb "$OFFGB" --cpu-offload-params routed_experts.w13_weight routed_experts.w2_weight \
  --engram-config '{"cpu_offload": true}' --attention-config '{"backend":"FLASHMLA_MEGA_ATTN_DSV41"}' --kv-cache-dtype fp8_ds_mla --no-swa-bounded-replay \
  --max-model-len 1048576 --max-num-seqs 24 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.97 \
  --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 --enable-auto-tool-choice --long-prefill-token-threshold 6144 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":[[1,8,5],[9,24,1]]}' \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,6,8,12,16,18,24,32,40,48,64,96,128]}' \
  --host 0.0.0.0 --port 30006 >/dev/null
echo "launched $NAME (image a3bc7f09, off$OFFGB, cache $CACHE, NO swa bounded replay)"
