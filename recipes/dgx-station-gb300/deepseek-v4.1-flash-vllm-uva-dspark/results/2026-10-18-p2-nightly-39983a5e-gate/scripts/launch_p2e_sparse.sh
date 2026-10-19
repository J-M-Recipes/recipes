#!/usr/bin/env bash
# launch_p2e_sparse.sh — Card P2e (2026-10-18, claude-opus-5-5 subagent for Milo) = launch_p2b_noreplay.sh (config C) + ONE variable:
#   TARGET attention backend FLASHMLA_MEGA_ATTN_DSV41 -> FLASHMLA_SPARSE_DSV41  (--attention-config), and the DSpark drafter set to
#   FLASHMLA_SPARSE_DSV41 explicitly in --speculative-config "attention_backend" (it would inherit the target backend anyway:
#   v1/worker/gpu/spec_decode/dspark/utils.py:16-31) -> whole model off MegaAttn (no fused o-proj FP8 cast from the fp32 accumulator).
# CPU-only preflight 10-18 (image a3bc7f09): AttentionConfig(backend="FLASHMLA_SPARSE_DSV41") parses; DeepseekV4FlashMLABackend
#   (get_name FLASHMLA_SPARSE_DSV41) supported_kv_cache_dtypes = ['auto','fp8_ds_mla','fp8'] -> --kv-cache-dtype fp8_ds_mla accepted;
#   _select_dsv4_attn_cls (nvidia/model.py:238-243) maps it to DeepseekV4FlashMLAAttention.
# Everything else byte-identical to launch_p2c.sh / launch_p2b_noreplay.sh (same image digest, cache/store mount, hook dir, off60, flags).
set -euo pipefail
NEW=/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170
IMAGE=vllm/vllm-openai@sha256:a3bc7f09c1e2e72e068021feb75118c92e98c538706caa0c5d1acc0bb9252623
CACHE=$HOME/dsv41/vllm-cache-p2-39983a5e
HOOK=$HOME/pin-hot-experts/v19/hook
OFFGB=${OFFGB:-60}
NAME=${1:-dsv41-vllm-1017-P2e-noreplay-sparse-EXP}
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "image $IMAGE not present"; exit 3; }
[[ "$(sha256sum "$HOOK/pin_hot_experts_hook.py" | cut -c1-8)" == "b5e4f4f8" ]] || { echo "hook sha mismatch — refusing"; exit 3; }
SPEC='{"method":"dspark","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":[[1,8,5],[9,24,1]],"attention_backend":"FLASHMLA_SPARSE_DSV41"}'
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
  --engram-config '{"cpu_offload": true}' --attention-config '{"backend":"FLASHMLA_SPARSE_DSV41"}' --kv-cache-dtype fp8_ds_mla --no-swa-bounded-replay \
  --max-model-len 1048576 --max-num-seqs 24 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.97 \
  --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 --enable-auto-tool-choice --long-prefill-token-threshold 6144 \
  --speculative-config "$SPEC" \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,6,8,12,16,18,24,32,40,48,64,96,128]}' \
  --host 0.0.0.0 --port 30006 >/dev/null
echo "launched $NAME (image a3bc7f09, off$OFFGB, NO swa bounded replay, target+drafter FLASHMLA_SPARSE_DSV41, fp8_ds_mla)"
