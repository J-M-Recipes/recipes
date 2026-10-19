#!/usr/bin/env bash
# launch_p2c.sh — Card P2c arms (2026-10-18, claude-opus-5-5 subagent for Milo) = launch_p2b_noreplay.sh (config C) + ONE variable:
#   VARIANT=norecord    -> bind-mount p2c-overrides/attention.py.norecord over the image's models/deepseek_v41/attention.py (:ro)
#                          (_use_v41_mxfp8_kv_record() -> False: pre-#56893 584 B V4 fp8_ds_mla record; Card X56893 technique)
#   VARIANT=draftsparse -> --speculative-config gains "attention_backend":"FLASHMLA_SPARSE_DSV41" (drafter off MegaAttn; target stays MegaAttn)
# Everything else byte-identical to launch_p2b_noreplay.sh (same image digest, cache/store mount, hook dir, off60, flags).
set -euo pipefail
NEW=/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170
IMAGE=vllm/vllm-openai@sha256:a3bc7f09c1e2e72e068021feb75118c92e98c538706caa0c5d1acc0bb9252623
CACHE=$HOME/dsv41/vllm-cache-p2-39983a5e
HOOK=$HOME/pin-hot-experts/v19/hook
OV=$HOME/dsv41/1017/p2c-overrides/attention.py.norecord
OFFGB=${OFFGB:-60}
VARIANT=${VARIANT:?VARIANT=norecord|draftsparse}
NAME=${1:?name}
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "image $IMAGE not present"; exit 3; }
[[ "$(sha256sum "$HOOK/pin_hot_experts_hook.py" | cut -c1-8)" == "b5e4f4f8" ]] || { echo "hook sha mismatch — refusing"; exit 3; }
EXTRA_MOUNT=()
SPEC='{"method":"dspark","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":[[1,8,5],[9,24,1]]}'
case "$VARIANT" in
  norecord)
    [[ "$(sha256sum "$OV" | cut -c1-8)" == "5b8e4188" ]] || { echo "override sha mismatch — refusing"; exit 3; }
    EXTRA_MOUNT=(-v "$OV":/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/attention.py:ro) ;;
  draftsparse)
    SPEC='{"method":"dspark","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":[[1,8,5],[9,24,1]],"attention_backend":"FLASHMLA_SPARSE_DSV41"}' ;;
  *) echo "bad VARIANT"; exit 3 ;;
esac
mkdir -p "$CACHE"
sudo -n sh -c "sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory" 2>/dev/null || true
docker run -d --name "$NAME" --gpus all --ipc host --network host --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK \
  -v "$NEW":/model:ro -v /mnt/miloark:/mnt/miloark:ro -v "$CACHE":/root/.cache/vllm \
  -v "$HOOK/sitecustomize.py":/usr/lib/python3.12/sitecustomize.py:ro -v "$HOOK":/w:ro -v $HOME/pin-hot-experts/v16:$HOME/pin-hot-experts/v16 \
  "${EXTRA_MOUNT[@]}" \
  -e PIN_MODE=split -e PIN_HOOK=/w/pin_hot_experts_hook.py -e PIN_ROWMAP=/w/rowmap-static-v1.json -e PYTORCH_CUDA_ALLOC_CONF=pinned_max_cached_size_mb:1024 -e VLLM_LOGGING_LEVEL=INFO -e VLLM_ENGINE_READY_TIMEOUT_S=7200 \
  -e VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/root/.cache/vllm/fi-autotune-v2 \
  "$IMAGE" \
  --model /model --served-model-name dsv41-flash-uva --trust-remote-code --tensor-parallel-size 1 --tokenizer-mode deepseek_v41 \
  --offload-backend uva --cpu-offload-gb "$OFFGB" --cpu-offload-params routed_experts.w13_weight routed_experts.w2_weight \
  --engram-config '{"cpu_offload": true}' --attention-config '{"backend":"FLASHMLA_MEGA_ATTN_DSV41"}' --kv-cache-dtype fp8_ds_mla --no-swa-bounded-replay \
  --max-model-len 1048576 --max-num-seqs 24 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.97 \
  --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 --enable-auto-tool-choice --long-prefill-token-threshold 6144 \
  --speculative-config "$SPEC" \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,6,8,12,16,18,24,32,40,48,64,96,128]}' \
  --host 0.0.0.0 --port 30006 >/dev/null
echo "launched $NAME (image a3bc7f09, off$OFFGB, NO swa bounded replay, VARIANT=$VARIANT)"
