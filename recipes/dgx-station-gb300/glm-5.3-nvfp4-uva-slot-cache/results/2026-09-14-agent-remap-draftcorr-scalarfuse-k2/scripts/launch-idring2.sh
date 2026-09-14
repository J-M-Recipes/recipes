#!/bin/bash
# launch-idring.sh — K1 256K DAILY profile, byte-identical vLLM args, plus the ID_RING routed-expert capture hook.
# One axis vs glm53-big-sc13g-mtp-ctx256k-DAILY-20260913: SLOT_CACHE_HOOK -> slot_cache_hook_idring2.py, SLOT_CACHE_ID_RING set.
set -e
RUN="${1:-sc13g-mtp-ctx256k-idring2-20260914}"
C=/home/milo/big-v1-campaign
mkdir -p "$C/runs/$RUN"; date -Is > "$C/runs/$RUN/launched_at.txt"
docker run -d --name "glm53-big-$RUN" --gpus all --shm-size 32g --network host \
  -v /home/exx/models/GLM-5.3-NVFP4-big:/model:ro \
  -v /home/milo/vllm-cache:/root/.cache/vllm \
  -v "$C/sitecustomize.py:/usr/lib/python3.12/sitecustomize.py:ro" \
  -v "$C:/w:ro" \
  -v "$C/capture:/wcap" \
  -e VLLM_LOGGING_LEVEL=INFO \
  -e VLLM_AUTOTUNE_CACHE_KEY=slotcache-S112 \
  -e EXACT_PIN=1 -e SLOT_CACHE=112 -e SLOT_CACHE_PER_LAYER=/w/trace/r1-base/slots-7360-ctx256k.json \
  -e SLOT_CACHE_HOOK=/w/slot_cache_hook_idring2.py -e SLOT_CACHE_ID_RING="${ID_RING:-300000}" \
  -e SLOT_CACHE_CAPTURE=0 -e SLOT_CACHE_ROUTER=ffi -e SLOT_CACHE_UNPACKED=0 -e SLOT_CACHE_LOGIT_RING=0 \
  -e SLOT_CACHE_STATS_SEC=20 -e SLOT_CACHE_BYPASS_TOKENS=16 \
  -e VLLM_API_KEY="$(cat /home/milo/.glm_api_key)" \
  vllm-glm53-uva:v0.28.0-2cf0a691 \
  /model --host 0.0.0.0 --port 30001 \
  --served-model-name glm-5.3-big --trust-remote-code --quantization modelopt --load-format safetensors \
  --offload-backend uva --cpu-offload-gb 420 --cpu-offload-params routed_experts.w13_weight routed_experts.w2_weight \
  --gpu-memory-utilization 0.95 --kv-cache-dtype bfloat16 --kv-cache-memory 25769803776 \
  --max-model-len 262144 --max-num-seqs 1 --max-num-batched-tokens 8192 \
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 \
  --compilation-config '{"mode":3,"backend":"eager"}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
echo "launched glm53-big-$RUN"
