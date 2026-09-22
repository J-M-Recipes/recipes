#!/usr/bin/env bash
# launch-hotsplit.sh TAG HOT_GIB CTX  -- MiMo-V2.6-Pro, stock loader, exact pin, per-expert HBM residency (hotsplit)
set -euo pipefail
TAG=$1; HOT=${2:-}; CTX=${3:-262144}; COUNTS=${COUNTS:-/w/dumps/expert_hist_mix.json}
if [ -n "$(docker ps -q --filter label=gpu-lane=1)" ] || nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 1 | grep -q .; then echo "GPU busy; refuse"; exit 4; fi
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
N=mimo26-pro-$TAG
docker run -d --name $N --label gpu-lane=1 --gpus all --ipc host --network host --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK \
 -v ${MODEL:-/models/MiMo-V2.6-Pro-RL}:/model:ro -v ${WORKDIR:-$PWD}/vllm-cache:/root/.cache/vllm -v ${WORKDIR:-$PWD}:/w:ro -v ${WORKDIR:-$PWD}/live:/live \
 -v ${WORKDIR:-$PWD}/patch/mimo_v2_omni.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2_omni.py:ro \
 -v ${WORKDIR:-$PWD}/patch/mimo_v2_hotsplit.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2.py:ro \
 -e VLLM_LOGGING_LEVEL=INFO -e VLLM_USE_DEEP_GEMM=0 -e VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1 \
 -e HOTSPLIT_COUNTS=$COUNTS ${LIVE:+-e HOTSPLIT_LIVE_COUNTS=/live/counts.json -e HOTSPLIT_LIVE_SECS=600} -e HOTSPLIT_WEIGHTS=decode=1,prefill=0 ${HOT:+-e HOTSPLIT_HOT_GIB=$HOT} \
 vllm/vllm-openai:nightly-d05da62e9ccdf8e342b15bf6785d83224cc165af \
 --model /model --served-model-name mimo26-pro --trust-remote-code --tensor-parallel-size 1 \
 --offload-backend uva --cpu-offload-gb 320 --cpu-offload-params routed_experts.w13_weight routed_experts.w2_weight \
 --max-model-len $CTX --max-num-seqs 8 --max-num-batched-tokens 2048 --gpu-memory-utilization 0.96 \
 --compilation-config "{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}" --tool-call-parser mimo --reasoning-parser mimo --enable-auto-tool-choice \
 --generation-config auto --moe-backend marlin --host 0.0.0.0 --port 30007 >/dev/null
nohup docker logs -f $N > $N.log 2>&1 &
echo "launched $N hot=${HOT:-default} ctx=$CTX counts=$COUNTS"
