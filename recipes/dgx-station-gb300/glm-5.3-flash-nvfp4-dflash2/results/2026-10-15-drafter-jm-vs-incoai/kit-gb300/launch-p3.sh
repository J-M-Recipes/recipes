#!/usr/bin/env bash
# P3 GB300 lane launcher (orchestrator decision 2026-10-15 02:05: move P3 to GB300 vLLM dflash lane).
# Form copied from /models/drafter-train/p2/scripts/sanity.sh (P0/P2-proven image + target + GPU pin).
# Changes vs sanity.sh, each for a stated reason:
#   --max-model-len 16384   cq holdout = prompt + 1024 new tokens; P1 holdout convs up to ~3K tokens
#   --served-model-name glm-5.3-flash   so the unmodified weekend clients' --model default works
#   --chat-template chat_template_mm.jinja (sha e5762d40, == Tony's Spark template == P1 generation template)
#   --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45
#   --default-chat-template-kwargs enable_thinking=false   all four as in the Spark launch-dflash-rank.sh
#   -v $P3/raw:/p3raw (rw) so in-container clients can write results
# Usage: launch-p3.sh <ARM> <draft_dir> <K>
set -Eeuo pipefail
ARM=$1; DRAFT=$2; K=$3
P3=/models/drafter-train/p3
NAME="p3-${ARM,,}-30021"
IMAGE=vllm-glm53f-p0:ac9126e58-aux-kvextra
MODEL=/models/GLM-5.3-Flash-NVFP4/aa28e1f54130286c95fee10d0705c74ce8743734/original
TEMPLATE=/models/drafter-train/p1/build/chat_template_mm.jinja
mkdir -p "$P3/raw"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --restart no --gpus "device=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55" --shm-size 32g --network host \
  -v /models:/models:ro -v "$P3/raw:/p3raw" -e HF_HUB_OFFLINE=1 "$IMAGE" \
  --model "$MODEL" --served-model-name glm-5.3-flash --host 127.0.0.1 --port 30021 \
  --trust-remote-code --max-model-len 16384 --max-num-seqs 16 --gpu-memory-utilization 0.85 \
  --speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":$K}" \
  --chat-template "$TEMPLATE" \
  --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45 \
  --default-chat-template-kwargs '{"enable_thinking":false}'
echo "P3_LAUNCHED arm=$ARM name=$NAME draft=$DRAFT k=$K"
