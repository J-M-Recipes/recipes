#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run an E1/E2 probe inside the serving image on a chosen GPU of this box.
#
#   ./run_probe.sh e1                    # GB300 (index 1), default args
#   GPU=0 ./run_probe.sh e2             # RTX PRO 6000 (index 0)
#   GPU=1 FORCE=1 ./run_probe.sh e2 --m 6 32 128
#
# The GPU index is a HOST index; inside the container it becomes device 0
# (CUDA_VISIBLE_DEVICES), which is what the Python scripts are told.
set -euo pipefail

PROBE_ARG=${1:-}
case "$PROBE_ARG" in
  e1) SCRIPT=e1_split_counter_probe.py ;;
  e2) SCRIPT=e2_moe_tile_probe.py ;;
  *)  echo "usage: $0 e1|e2 [extra python args]" >&2; exit 2 ;;
esac
shift || true

GPU=${GPU:-1}
NEED_MIB=${NEED_MIB:-2048}
IMAGE=${IMAGE:-3af6a45b060d}

command -v docker >/dev/null || { echo "docker not found" >&2; exit 2; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || {
  echo "image $IMAGE not present on this box (set IMAGE=...)" >&2; exit 2; }

FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$GPU")
UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU" | tr -d ' ')
echo "[run] GPU $GPU ($NAME, $UUID): ${FREE} MiB free; need ${NEED_MIB} MiB"
if [ "$FREE" -lt "$NEED_MIB" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "[run] REFUSING: not enough free memory on GPU $GPU." >&2
  echo "[run] A new CUDA context alone costs ~0.5 GiB; pick another GPU," >&2
  echo "[run] wait for a quiet window, or pass FORCE=1 deliberately." >&2
  exit 3
fi

cd "$(dirname "$0")"
mkdir -p results
STAMP=$(date +%Y%m%d-%H%M%S)
JSON="/probes/results/${PROBE_ARG}-${STAMP}.json"

# CDI, not --gpus: on DGX OS the add-in GPU is only visible through
# nvidia.com/gpu=<UUID> (legacy --gpus cannot see it). One device in the
# container, so it is device 0 for the Python script.
# The flashinfer JIT cache is mounted so the ~15-20 min sm100/sm120 kernel
# build happens once, not every run.
mkdir -p "$HOME/.cache/flashinfer-probe"
docker run --rm --device "nvidia.com/gpu=${UUID}" \
  -e PYTHONUNBUFFERED=1 \
  -v "$PWD":/probes \
  -v "$HOME/.cache/flashinfer-probe":/root/.cache/flashinfer \
  --entrypoint python3 \
  "$IMAGE" "/probes/$SCRIPT" --device 0 --json "$JSON" "$@"

echo "[run] result: $(pwd)/results/${PROBE_ARG}-${STAMP}.json"
