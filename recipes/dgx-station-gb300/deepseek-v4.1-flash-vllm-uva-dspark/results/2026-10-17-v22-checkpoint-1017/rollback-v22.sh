#!/usr/bin/env bash
# Roll back from the v22 (1017 checkpoint) candidate to the kept V20 container.
# Usage: V22=<v22-container> V20=<v20-container> GPU=<gb300-index> ./rollback-v22.sh
set -euo pipefail
: "${V22:?set V22 to the v22 container name}" "${V20:?set V20 to the kept V20 container name}" "${GPU:?set GPU to the GB300 index}"
docker container inspect "$V20" >/dev/null
docker stop -t 60 "$V22"
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
sudo nvidia-smi -i "$GPU" --gpu-reset
docker start "$V20"
timeout 900 docker logs -f "$V20" 2>&1 | grep -m1 -E 'Application startup complete|Traceback|EngineCore failed' || true
