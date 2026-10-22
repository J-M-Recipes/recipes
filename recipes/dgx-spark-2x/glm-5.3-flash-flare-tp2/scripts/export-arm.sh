#!/bin/bash
# export-arm.sh TAG ROLE(rank0|peer)  env: CONC MTP CAP CHUNK EXTRA
set -u
WORK=${WORK:-$HOME/flare-prep-glm}   # campaign dir on the rank-0 node (logs, results, tools/)
TAG=$1; ROLE=$2
IMG=${IMG:-flare-prep:47581cc}
CKPT=${CKPT:-$HOME/models/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD-dec48abd}
RDV=${RDV:?set RDV=<rank-0 fabric IP>:29630}
OUT=/root/.cache/flare/runtime/glm53-fp4local-tp2-$TAG.safetensors
CONC=${CONC:-4}; MTP=${MTP:-5}; CHUNK=${CHUNK:-4096}; CAP=${CAP:-}; EXTRA=${EXTRA:-}
COMMON="--network host --gpus all --ipc host --ulimit memlock=-1:-1 --device /dev/infiniband --cap-add IPC_LOCK --cap-add SYS_NICE --security-opt seccomp=unconfined -v $CKPT:$CKPT:ro -v ${FLARE_HOME:-$HOME/flare}/config:/root/.config/flare -v ${FLARE_HOME:-$HOME/flare}/cache:/root/.cache"
L=$WORK/export-$TAG-$ROLE.log
exec > >(tee -a $L) 2>&1
echo "START $ROLE $TAG $(date) IMG=$IMG CKPT=$CKPT CONC=$CONC MTP=$MTP CAP=$CAP CHUNK=$CHUNK EXTRA=$EXTRA"
docker ps --format '{{.Names}}' | grep -qE '^flare-glm-' && { echo "REFUSE: flare-glm container running"; exit 2; }
sudo -S -p '' -n sh -c 'sync; echo 1 > /proc/sys/vm/compact_memory; echo 3 > /proc/sys/vm/drop_caches'; free -g | awk 'NR==2{print "MemAvailable GiB", $7}'
docker rm -f flare-glm-$ROLE >/dev/null 2>&1
if [ "$ROLE" = rank0 ]; then
  CAPF=""; [ -n "$CAP" ] && CAPF="--speculative-token-capacity $CAP"
  docker run -d --name flare-glm-rank0 $COMMON $IMG .venv/bin/python -m ignition export $CKPT --gpus 0 --ranks 2 --rendezvous $RDV \
    --max-concurrency $CONC --mtp-draft-tokens $MTP --prefill-chunk-size $CHUNK $CAPF $EXTRA --output $OUT --verbose
else
  docker run -d --name flare-glm-peer $COMMON $IMG .venv/bin/ignition join $RDV --rank 1 --ranks 2
fi
sleep 2; docker ps --format '{{.Names}} {{.Status}}' | grep flare-glm
