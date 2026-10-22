#!/bin/bash
# serve-arm.sh PKGTAG [runtime flags...]  -- run on spark5 (controller). Detached orbit group "glm"; logs to serve-<PKGTAG>.log
set -u
WORK=${WORK:-$HOME/flare-prep-glm}   # campaign dir on the rank-0 node (logs, results, tools/)
CLUSTER=${CLUSTER:-sparks56}; RANK0=${RANK0:-spark5}   # orbit cluster name, rank-0 node name
PKG=$1; shift
P=$HOME/flare/cache/flare/runtime/glm53-fp4local-tp2-$PKG.safetensors
FP=${FP:?set FP to the package signer fingerprint (flare inspect <package> prints it)}
L=$WORK/serve-$PKG.log
export PATH=$HOME/flare/bin:$PATH
echo "SERVE START $(date) pkg=$P fp=$FP flags=$*" | tee -a $L
for h in ${NODES:?set NODES="<rank0-ssh> <rank1-ssh>"}; do ssh -o BatchMode=yes $h "sudo -n sh -c 'sync; echo 1 > /proc/sys/vm/compact_memory; echo 3 > /proc/sys/vm/drop_caches'"; done
flare orbit run $RANK0:$P --cluster $CLUSTER --name glm --gpus 0 --host 0.0.0.0 --port 8003 --served-model-name GLM-5.3-Flash \
  --dangerously-approve-key $FP --dangerously-accept-publisher local-development "$@" >> $L 2>&1
echo "SERVE EXIT $? $(date)" >> $L
