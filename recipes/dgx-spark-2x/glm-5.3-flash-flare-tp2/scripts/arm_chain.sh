#!/bin/bash
# arm_chain.sh PKGTAG ARMTAG MAXC [runtime flags...] -- on spark5: stop group, serve, wait ready (25 min cap), battery, END marker.
set -u
WORK=${WORK:-$HOME/flare-prep-glm}   # campaign dir on the rank-0 node (logs, results, tools/)
CLUSTER=${CLUSTER:-sparks56}; RANK0=${RANK0:-spark5}
PKG=$1; ARM=$2; MAXC=$3; shift 3
H=$WORK; C=$H/chain-$ARM.log; export PATH=$HOME/flare/bin:$PATH
exec >> $C 2>&1
echo "CHAIN START $(date) pkg=$PKG arm=$ARM maxc=$MAXC flags=$*"
flare orbit stop glm --cluster $CLUSTER >/dev/null 2>&1; sleep 5
setsid nohup bash $H/serve-arm.sh $PKG "$@" > /dev/null 2>&1 < /dev/null &
t0=$(date +%s)
until curl -s -m 3 http://127.0.0.1:8003/v1/models | grep -q GLM; do
  sleep 15; el=$(( $(date +%s) - t0 ))
  if grep -q "SERVE EXIT" $H/serve-$PKG.log 2>/dev/null && ! pgrep -f "orbit run $RANK0" >/dev/null; then echo "SERVE DIED after ${el}s"; tail -30 $H/serve-$PKG.log; echo "CHAIN END FAIL $(date)"; exit 1; fi
  [ $el -gt 1500 ] && { echo "READY TIMEOUT"; tail -30 $H/serve-$PKG.log; echo "CHAIN END FAIL $(date)"; exit 1; }
done
echo "READY after $(( $(date +%s) - t0 ))s $(date)"
grep -iE "KV|tokens|pages|cache" $H/serve-$PKG.log | tail -8
bash $H/arm_battery.sh $ARM $MAXC
echo "CHAIN END OK $(date)"
