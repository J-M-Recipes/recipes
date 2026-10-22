#!/bin/bash
# campaign.sh EXPORTTAG -- on spark5. Wait for export, then run arms.txt rows (ARM PKG MAXC [flags]) whose ARM has no done marker. Re-reads arms.txt per arm.
set -u
WORK=${WORK:-$HOME/flare-prep-glm}   # campaign dir on the rank-0 node (logs, results, tools/)
H=$WORK; X=$1; L=$H/campaign.log
exec >> $L 2>&1
echo "CAMPAIGN START $(date) waiting for export $X"
if [ "$X" != none ]; then
  while [ "$(docker inspect -f '{{.State.Status}}' flare-glm-rank0 2>/dev/null)" = running ]; do sleep 30; done
  rc=$(docker inspect -f '{{.State.ExitCode}}' flare-glm-rank0); docker logs flare-glm-rank0 > $H/export-$X-rank0.full.log 2>&1
  echo "EXPORT $X rc=$rc $(date)"; grep -E "^\[    OK" $H/export-$X-rank0.full.log | cut -c1-120
  [ "$rc" != 0 ] && { echo "CAMPAIGN STOP export failed"; grep -E "Error|Traceback" $H/export-$X-rank0.full.log | tail -5; exit 1; }
  sudo -S -p '' -n chown -R $(id -u):$(id -g) ${FLARE_HOME:-$HOME/flare}/cache; docker rm flare-glm-rank0 >/dev/null; ssh ${PEER_SSH:?set PEER_SSH=<rank1-ssh>} "docker rm flare-glm-peer >/dev/null 2>&1"
fi
while true; do
  [ "$(cat $H/control 2>/dev/null)" = stop ] && { echo "CAMPAIGN STOPPED by control $(date)"; exit 0; }
  row=$(grep -v '^#' $H/arms.txt | while read a p m f; do [ -n "$a" ] && [ ! -f $H/results/$a.done ] && { echo "$a $p $m $f"; break; }; done)
  [ -z "$row" ] && { echo "CAMPAIGN DONE $(date)"; exit 0; }
  set -- $row; A=$1; P=$2; MC=$3; shift 3
  echo "ARM $A pkg=$P maxc=$MC flags=$* start $(date)"
  bash $H/arm_chain.sh $P $A $MC "$@"; rc=$?
  mkdir -p $H/results; echo "rc=$rc $(date)" > $H/results/$A.done
  echo "ARM $A end rc=$rc $(date)"; tail -12 $H/results/$A/battery.log 2>/dev/null | cut -c1-200
done
