#!/bin/bash
# xfer-20261121 card helper v2 — arms are built on PRODUCTION (Pg64 since 2026-11-21 12:08: garner/pg64-work). One op per call; ends "OP <op> <name> END rc=<n>".
#   xfer_arm2.sh arm NAME OVSETS ARMENV SIDEENV
#        OVSETS: comma list under $XF/ov whose files may REPLACE pg63 overlay files (e.g. yieldcad) or none
#        ARMENV: full ds41 env (replaces pg63's DS41_ARM_ENV; "K=V K=V"), or "-" = keep pg63's
#        SIDEENV: sidecar env ("K=V ..."), or "-" = none
#   xfer_arm2.sh restore    -> production (pg64-work)
set -uo pipefail
GR=~/dsv41/garner; XF=~/dsv41/xfer-20261121; PROD=$GR/pg64-work
MODEL_DIR=/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170
LO1=~/dsv41/m3/LANE-OWNER.txt; LO2=~/dsv41/m3/logs/LANE-OWNER.txt
ts(){ date "+%F %T %Z"; }; lo(){ echo "- $(ts) $*" >> $LO1; echo "- $(ts) $*" >> $LO2; }; log(){ echo "$(ts) [xfer2] $*"; }
op=${1:?op}; name=${2:-pg63}
cur_work(){ local w; w=$(ps -eo args | awk 'index($0,"sidecar_watchdog_p3.py --server ds41") && $1 ~ /python3/ {print $2}' | head -1); [ -n "$w" ] && dirname "$w" || echo "$PROD"; }
down_any(){ local w; w=$(cur_work); log "down via WORK=$w"; WORK=$w DS41_HOME=$GR/home bash $GR/launcher/launch-pg61.sh down; }
settle(){ sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; sleep 45; free -g | sed -n 2p; }
finish(){ log "OP $op $name END rc=$1"; exit $1; }
other=$(docker ps --format '{{.Names}}' | grep -vxE 'ds41|ds41-sidecar' || true); [ -n "$other" ] && { log "ABORT foreign containers: $other"; finish 3; }
case $op in
arm)
  ovs=${3:?}; armenv=${4:?}; sideenv=${5:?}; A=$XF/work2-$name
  echo "xfer-20261121 card window (Milo; James 11/21 'Go all'): lane may be dark; restore scripted (xfer_arm2.sh restore -> Pg64)" > ~/dsv41/m3/logs/MAINTENANCE
  [ -e "$A" ] && chmod -R u+w "$A" 2>/dev/null; rm -rf "$A"; [ -e "$A" ] && finish 4; mkdir -p "$A"
  cp -a $PROD/tree $PROD/overlay-vllm $PROD/sidecar_watchdog_p3.py $PROD/policy.env "$A"/ || finish 4
  chmod -R u+w "$A" || finish 4
  cp $XF/serve.sh.arm2 "$A/tree/serve/serve.sh" || finish 4
  sed -i "s#^DS41_OVERLAY_58132_DIR=.*#DS41_OVERLAY_58132_DIR=$A/overlay-vllm#" "$A/policy.env"
  if [ "$armenv" != "-" ]; then sed -i '/^DS41_ARM_ENV=/d' "$A/policy.env"; echo "DS41_ARM_ENV=$armenv" >> "$A/policy.env"; fi
  [ "$sideenv" != "-" ] && echo "DS41_SIDECAR_ENV=$sideenv" >> "$A/policy.env"
  nrep=0
  if [ "$ovs" != none ]; then for s in ${ovs//,/ }; do
      [ -d "$XF/ov/$s" ] || { log "missing overlay set $s"; finish 4; }
      for f in $(cd "$XF/ov/$s" && find . -type f -name '*.py'); do mkdir -p "$A/overlay-vllm/$(dirname "$f")"; cp "$XF/ov/$s/$f" "$A/overlay-vllm/$f"; nrep=$((nrep+1)); done
  done; fi
  nd=$(diff -rq $PROD/tree "$A/tree" | grep -v __pycache__ | wc -l); no=$(diff -rq $PROD/overlay-vllm "$A/overlay-vllm" | grep -v __pycache__ | wc -l)
  log "arm $name prepared: ovsets=$ovs armenv='$armenv' sideenv='$sideenv'; tree diffs=$nd (expect 1) overlay diffs=$no (<= $nrep)"; cat "$A/policy.env" | grep -E "ARM_ENV|SIDECAR_ENV|OVERLAY"
  [ "$nd" = 1 ] || finish 5; [ "$no" -le "$nrep" ] || finish 5
  lo "XFER2 card arm $name START (Pg64-based; ovsets=$ovs env='$armenv' sidecar='$sideenv') — Pg64 stopped-and-kept, lane DARK until arm binds; restore scripted (xfer_arm2.sh restore)"
  down_any || finish 6; settle
  if env MODEL_DIR=$MODEL_DIR WORK=$A DS41_HOME=$GR/home bash $GR/launcher/launch-pg61.sh up; then
    envs=$(docker inspect -f '{{json .Config.Env}}' ds41); senv=$(docker inspect -f '{{json .Config.Env}}' ds41-sidecar)
    docker inspect -f '{{json .Args}}' ds41 | grep -q '"--long-prefill-token-threshold","6144"' || { log "ARG CHECK FAILED"; finish 7; }
    want=$armenv; [ "$want" = "-" ] && want=$(grep '^DS41_ARM_ENV=' $PROD/policy.env | cut -d= -f2-)
    for kv in $want; do echo "$envs" | grep -q "\"$kv\"" || { log "ENV CHECK FAILED $kv"; finish 7; }; done
    if [ "$sideenv" != "-" ]; then for kv in $sideenv; do echo "$senv" | grep -q "\"$kv\"" || { log "SIDECAR ENV CHECK FAILED $kv"; finish 7; }; done; fi
    for f in $(cd "$A/overlay-vllm" && find v1/core/sched -name '*.py' 2>/dev/null); do
      h1=$(sha256sum "$A/overlay-vllm/$f" | cut -c1-16); h2=$(docker exec ds41 sha256sum "/usr/local/lib/python3.12/dist-packages/vllm/$f" | cut -c1-16)
      [ "$h1" = "$h2" ] || { log "OVERLAY CHECK FAILED $f"; finish 7; }; log "overlay verified: $f $h2"; done
    lo "XFER2 card arm $name BOUND guarded on :30006"
    finish 0
  else finish 8; fi ;;
restore)
  if [ "$(cur_work)" = "$PROD" ] && [ "$(docker inspect -f '{{.State.Running}}' ds41 2>/dev/null)" = true ] && curl -sf -m 5 http://127.0.0.1:30006/v1/models | grep -q dsv41-flash-uva; then
    log "production Pg64 already serving"; [ -e ~/dsv41/m3/logs/MAINTENANCE ] && mv ~/dsv41/m3/logs/MAINTENANCE ~/dsv41/m3/logs/MAINTENANCE.stale-xfer2-$(date +%Y%m%d-%H%M); finish 0; fi
  lo "XFER2 card RESTORE Pg64 starting"
  down_any || finish 6; settle
  if env MODEL_DIR=$MODEL_DIR WORK=$PROD DS41_HOME=$GR/home bash $GR/launcher/launch-pg61.sh up; then
    e=$(docker inspect -f '{{json .Config.Env}}' ds41)
    echo "$e" | grep -q '"VLLM_PREFILL_CADENCE=3"' || { log "restore env check failed (cadence)"; finish 7; }
    echo "$e" | grep -q '"VLLM_PREFILL_YIELD=short"' || { log "restore env check failed (yield missing)"; finish 7; }
    docker inspect -f '{{json .Config.Env}}' ds41-sidecar | grep -q B12X_DYNAMIC && { log "restore sidecar env check failed"; finish 7; }
    mv ~/dsv41/m3/logs/MAINTENANCE ~/dsv41/m3/logs/MAINTENANCE.stale-xfer2-$(date +%Y%m%d-%H%M) 2>/dev/null
    lo "XFER2 card RESTORE done: Pg64 (pg64-work) serving :30006; p3 watchdog ENFORCE; MAINTENANCE archived"; finish 0
  else lo "XFER2 card RESTORE FAILED; lane DARK; MAINTENANCE kept"; finish 8; fi ;;
*) finish 2 ;;
esac
