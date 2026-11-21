#!/bin/bash
# xfer-20261121 card helper on the GB300 box (cloned from owen-20261116/owen_arm.sh). One op per call; every op ends "OP <op> <name> END rc=<n>".
#   xfer_arm.sh arm NAME OVSETS ARMENV   -> stop whatever ds41 pair runs, prep work-NAME (pg62-work + overlay sets), drop caches, launch-pg61 up
#        OVSETS = comma list of dirs under $XF/ov (e.g. cadence  or  g59131,r63540  or none); ARMENV = "K=V K=V" or "-"
#   xfer_arm.sh restore                  -> stop whatever runs, drop caches, launch-pg61 up on pg62-work (production)
set -uo pipefail
GR=~/dsv41/garner; XF=~/dsv41/xfer-20261121
MODEL_DIR=/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170
LO1=~/dsv41/m3/LANE-OWNER.txt; LO2=~/dsv41/m3/logs/LANE-OWNER.txt
ts(){ date "+%F %T %Z"; }
lo(){ echo "- $(ts) $*" >> $LO1; echo "- $(ts) $*" >> $LO2; }
log(){ echo "$(ts) [xfer] $*"; }
op=${1:?op}; name=${2:-pg62}
cur_work(){
  local w; w=$(ps -eo args | awk 'index($0,"sidecar_watchdog_p3.py --server ds41") && $1 ~ /python3/ {print $2}' | head -1)
  [ -n "$w" ] && dirname "$w" || echo "$GR/pg62-work"; }
down_any(){
  local w; w=$(cur_work); log "down via WORK=$w"
  env -u DS41_LPT -u DS41_ARM_ENV WORK=$w DS41_HOME=$GR/home bash $GR/launcher/launch-pg61.sh down || return 1
}
running_ds41(){ [ "$(docker inspect -f '{{.State.Running}}' ds41 2>/dev/null)" = true ]; }
settle(){ sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; sleep 45; free -g | sed -n 2p; df -h / | tail -1; }
finish(){ log "OP $op $name END rc=$1"; exit $1; }
other=$(docker ps --format '{{.Names}}' | grep -vxE 'ds41|ds41-sidecar' || true)
[ -n "$other" ] && { log "ABORT foreign containers: $other"; finish 3; }
case $op in
arm)
  ovs=${3:?ovsets}; armenv=${4:?armenv}; [ "$armenv" = "-" ] && armenv=""
  A=$XF/work-$name
  echo "xfer-20261121 card window (Milo; James 11/21 'let's get to work'): lane may be dark; restore is scripted (xfer_arm.sh restore)" > ~/dsv41/m3/logs/MAINTENANCE
  [ -e "$A" ] && chmod -R u+w "$A" 2>/dev/null; rm -rf "$A"; [ -e "$A" ] && finish 4; mkdir -p "$A"
  cp -a $GR/pg62-work/tree $GR/pg62-work/overlay-vllm $GR/pg62-work/sidecar_watchdog_p3.py $GR/pg62-work/policy.env "$A"/ || finish 4
  chmod -R u+w "$A" || finish 4
  cp $XF/serve.sh.arm "$A/tree/serve/serve.sh" || finish 4
  sed -i "s#^DS41_OVERLAY_58132_DIR=.*#DS41_OVERLAY_58132_DIR=$A/overlay-vllm#" "$A/policy.env"
  nfiles=0
  if [ "$ovs" != none ]; then
    for s in ${ovs//,/ }; do
      [ -d "$XF/ov/$s" ] || { log "missing overlay set $s"; finish 4; }
      ( cd "$XF/ov/$s" && find . -type f -name '*.py' ) | while read -r f; do
        [ -e "$A/overlay-vllm/$f" ] && { log "overlay collision $f"; exit 9; }
        mkdir -p "$A/overlay-vllm/$(dirname "$f")"; cp "$XF/ov/$s/$f" "$A/overlay-vllm/$f"
      done || finish 4
      nfiles=$((nfiles + $(cd "$XF/ov/$s" && find . -type f -name '*.py' | wc -l)))
    done
  fi
  nd=$(diff -rq $GR/pg62-work/tree "$A/tree" | grep -v '__pycache__' | wc -l)
  no=$(diff -rq $GR/pg62-work/overlay-vllm "$A/overlay-vllm" | grep -v '__pycache__' | wc -l)
  log "arm $name prepared: ovsets=$ovs armenv='$armenv'; tree diffs=$nd (expect 1: serve.sh); overlay diffs=$no (expect $nfiles)"
  diff -rq $GR/pg62-work/tree "$A/tree" | grep -v __pycache__; diff -rq $GR/pg62-work/overlay-vllm "$A/overlay-vllm" | grep -v __pycache__
  [ "$nd" = 1 ] || finish 5; [ "$no" = "$nfiles" ] || finish 5
  lo "XFER card arm $name START (ovsets=$ovs env='$armenv') — Pg62 stopped-and-kept, lane DARK until arm binds; restore scripted (xfer_arm.sh restore)"
  down_any || finish 6
  settle
  if env DS41_LPT=6144 DS41_ARM_ENV="$armenv" WORK=$A MODEL_DIR=$MODEL_DIR DS41_HOME=$GR/home bash $GR/launcher/launch-pg61.sh up; then
    args=$(docker inspect -f '{{json .Args}}' ds41); envs=$(docker inspect -f '{{json .Config.Env}}' ds41)
    echo "$args" | grep -q '"--long-prefill-token-threshold","6144"' || { log "ARG CHECK FAILED"; finish 7; }
    for kv in $armenv; do echo "$envs" | grep -q "\"$kv\"" || { log "ENV CHECK FAILED $kv"; finish 7; }; done
    if [ "$ovs" != none ]; then
      for s in ${ovs//,/ }; do
        for f in $(cd "$XF/ov/$s" && find . -type f -name '*.py'); do
          h1=$(sha256sum "$XF/ov/$s/$f" | cut -c1-16); h2=$(docker exec ds41 sha256sum "/usr/local/lib/python3.12/dist-packages/vllm/${f#./}" | cut -c1-16)
          [ "$h1" = "$h2" ] || { log "OVERLAY CHECK FAILED $f $h1 != $h2"; finish 7; }
          log "overlay verified in container: $f $h2"
        done
      done
    fi
    lo "XFER card arm $name BOUND guarded on :30006 (ovsets=$ovs env='$armenv')"
    finish 0
  else finish 8; fi ;;
restore)
  if [ "$(cur_work)" = "$GR/pg62-work" ] && running_ds41 && curl -sf -m 5 http://127.0.0.1:30006/v1/models | grep -q dsv41-flash-uva \
     && docker inspect -f '{{json .Args}}' ds41 | grep -q '"--long-prefill-token-threshold","6144"' \
     && ! docker inspect -f '{{json .Config.Env}}' ds41 | grep -qE 'VLLM_PREFILL_(YIELD|CADENCE)'; then
    log "production Pg62 already serving; nothing to restore"
    [ -e ~/dsv41/m3/logs/MAINTENANCE ] && mv ~/dsv41/m3/logs/MAINTENANCE ~/dsv41/m3/logs/MAINTENANCE.stale-xfer-$(date +%Y%m%d-%H%M)
    finish 0
  fi
  lo "XFER card RESTORE Pg62 starting"
  down_any || finish 6
  settle
  if env -u DS41_LPT -u DS41_ARM_ENV WORK=$GR/pg62-work MODEL_DIR=$MODEL_DIR DS41_HOME=$GR/home bash $GR/launcher/launch-pg61.sh up; then
    docker inspect -f '{{json .Args}}' ds41 | grep -q '"--long-prefill-token-threshold","6144"' || { log "restore arg check failed"; finish 7; }
    docker inspect -f '{{json .Config.Env}}' ds41 | grep -qE 'VLLM_PREFILL_(YIELD|CADENCE)' && { log "restore env check failed"; finish 7; }
    [ -e ~/dsv41/m3/logs/MAINTENANCE ] && mv ~/dsv41/m3/logs/MAINTENANCE ~/dsv41/m3/logs/MAINTENANCE.stale-xfer-$(date +%Y%m%d-%H%M)
    lo "XFER card RESTORE done: Pg62 (pg62-work, production args) serving :30006; p3 watchdog ENFORCE; MAINTENANCE archived"
    finish 0
  else lo "XFER card RESTORE FAILED; lane DARK; MAINTENANCE kept"; finish 8; fi ;;
*) finish 2 ;;
esac
