#!/bin/bash
# Roll back Pg67a -> Pg66 (production tree garner/pg66-work, untouched). ~10 min dark.
set -uo pipefail
GR=~/dsv41/garner
MODEL_DIR=/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170
LOG=$GR/promote/rollback-pg67a-$(date +%Y%m%d-%H%M).log; exec >>"$LOG" 2>&1
ts(){ date "+%F %T %Z"; }; lo(){ echo "- $(ts) $*" >> ~/dsv41/m3/LANE-OWNER.txt; }
lo "ROLLBACK Pg67a -> Pg66 starting"
env WORK=~/dsv41/sidecar-resid/work-Pg67a DS41_HOME=$GR/home bash $GR/launcher/launch-pg61.sh down || true
sleep 20
for i in $(seq 1 60); do docker ps -a --format '{{.Names}}' | grep -q '^ds41\(-sidecar\)\?$' || break; sleep 2; done
sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; sleep 45
if env WORK=$GR/pg66-work MODEL_DIR=$MODEL_DIR DS41_HOME=$GR/home bash $GR/launcher/launch-pg61.sh up; then
  sed -i 's#/sidecar-resid/work-Pg67a#/garner/pg66-work#' $GR/promote/autostart-pg61.sh
  lo "ROLLBACK done: Pg66 (pg66-work) serving :30006; autostart repointed to pg66-work"
else lo "ROLLBACK FAILED; lane DARK; see $LOG"; exit 1; fi
