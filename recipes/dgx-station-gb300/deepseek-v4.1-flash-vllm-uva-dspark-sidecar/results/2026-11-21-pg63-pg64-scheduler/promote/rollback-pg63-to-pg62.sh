#!/bin/bash
# Roll back Pg63 -> Pg62 (production tree garner/pg62-work, untouched). ~7 min dark.
set -uo pipefail
GR=~/dsv41/garner
MODEL_DIR=/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170
LOG=$GR/promote/rollback-pg63-$(date +%Y%m%d-%H%M).log; exec >>"$LOG" 2>&1
ts(){ date "+%F %T %Z"; }; lo(){ echo "- $(ts) $*" >> ~/dsv41/m3/LANE-OWNER.txt; echo "- $(ts) $*" >> ~/dsv41/m3/logs/LANE-OWNER.txt; }
lo "ROLLBACK Pg63 -> Pg62 starting"
env WORK=$GR/pg63-work DS41_HOME=$GR/home bash $GR/launcher/launch-pg61.sh down || true
sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; sleep 45
if env WORK=$GR/pg62-work MODEL_DIR=$MODEL_DIR DS41_HOME=$GR/home bash $GR/launcher/launch-pg61.sh up; then
  lo "ROLLBACK done: Pg62 (pg62-work) serving :30006; repoint ~/.hermes/profiles/milo/scripts/dsv41_pg61_lane_check.sh W= back to pg62-work (.bak-pg62 kept)"
else lo "ROLLBACK FAILED; lane DARK; see $LOG"; exit 1; fi
