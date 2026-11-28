#!/bin/bash
# Roll back Pg66 -> Pg65 (production tree garner/pg65-work, untouched). ~7 min dark.
# Faster alternative without a new tree: remove w8oproj,lowb_aux from DS41_FASTPATH_COMPONENTS_OVERRIDE in pg66-work/policy.env
# (chmod u+w first) and relaunch pg66-work.
set -uo pipefail
GR=~/dsv41/garner
MODEL_DIR=/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170
LOG=$GR/promote/rollback-pg66-$(date +%Y%m%d-%H%M).log; exec >>"$LOG" 2>&1
ts(){ date "+%F %T %Z"; }; lo(){ echo "- $(ts) $*" >> ~/dsv41/m3/LANE-OWNER.txt; echo "- $(ts) $*" >> ~/dsv41/m3/logs/LANE-OWNER.txt; }
lo "ROLLBACK Pg66 -> Pg65 starting"
env WORK=$GR/pg66-work DS41_HOME=$GR/home bash $GR/launcher/launch-pg61.sh down || true
sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; sleep 45
if env WORK=$GR/pg65-work MODEL_DIR=$MODEL_DIR DS41_HOME=$GR/home bash $GR/launcher/launch-pg61.sh up; then
  sed -i 's#/garner/pg66-work#/garner/pg65-work#' $GR/promote/autostart-pg61.sh
  lo "ROLLBACK done: Pg65 (pg65-work) serving :30006; autostart repointed to pg65-work; repoint M4 ~/.hermes/profiles/milo/scripts/dsv41_pg61_lane_check.sh W= back to pg65-work"
else lo "ROLLBACK FAILED; lane DARK; see $LOG"; exit 1; fi
