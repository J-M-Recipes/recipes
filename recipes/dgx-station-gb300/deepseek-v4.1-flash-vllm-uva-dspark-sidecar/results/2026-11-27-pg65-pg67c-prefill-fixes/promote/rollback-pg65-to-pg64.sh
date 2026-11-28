#!/bin/bash
# Roll back Pg65 -> Pg64 (production tree garner/pg64-work, untouched). ~7 min dark.
# Faster alternative without a rebuild: add VLLM_DSV41_DSML_TOKEN_PROVENANCE=0 to DS41_ARM_ENV in pg65-work/policy.env and relaunch.
set -uo pipefail
GR=~/dsv41/garner
MODEL_DIR=/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170
LOG=$GR/promote/rollback-pg65-$(date +%Y%m%d-%H%M).log; exec >>"$LOG" 2>&1
ts(){ date "+%F %T %Z"; }; lo(){ echo "- $(ts) $*" >> ~/dsv41/m3/LANE-OWNER.txt; echo "- $(ts) $*" >> ~/dsv41/m3/logs/LANE-OWNER.txt; }
lo "ROLLBACK Pg65 -> Pg64 starting"
env WORK=$GR/pg65-work DS41_HOME=$GR/home bash $GR/launcher/launch-pg61.sh down || true
sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; sleep 45
if env WORK=$GR/pg64-work MODEL_DIR=$MODEL_DIR DS41_HOME=$GR/home bash $GR/launcher/launch-pg61.sh up; then
  lo "ROLLBACK done: Pg64 (pg64-work) serving :30006; repoint ~/.hermes/profiles/milo/scripts/dsv41_pg61_lane_check.sh W= back to pg64-work"
else lo "ROLLBACK FAILED; lane DARK; see $LOG"; exit 1; fi
