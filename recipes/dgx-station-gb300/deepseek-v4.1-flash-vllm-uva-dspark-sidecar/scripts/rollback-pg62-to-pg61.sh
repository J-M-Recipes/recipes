#!/bin/bash
# Published copy (2026-11-15) of the reference-box rollback script. Scrubbed: box home path -> $HOME, GR/MODEL_DIR
# overridable from the environment; nothing else changed. NEVER DRILLED: no real rollback has run (no --dry-run log is
# in results/2026-11-14-pg62-w1-exact/raw/ either; the dry run checks pg61-work's BUILD-SHA256SUMS). After a real
# rollback, point the box's autostart (promote/autostart-pg61.sh, WORK=pg62-work since the cutover) and the lane check
# (PG_WORK) back to pg61-work by hand; this script does not do it.
# Rollback Pg62 -> Pg61 on :30006. Usage: rollback-pg62-to-pg61.sh [--dry-run]
set -uo pipefail
GR=${GR:-$HOME/dsv41/garner}; MODEL_DIR=${MODEL_DIR:-/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170}
ts(){ date "+%F %T %Z"; }; LO=$HOME/dsv41/m3/LANE-OWNER.txt
if [ "${1:-}" = --dry-run ]; then
  echo "would: launch-pg61.sh down (WORK=$GR/pg62-work); drop caches; launch-pg61.sh up (WORK=$GR/pg61-work)"
  ( cd $GR/pg61-work && sha256sum -c --quiet BUILD-SHA256SUMS ) && echo "pg61-work build hashes OK" || echo "pg61-work HASH MISMATCH"
  docker ps --format '{{.Names}}' | tr '\n' ' '; echo; exit 0; fi
LOG=$GR/promote/rollback-pg62-$(date +%Y%m%d-%H%M).log; exec >>"$LOG" 2>&1
echo "- $(ts) ROLLBACK Pg62 -> Pg61 starting; log $LOG" >> $LO
( cd $GR/pg61-work && sha256sum -c --quiet BUILD-SHA256SUMS ) || { echo "$(ts) ABORT pg61-work hash mismatch"; exit 5; }
env WORK=$GR/pg62-work DS41_HOME=$GR/home bash $GR/launcher/launch-pg61.sh down || true
[ -z "$(docker ps -q)" ] || { echo "$(ts) ABORT containers still running"; exit 3; }
sudo -S -p '' -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; sleep 45
if env WORK=$GR/pg61-work MODEL_DIR=$MODEL_DIR DS41_HOME=$GR/home bash $GR/launcher/launch-pg61.sh up; then
  echo "- $(ts) ROLLBACK done: Pg61 serving :30006 (pg61-work); p3 watchdog ENFORCE; autostart/lane-check must be pointed back to pg61-work" >> $LO; echo "$(ts) UP Pg61"
else rc=$?; echo "- $(ts) ROLLBACK FAILED rc=$rc; lane DARK; see $LOG" >> $LO; exit $rc; fi
