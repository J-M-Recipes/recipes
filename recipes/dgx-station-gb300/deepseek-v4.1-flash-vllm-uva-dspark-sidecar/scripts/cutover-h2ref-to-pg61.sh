#!/usr/bin/env bash
# cutover-h2ref-to-pg61.sh — move :30006 from h2-REF to Pg61 (Garner 8043a681 + our patch set, launch-pg61.sh).
#   --dry-run : preconditions only.
# Order: preconditions (h2-REF serving, no card runner, BUILD-SHA256SUMS verify) -> MAINTENANCE -> stop the v2 watchdog
# -> stop h2-REF pair (kept, restart=no) -> drop caches -> launch-pg61.sh up (verifies in-container hashes, arms the
# p3 watchdog in ENFORCE) -> smoke -> freeze the build (chmod a-w) -> clear MAINTENANCE -> owner line.
# Any failure after the stop runs rollback-pg61-to-h2ref.sh.
set -uo pipefail
DRY=0; [[ ${1:-} == --dry-run ]] && DRY=1
HERE=$(cd "$(dirname "$0")" && pwd)
GR=$HOME/dsv41/garner; M3=$HOME/dsv41/m3
LW=$GR/pg61-work; LAUNCH=${LAUNCH:-$GR/launcher/launch-pg61.sh}; GH=$GR/home
MODEL=${MODEL_DIR:-/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170}
SRV=dsv41-vllm-1017-h2-REF; RTX=dsv41-vllm-1017-h2-REF-rtx
OWNER=$M3/LANE-OWNER.txt; MAINT=$M3/logs/MAINTENANCE; U=http://127.0.0.1:30006
log(){ printf '%s [cutover-pg61] %s\n' "$(date '+%F %T %Z')" "$*"; }
wd_pids(){ ps -eo pid=,comm=,args= | awk '$2=="python3" && index($0,"sidecar_watchdog.py"){print $1}'; }
running(){ [[ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" == true ]]; }
pre=0
pgrep -f "g[0-9]+b?_runner.sh" >/dev/null && { log "REFUSE: a card runner is live"; pre=1; }
running $SRV && running $RTX || { log "REFUSE: h2-REF pair is not the serving lane"; pre=1; }
docker ps --format '{{.Names}}' | grep -qx ds41 && { log "REFUSE: ds41 already running"; pre=1; }
( cd $LW && sha256sum -c --quiet BUILD-SHA256SUMS ) || { log "REFUSE: pg61-work differs from BUILD-SHA256SUMS"; pre=1; }
bash $HERE/rollback-pg61-to-h2ref.sh --dry-run >/dev/null 2>&1 || log "note: rollback dry-run refuses while h2-REF serves only if containers are missing"
log "plan: stop $SRV/$RTX (kept); launch-pg61.sh up; smoke; freeze build"
[[ $DRY == 1 ]] && { log "DRY RUN: preconditions $([[ $pre == 0 ]] && echo OK || echo FAILED)"; exit $pre; }
[[ $pre == 0 ]] || exit 4
echo "cutover-pg61 $(date '+%F %T %Z')" > $MAINT
echo "- $(date '+%F %T %Z') CUTOVER h2-REF -> Pg61 starting (h2-REF pair stopped-and-kept = rollback; rollback-pg61-to-h2ref.sh)" >> $OWNER
for p in $(wd_pids); do kill "$p"; done; sleep 3
docker stop -t 60 $SRV >/dev/null; docker stop -t 60 $RTX >/dev/null; log "h2-REF pair stopped (kept)"
sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; sleep 45
if ! WORK=$LW MODEL_DIR=$MODEL DS41_HOME=$GH bash $LAUNCH up; then
  log "LAUNCHER UP FAILED -> rollback"; bash $HERE/rollback-pg61-to-h2ref.sh; exit 3; fi
if ! curl -s -m 60 $U/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"dsv41-flash-uva","messages":[{"role":"user","content":"Reply with exactly: lane ok"}],"max_tokens":16,"temperature":0,"chat_template_kwargs":{"thinking":false}}' | grep -q "lane ok"; then
  log "SMOKE FAILED -> rollback"; bash $HERE/rollback-pg61-to-h2ref.sh; exit 3; fi
log "smoke OK"
chmod -R a-w $LW/tree $LW/overlay-vllm $LW/BUILD-SHA256SUMS $LW/policy.env $LW/sidecar_watchdog_p3.py; sudo -n chmod a-w $LW/tree/fastpath/libw6mk.so; log "build frozen (chmod a-w; libw6mk.so is root-owned)"
rm -f $MAINT
echo "- $(date '+%F %T %Z') CUTOVER done: Pg61 (ds41 + ds41-sidecar, launch-pg61.sh) serving :30006 + p3 watchdog ENFORCE; soak 48 h; rollback: ~/dsv41/garner/promote/rollback-pg61-to-h2ref.sh" >> $OWNER
log "DONE: Pg61 serving"
