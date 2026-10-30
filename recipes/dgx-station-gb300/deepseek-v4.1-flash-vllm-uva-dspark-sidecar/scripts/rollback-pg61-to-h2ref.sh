#!/usr/bin/env bash
# rollback-pg61-to-h2ref.sh — put :30006 back on h2-REF (dsv41-vllm-1017-h2-REF + -rtx, stopped-and-kept) from Pg61.
#   --dry-run : print the plan and check preconditions (no docker changes).
# Order: MAINTENANCE flag -> p3 watchdog + Pg61 down (launcher) -> drop caches -> rtx first, wait for a FRESH
# 'serving' line -> server -> bound -> re-arm the v2 fail-stop watchdog from its argv file (mapfile, one arg per line)
# BEFORE judging the smoke -> smoke -> clear MAINTENANCE -> owner line. Refuses while a card runner is live.
set -uo pipefail
DRY=0; [[ ${1:-} == --dry-run ]] && DRY=1
GR=$HOME/dsv41/garner; M3=$HOME/dsv41/m3
LW=$GR/pg61-work; LAUNCH=${LAUNCH:-$GR/launcher/launch-pg61.sh}; GH=$GR/home
SRV=dsv41-vllm-1017-h2-REF; RTX=dsv41-vllm-1017-h2-REF-rtx
WDARGV=$M3/ap2/promote/wd-argv-dsv41-vllm-1017-h2-REF.txt; WDS=$M3/logs/sidecar_watchdog.status.json
OWNER=$M3/LANE-OWNER.txt; MAINT=$M3/logs/MAINTENANCE; U=http://127.0.0.1:30006
log(){ printf '%s [rollback-pg61] %s\n' "$(date '+%F %T %Z')" "$*"; }
wd_pids(){ ps -eo pid=,comm=,args= | awk '$2=="python3" && index($0,"sidecar_watchdog.py"){print $1}'; }
running(){ [[ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" == true ]]; }
pre=0
pgrep -f "g[0-9]+b?_runner.sh" >/dev/null && { log "REFUSE: a card runner is live"; pre=1; }
for c in $SRV $RTX; do docker inspect "$c" >/dev/null 2>&1 || { log "REFUSE: $c does not exist (nothing to roll back to)"; pre=1; }; done
[[ -s $WDARGV ]] || { log "REFUSE: watchdog argv $WDARGV missing"; pre=1; }
[[ -x $LAUNCH || -f $LAUNCH ]] || { log "REFUSE: launcher missing"; pre=1; }
log "plan: MAINTENANCE; launch-pg61.sh down; start $RTX (fresh serving) then $SRV; re-arm watchdog ($(wc -l < $WDARGV) argv lines); smoke"
log "now serving: $(docker ps --format '{{.Names}}' | grep -E '^ds41|^dsv41-' | tr '\n' ' ')"
[[ $DRY == 1 ]] && { log "DRY RUN: preconditions $([[ $pre == 0 ]] && echo OK || echo FAILED)"; exit $pre; }
[[ $pre == 0 ]] || exit 4
echo "rollback-pg61 $(date '+%F %T %Z')" > $MAINT
WORK=$LW DS41_HOME=$GH bash $LAUNCH down; log "Pg61 down"
sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; sleep 20
t0=$(date -u +%FT%TZ); docker start $RTX >/dev/null
for i in $(seq 1 120); do docker logs --since "$t0" $RTX 2>&1 | grep -q '^serving' && { log "rtx serving (fresh)"; break; }; sleep 5; done
docker start $SRV >/dev/null
for i in $(seq 1 240); do curl -sf -m 3 $U/v1/models | grep -q dsv41-flash-uva && { log "BOUND after $((i*5))s"; break; }; sleep 5; done
if [[ -z $(wd_pids) ]]; then mapfile -t WA < "$WDARGV"; ( cd $M3; setsid nohup "${WA[@]}" >/dev/null 2>&1 </dev/null & ); sleep 8; fi
grep -q '"trip": null' $WDS && log "watchdog armed pid=$(wd_pids | tr '\n' ' ')" || log "WARN watchdog not trip null"
if curl -s -m 60 $U/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"dsv41-flash-uva","messages":[{"role":"user","content":"Reply with exactly: lane ok"}],"max_tokens":16,"temperature":0,"chat_template_kwargs":{"thinking":false}}' | grep -q "lane ok"; then
  log "smoke OK"; rm -f $MAINT; echo "- $(date '+%F %T %Z') ROLLBACK Pg61 -> h2-REF done; h2-REF pair serving + v2 watchdog" >> $OWNER; exit 0
fi
log "SMOKE FAILED — lane left in MAINTENANCE with the watchdog armed; investigate"; exit 2
