#!/usr/bin/env bash
# restore_after.sh — put :30006 back on the production h2-REF pair + fail-stop watchdog after any extra boot
# (used after verify_b.sh). Idempotent; mirrors what cardTR2_runner.sh does at END.
set -uo pipefail
log(){ echo "[$(date '+%F %T')] $*"; }
for n in $(docker ps --format '{{.Names}}' | grep -E '^dsv41-vllm-'); do docker stop -t 60 "$n" >/dev/null; log "stopped $n"; done
sudo -n iptables -D INPUT -p tcp --dport 30006 ! -i lo -j REJECT --reject-with tcp-reset 2>/dev/null && log "ISOLATE removed"
log "starting dsv41-vllm-1017-h2-REF-rtx"
docker start dsv41-vllm-1017-h2-REF-rtx >/dev/null
for i in $(seq 1 120); do docker logs --tail 8 dsv41-vllm-1017-h2-REF-rtx 2>&1 | grep -qE "serving|Application startup complete|listening" && { log "rtx ready"; break; }; sleep 5; done
log "starting dsv41-vllm-1017-h2-REF"
docker start dsv41-vllm-1017-h2-REF >/dev/null
for i in $(seq 1 180); do curl -sf -m 3 http://127.0.0.1:30006/v1/models >/dev/null 2>&1 && { log "server BOUND after $((i*5))s"; break; }; sleep 5; done
WDARGV=$HOME/dsv41/m3/ap2/promote/wd-argv-dsv41-vllm-1017-h2-REF.txt
if [[ -s $WDARGV ]]; then
  # The argv file is NEWLINE-SEPARATED, one argument per line (not a shell command line): read it into an array
  # and exec it. `bash -c "$(cat $WDARGV)"` starts `python3` with no args, dies on the second line, and leaves the
  # lane serving unguarded — that is exactly how the first restore after Card TR2 came back without a watchdog.
  mapfile -t WDARGS < "$WDARGV"
  ( cd $HOME/dsv41/m3 && setsid nohup "${WDARGS[@]}" >/dev/null 2>&1 </dev/null & )
  sleep 6
  ps -eo pid,ppid,comm,args | awk '$3=="python3" && index($0,"sidecar_watchdog.py"){print "watchdog:", $1, "ppid", $2}' | head -2
  grep -q '"trip": null' $HOME/dsv41/m3/logs/sidecar_watchdog.status.json && log "watchdog armed, trip null" \
    || log "WARN: watchdog status file does not read trip null"
else
  log "WARN: watchdog argv file missing ($WDARGV)"
fi
echo "- $(date '+%F %T %Z') lane restored to dsv41-vllm-1017-h2-REF pair by restore_after.sh" >> $HOME/dsv41/m3/LANE-OWNER.txt
log "restore done"