#!/usr/bin/env bash
# dsv41_pg61_lane_check.sh — silent-unless-wrong health check for the :30006 DSV4.1 lane on the GB300 ($GB300_SSH),
# Pg61 era (cut over 2026-10-29): Garner 8043a681 + our patch set via ~/dsv41/garner/launcher/launch-pg61.sh,
# containers ds41 + ds41-sidecar, fail-closed sidecar_watchdog_p3.py (ENFORCE, status ~/dsv41/garner/pg61-work/wdp3.*).
# Healthy = UP exactly "ds41 ds41-sidecar", BOUND=1, P3 alive, status age < 60 s, trip null, no TRIP in the log,
# BUILD=OK (frozen build matches BUILD-SHA256SUMS). MAINTENANCE flag (<= 3 h old) = planned window, silent.
# Prints NOTHING when healthy (cron no_agent). Previous lane check (h2-REF): dsv41_v27_lane_check.sh (kept).
out=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "${GB300_SSH:?set GB300_SSH=user@host}" '
M=$HOME/dsv41/m3/logs/MAINTENANCE; W=$HOME/dsv41/garner/pg61-work
if [ -e $M ]; then a=$(( $(date +%s) - $(stat -c %Y $M) )); [ $a -lt 10800 ] && { echo MAINT; exit 0; }; echo "MAINT_STALE=${a}s $(cat $M)"; fi
up=$(docker ps --format "{{.Names}}" | grep -E "^ds41|^dsv41-vllm-" | sort | tr "\n" " ")
bound=$(curl -s -m 5 http://127.0.0.1:30006/v1/models | grep -c dsv41-flash-uva)
p3=dead; ps -eo comm=,args= | awk "\$1==\"python3\" && index(\$0,\"sidecar_watchdog_p3.py\") && index(\$0,\"--mode enforce\")" | grep -q . && p3=alive
st=$(python3 -c "import json,time;d=json.load(open(\"$W/wdp3.status.json\"));print(d.get(\"trip\"), d.get(\"server_up\"), round(time.time()-d[\"t\"]))" 2>/dev/null)
trip=$(grep -E "TRIP|fail-closed" $W/wdp3.log 2>/dev/null | tail -2 | tr "\n" ";")
build=BAD; (cd $W && sha256sum -c --quiet BUILD-SHA256SUMS >/dev/null 2>&1) && build=OK
echo "UP=$up|BOUND=$bound|P3=$p3|ST=$st|BUILD=$build|TRIP=$trip"
' 2>&1)
[ "$out" = MAINT ] && exit 0
case "$out" in
  "UP=ds41 ds41-sidecar |BOUND=1|P3=alive|ST=None True "*"|BUILD=OK|TRIP=")
     age=$(echo "$out" | sed -E 's/.*ST=None True ([0-9]+).*/\1/'); [ "${age:-999}" -lt 60 ] && exit 0 ;;
esac
echo "⚠️ Pg61 lane check ($(date '+%-I:%M%p %Z')): something is off on the GB300 :30006 lane."
echo "$out" | tr '|' '\n' | sed 's/^/  /'
echo "Healthy = UP exactly ds41 + ds41-sidecar, BOUND=1, P3=alive, ST=None True <age<60s>, BUILD=OK, TRIP empty."
echo "The p3 watchdog is FAIL-STOP: on a trip ds41 is killed and the lane stays dark. Read ~/dsv41/garner/pg61-work/wdp3.log, docker logs ds41-sidecar, journalctl -k | grep Xid."
echo "Rollback to h2-REF (one command): ssh $GB300_SSH 'bash ~/dsv41/garner/promote/rollback-pg61-to-h2ref.sh' (try --dry-run first)."
