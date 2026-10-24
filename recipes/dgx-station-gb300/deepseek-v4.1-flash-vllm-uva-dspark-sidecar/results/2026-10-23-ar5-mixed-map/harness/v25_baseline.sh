#!/usr/bin/env bash
# v25_baseline.sh — AR4: run the AR workloads on production v25 (sidecar, fixed map) as-is.
# Restores v25 per RESTORE-v25.md, waits for readiness, runs mixed C1/C4 + shift protocol.
set -uo pipefail
AR=$HOME/dsv41/ar
RUN=$AR/v25
S=$RUN/STATUS
SRV=dsv41-vllm-1017-v25-sidecar-BOUND-REF
MIX_S=${MIX_S:-180}
PHASE_S=${PHASE_S:-180}
mkdir -p "$RUN"
log() { echo "$(date '+%F %T %Z') $*" | tee -a "$S"; }
die() { log "STOP: $*"; exit 1; }

log "AR4 start: restore v25 sidecar pair"
docker ps --format '{{.Names}}' | grep -q v26ar && die "an AR container is still running"
SINCE=$(date -u +%Y-%m-%dT%H:%M:%SZ)
docker start "$SRV-rtx" >/dev/null || die "sidecar start failed"
ok=0
for i in $(seq 1 240); do
  docker logs --since "$SINCE" "$SRV-rtx" 2>&1 | grep -qx "serving" && { ok=1; break; }
  docker ps -q -f "name=^${SRV}-rtx$" | grep -q . || die "sidecar exited"
  sleep 5
done
[ $ok = 1 ] || die "sidecar never printed 'serving' (20 min)"
log "sidecar up: $(docker logs --tail 2 "$SRV-rtx" 2>&1 | tail -1 | cut -c1-160)"
sleep 20
docker start "$SRV" >/dev/null || die "server start failed"
T0=$(date +%s)
while true; do
  curl -sf -m 5 localhost:30006/v1/models 2>/dev/null | grep -q dsv41-flash-uva && break
  docker ps -q -f "name=^${SRV}$" | grep -q . || { docker logs --tail 120 "$SRV" > "$RUN/boot-fail.log" 2>&1; die "server exited during boot"; }
  [ $(( $(date +%s) - T0 )) -gt 2400 ] && die "boot timeout"
  sleep 10
done
log "v25 ready after $(( $(date +%s) - T0 )) s"
python3 - <<'PY' > /dev/null 2>&1
import json, urllib.request
b = {"model": "dsv41-flash-uva", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16, "temperature": 0}
urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:30006/v1/chat/completions",
    data=json.dumps(b).encode(), headers={"Content-Type": "application/json"}), timeout=300).read()
PY
for c in 1 4; do
  log "mixed v25 C$c ${MIX_S}s"
  python3 "$AR/ar_mixed.py" v25-c$c "$RUN" $c "$MIX_S" > "$RUN/mixed-v25-c$c.log" 2>&1 || die "mixed c$c"
  log "  $(tail -1 "$RUN/mixed-v25-c$c.log" | cut -c1-300)"
done
log "shift protocol v25"
python3 "$AR/ar_shift.py" v25 "$RUN" "$PHASE_S" > "$RUN/shift-v25.log" 2>&1 || die "shift"
log "  $(tail -1 "$RUN/shift-v25.log")"
log "AR4 done; v25 left serving on :30006 (watchdog NOT re-armed by this script)"
