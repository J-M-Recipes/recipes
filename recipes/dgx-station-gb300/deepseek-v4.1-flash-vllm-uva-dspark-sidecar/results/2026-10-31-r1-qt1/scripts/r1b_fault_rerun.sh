#!/usr/bin/env bash
# r1b_fault_rerun.sh — CARD R1 phase P2b: re-run of the watchdog fault injection with amendment A4 (wait for the
# watchdog to exit before judging). Runs right after R1 closes, lane = Pg61 serving. Kills the lane, restores it.
# Launch: cd ~/dsv41/garner/r1 && GO=1 setsid nohup bash r1b_fault_rerun.sh </dev/null >/dev/null 2>&1 &
set -uo pipefail
GR=$HOME/dsv41/garner; M3=$HOME/dsv41/m3; LW=$GR/pg61-work; HERE=$GR/r1; LAUNCH=$GR/launcher/launch-pg61.sh
MODEL=/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170
R=$HERE/results-R1b-$(date +%Y%m%d-%H%M); KV_EXPECT='11,696,582 tokens'
[[ ${GO:-0} == 1 ]] || { echo "REFUSE: GO=1"; exit 9; }
mkdir -p "$R" || exit 5; exec 3>> "$R/runner.log"; exec >> "$R/runner.log" 2>&1
log(){ printf '%s [R1b] %s\n' "$(date '+%F %T %Z')" "$*" >&3; }
own(){ echo "- $(date '+%F %T %Z') CARD R1b (fault re-run, amendment A4): $*" >> "$M3/LANE-OWNER.txt"; }
bound(){ curl -sf -m 3 http://127.0.0.1:30006/v1/models 2>/dev/null | grep -q dsv41-flash-uva; }
isolate_on(){ sudo -n iptables -C INPUT -p tcp --dport 30006 ! -i lo -j REJECT --reject-with tcp-reset 2>/dev/null \
  || sudo -n iptables -I INPUT -p tcp --dport 30006 ! -i lo -j REJECT --reject-with tcp-reset; }
isolate_off(){ while sudo -n iptables -D INPUT -p tcp --dport 30006 ! -i lo -j REJECT --reject-with tcp-reset 2>/dev/null; do :; done; }
trap 'isolate_off' EXIT
bound || { log "REFUSE: lane not bound"; exit 1; }
grep -q '"trip": null' "$LW/wdp3.status.json" || { log "REFUSE: watchdog not trip null"; exit 1; }
pgrep -f "r1_runner.sh" >/dev/null && { log "REFUSE: R1 runner still live"; exit 1; }
isolate_on || { log "REFUSE: isolate failed"; exit 1; }
echo "r1b fault re-run $(date '+%F %T %Z')" > "$M3/logs/MAINTENANCE"; own "taking :30006 dark (~9 min)"
bash "$HERE/wd3_fault.sh" "$R/P2"; log "P2b fault rc=$? $(grep -o '"pass": [a-z]*' "$R/P2/verdict.json")"
WORK=$LW DS41_HOME=$GR/home bash "$LAUNCH" down > "$R/down.log" 2>&1; log "down rc=$?"
t0=$(date +%s); WORK=$LW MODEL_DIR=$MODEL DS41_HOME=$GR/home bash "$LAUNCH" up > "$R/up.log" 2>&1; rc=$?
log "up rc=$rc bind $(( $(date +%s) - t0 )) s"
if [[ $rc != 0 ]]; then WORK=$LW DS41_HOME=$GR/home bash "$LAUNCH" down >> "$R/up.log" 2>&1
  WORK=$LW MODEL_DIR=$MODEL DS41_HOME=$GR/home bash "$LAUNCH" up >> "$R/up.log" 2>&1; log "retry up rc=$?"; fi
docker logs ds41 2>&1 | grep -m1 'GPU KV cache size' > "$R/kv.txt"; grep -q "$KV_EXPECT" "$R/kv.txt" && log "KV PASS" || log "KV FAIL $(cat "$R/kv.txt")"
WORK=$LW DS41_HOME=$GR/home bash "$LAUNCH" verify > "$R/verify.log" 2>&1; log "verify rc=$?"
( cd "$LW" && sha256sum -c --quiet BUILD-SHA256SUMS ) && log "BUILD=OK" || log "BUILD=BAD"
isolate_off; log "ISOLATE off"
if bound; then rm -f "$M3/logs/MAINTENANCE"; own "END: Pg61 serving; soak clock $(cat "$LW/soak-start.json"); $R"; log "END serving"
else own "END: lane DARK, needs hands; $R"; log "END DARK"; fi
