#!/usr/bin/env bash
# cardS0b_runner.sh — Card S0 supplement (same window, same axis): resolve the one anomaly in cardS0 —
#   S0 cold-64K TTFT 3.78 s vs CTL 2.20/2.18 and replay r1 301 vs 412/… (first-request effects?) — with N=3 cold probes + a
#   second run_window per arm. Arms: B-CTL1 = REF live (as left by cardS0) -> B-S0 = docker start dsv41-vllm-1017-S0-pinexact-EXP-<suffix>
#   (same container, same env) -> B-CTL2 = REF docker start, LEFT SERVING. No new axis; no bars change (informational for cold-prefill only).
# 2026-10-18 ~09:35 CDT, Milo subagent (claude-opus-5-5). Waits for cardS0_runner.sh to exit first.
set -uo pipefail
W=$HOME/dsv41/1017; R=$W/results-cardS0/supp; mkdir -p "$R"
U=http://127.0.0.1:30006; M=dsv41-flash-uva
REF=dsv41-vllm-1017-v22-ksched1to8-BOUND-REF
exec >> "$R/runner.log" 2>&1
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
drop_caches(){ sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory' 2>/dev/null || true; }
stop_all(){ local n; for n in $(docker ps --format '{{.Names}}' | grep -E '^dsv41-vllm-'); do log "stop-and-keep $n"; docker stop "$n" >/dev/null; done; sleep 5; }
wait_bind(){ local n=$1 i
  for ((i=0;i<450;i++)); do
    curl -s -m 2 $U/v1/models 2>/dev/null | grep -q "$M" && { log "BOUND $n after $((i*10))s"; return 0; }
    docker ps --format '{{.Names}}' | grep -qx "$n" || { log "DIED $n"; return 1; }
    sleep 10
  done; log "TIMEOUT $n"; return 1; }
gpu_foreign(){ local c out=""; for c in $(docker ps --format '{{.Names}}' | grep -vE '^dsv41-vllm-'); do
    docker inspect "$c" --format '{{json .HostConfig.DeviceRequests}}' 2>/dev/null | grep -qi gpu && out="$out $c"; done; echo "$out"; }
arm(){ local tag=$1
  BASE_URL=$U/v1 MODEL=$M API_KEY=none SIZES=65536 N=3 THINKING=0 timeout 900 python3 $HOME/dsv41/cold_prefill_probe.py > "$R/cold64k-$tag.txt" 2>&1 || log "cold64k $tag rc!=0"
  log "COLD64K $tag: $(grep PREFILL "$R/cold64k-$tag.txt" | tail -1)"
  bash $HOME/dsv41/run_window_T1T2.sh "S0b-$tag" > "$R/nohup-$tag.log" 2>&1
  mv $HOME/dsv41/overnight-T1T2/*"S0b-$tag"[-.]* "$R"/ 2>/dev/null; mv $HOME/dsv41/overnight-T1T2/*"S0b-$tag" "$R"/ 2>/dev/null
  log "WINDOW $tag: $(grep -h 'conc=' "$R/window-S0b-$tag.log" 2>/dev/null | grep -E 'r2\]' | tr '\n' ';' | cut -c1-300)"
  log "REPLAY $tag: $(grep -h 'REPLAY' "$R/window-S0b-$tag.log" 2>/dev/null | cut -c1-120 | tr '\n' ';')"
  BASE_URL=$U/v1 MODEL=$M API_KEY=none SIZES=65536 N=2 THINKING=0 timeout 900 python3 $HOME/dsv41/cold_prefill_probe.py > "$R/cold64k-post-$tag.txt" 2>&1 || true
  log "COLD64K-POST $tag: $(grep PREFILL "$R/cold64k-post-$tag.txt" | tail -1)"; }
log "===== CARD S0b START ====="
for ((i=0;i<120;i++)); do ps -eo args | grep -E '^bash .*cardS0_runner\.sh' >/dev/null || break; sleep 30; done
ps -eo args | grep -E '^bash .*cardS0_runner\.sh' >/dev/null && { log "ABORT: cardS0_runner still running after 60 min"; exit 3; }
g=$(gpu_foreign); [[ -n ${g// /} ]] && { log "REFUSE foreign GPU container:$g"; exit 4; }
CAND=$(docker ps -a --format '{{.Names}}' | grep -E '^dsv41-vllm-1017-S0-pinexact-EXP-' | grep -v STALE | head -1)
[[ -n $CAND ]] || { log "ABORT: no suffixed S0 container"; exit 5; }
curl -s -m 3 $U/v1/models | grep -q $M || { log "ABORT: REF not bound"; exit 5; }
log "CAND=$CAND"
arm CTL1
stop_all; drop_caches
t0=$(date +%s); docker start "$CAND" >/dev/null
if wait_bind "$CAND"; then log "boot->ready S0 $(( $(date +%s) - t0 ))s"; arm S0; else log "S0 restart failed to bind"; fi
stop_all; drop_caches
t0=$(date +%s); docker start "$REF" >/dev/null
if wait_bind "$REF"; then log "boot->ready CTL2 $(( $(date +%s) - t0 ))s"; arm CTL2; else log "CTL2 REF FAILED TO BIND — lane dark, needs hands"; fi
log "===== CARD S0b END (lane = $REF; bound=$(curl -s -m 3 $U/v1/models | grep -c $M)) ====="
