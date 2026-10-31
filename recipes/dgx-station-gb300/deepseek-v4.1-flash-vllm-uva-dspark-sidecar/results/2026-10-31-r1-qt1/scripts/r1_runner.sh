#!/usr/bin/env bash
# r1_runner.sh — CARD R1: one dark window on :30006 (DSV4.1 sidecar recipe, Pg61 release gates).
#   P0 preflight (lane up) · P2 fault-inject the deployed p3 watchdog (DARK starts) · P3 restore boot + bring-up tap
#   · P4 rollback drill Pg61 -> h2-REF (+ QT1 h2-REF reference arm if staged) · P5 cutover back · P6 close + offline
#   cosine reference on the P3 capture (lane serving again).
# P1 (live cosine tap) ran BEFORE this card on the serving lane: ~/dsv41/garner/r1/live-tap2/.
# Bars + aborts: R1-PREREG.md next to this file. Derived from W3's r1_runner.DRAFT.sh; Milo's changes 2026-10-31:
#   isolate (non-loopback :30006 REJECT) for the dark phases, P1 moved out of the window, fixed tap flags, fault
#   script tests the deployed watchdog, QT1 h2-REF hook in P4, explicit isolate_off on every exit path.
# Launch ON the box:  cd ~/dsv41/garner/r1 && GO=1 setsid nohup bash r1_runner.sh </dev/null >/dev/null 2>&1 &
set -uo pipefail
GR=$HOME/dsv41/garner; M3=$HOME/dsv41/m3; LW=$GR/pg61-work; HERE=$GR/r1
LAUNCH=$GR/launcher/launch-pg61.sh; RB=$GR/promote/rollback-pg61-to-h2ref.sh; CU=$GR/promote/cutover-h2ref-to-pg61.sh
SOAK=$HERE/soak_check_local.sh
SRV=ds41; SIDE=ds41-sidecar; H2=dsv41-vllm-1017-h2-REF
MODEL=/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170
R=${R:-$HERE/results-R1-$(date +%Y%m%d-%H%M)}; ABORT=$HERE/ABORT
KV_EXPECT='11,696,582 tokens'; P3_SHA=ba8373e9bdd9a0225620142ea93330a99caa597826910399ab4c741ee8eabc9f
[[ ${GO:-0} == 1 ]] || { echo "REFUSE: set GO=1 (James's go)"; exit 9; }
[[ -e $R ]] && { echo "REFUSE: $R exists"; exit 5; }
mkdir -p "$R"/{P0,P2,P3,P4,P5,P6} || exit 5
exec 3>> "$R/runner.log"; exec >> "$R/runner.log" 2>&1
log(){ printf '%s [R1] %s\n' "$(date '+%F %T %Z')" "$*" >&3; }
own(){ echo "- $(date '+%F %T %Z') CARD R1 (release gates): $*" >> "$M3/LANE-OWNER.txt"; }
abort_check(){ [[ -e $ABORT ]] && { log "ABORT file present"; return 1; }; return 0; }
running(){ [[ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" == true ]]; }
bound(){ curl -sf -m 3 http://127.0.0.1:30006/v1/models 2>/dev/null | grep -q dsv41-flash-uva; }
shm_state(){ local o=""; for f in /dev/shm/vllm_peer_tier2 /dev/shm/vllm_peer_tier3; do [[ -e $f ]] && o="$o${f##*_} " || o="$o- "; done; echo "$o"; }
kvline(){ docker logs "$SRV" 2>&1 | grep -m1 'GPU KV cache size' || echo NO-KV-LINE; }
pids_of(){ ps -eo pid=,ppid=,comm=,args= | awk -v k="$1" '$3=="python3" && index($0,k){print $1, $2}'; }
isolate_on(){ sudo -n iptables -C INPUT -p tcp --dport 30006 ! -i lo -j REJECT --reject-with tcp-reset 2>/dev/null \
  || sudo -n iptables -I INPUT -p tcp --dport 30006 ! -i lo -j REJECT --reject-with tcp-reset; }
isolate_off(){ while sudo -n iptables -D INPUT -p tcp --dport 30006 ! -i lo -j REJECT --reject-with tcp-reset 2>/dev/null; do :; done; }
restarts(){ for c in $SRV $SIDE $H2 $H2-rtx; do printf '%s %s\n' "$c" "$(docker inspect -f '{{.HostConfig.RestartPolicy.Name}} {{.State.Status}} {{.RestartCount}}' "$c" 2>&1)"; done > "$1"; }
shm_start(){ ( while :; do printf '%s %s\n' "$(date +%s.%N)" "$(shm_state)" >> "$R/shm.tsv"; sleep 0.1; done ) & echo $! > "$R/shm.pid"; }
shm_stop(){ [[ -f $R/shm.pid ]] && kill "$(cat "$R/shm.pid")" 2>/dev/null; rm -f "$R/shm.pid"; }
hold(){ local i; for i in $(seq 1 "${1:-60}"); do abort_check || return 1; sleep 1; done; }
trap 'isolate_off; shm_stop' EXIT

P0(){
  log "P0 preflight"
  bash "$RB" --dry-run > "$R/P0/rb-dry.log" 2>&1; grep -q 'preconditions OK' "$R/P0/rb-dry.log" || { log "REFUSE: rollback dry-run"; return 1; }
  sudo -n true || { log "REFUSE: sudo -n"; return 1; }
  sha256sum "$LW/sidecar_watchdog_p3.py" > "$R/P0/wd3.sha256"; grep -q $P3_SHA "$R/P0/wd3.sha256" || { log "REFUSE: p3 file is not ba8373e9"; return 1; }
  ( cd "$LW" && sha256sum -c --quiet BUILD-SHA256SUMS ) || { log "REFUSE: BUILD=BAD"; return 1; }
  bound || { log "REFUSE: lane not bound"; return 1; }
  grep -q '"trip": null' "$LW/wdp3.status.json" || { log "REFUSE: p3 watchdog not trip null"; return 1; }
  pgrep -f "_runner.sh" | grep -vx "$$" >/dev/null && { log "REFUSE: another card runner is live"; return 1; }
  docker ps --format '{{.Names}}' | grep -vxE "$SRV|$SIDE" | grep . && { log "REFUSE: foreign container running"; return 1; }
  for c in "$H2" "$H2-rtx"; do docker inspect "$c" >/dev/null 2>&1 || { log "REFUSE: $c missing"; return 1; }; done
  docker inspect -f '{{json .Args}}' $SRV > "$R/P0/args.json"; docker inspect -f '{{json .Config.Env}}' $SRV > "$R/P0/env.json"
  restarts "$R/P0/restart.txt"; kvline > "$R/P0/kv.txt"; free -g > "$R/P0/free.txt"
  curl -s -m 10 http://127.0.0.1:30006/metrics > "$R/P0/metrics.txt"
  log "P0 OK: $(cat "$R/P0/kv.txt" | grep -o 'GPU KV cache size.*') | shm $(shm_state)"
}

P2(){
  log "P2 fault injection on the deployed p3 watchdog — DARK WINDOW STARTS"
  own "taking :30006 dark (R1: watchdog fault-injection, restore, rollback drill to h2-REF, cutover back; ~60 min)"
  echo "r1 dark start $(date '+%F %T %Z')" > "$M3/logs/MAINTENANCE"
  isolate_on || { log "REFUSE: isolate rule failed to insert — window not started"; rm -f "$M3/logs/MAINTENANCE"; return 1; }
  log "ISOLATE on (loopback only)"
  bash "$HERE/wd3_fault.sh" "$R/P2"; local rc=$?; log "P2 fault rc=$rc ($(grep -o '"pass": [a-z]*' "$R/P2/verdict.json"))"
  WORK=$LW DS41_HOME=$GR/home bash "$LAUNCH" down > "$R/P2/down.log" 2>&1; log "P2 down rc=$? shm=$(shm_state)"
  restarts "$R/P2/restart.txt"
  return 0   # a FAIL verdict is a result, not a reason to leave the lane dark; P3 restores either way
}

P3(){
  log "P3 restore boot #1 (launch-pg61.sh up)"
  local t0=$(date +%s)
  WORK=$LW MODEL_DIR=$MODEL DS41_HOME=$GR/home bash "$LAUNCH" up > "$R/P3/up.log" 2>&1; local rc=$?
  log "P3 up rc=$rc bind $(( $(date +%s) - t0 )) s"; [[ $rc == 0 ]] || return 1
  kvline > "$R/P3/kv.txt"; if grep -q "$KV_EXPECT" "$R/P3/kv.txt"; then echo "A_KV PASS" >> "$R/gates.txt"; else echo "A_KV FAIL $(cat "$R/P3/kv.txt")" >> "$R/gates.txt"; log "P3 KV MISMATCH (gate FAIL): $(cat "$R/P3/kv.txt")"; fi
  pids_of sidecar_watchdog_p3.py > "$R/P3/wd.txt"; restarts "$R/P3/restart.txt"
  hold 30 || return 1
  python3 "$HERE/peer3_cos_tap.py" --shm /dev/shm/vllm_peer_tier3 --layers all --samples-per-layer 10 --drive 4 \
    --max-seconds 420 --out "$R/P3/tap" --overhead "$R/P3/overhead.csv" > "$R/P3/tap.log" 2>&1
  log "P3 bring-up tap rc=$? ($(tail -1 "$R/P3/tap.log"))"
  bound || { log "P3 not bound after tap"; return 1; }
}

P4(){
  log "P4 rollback drill Pg61 -> h2-REF"
  local t0=$(date +%s)
  PS4='+$(date +%s.%N) ' bash -x "$RB" > "$R/P4/trace.log" 2>&1; local rc=$?
  log "P4 rollback rc=$rc in $(( $(date +%s) - t0 )) s"; [[ $rc == 0 ]] || return 1
  # the rollback clears MAINTENANCE on success; keep the lane flagged until the cutover back (H9/H11)
  echo "r1 drill on h2-REF $(date '+%F %T %Z')" > "$M3/logs/MAINTENANCE"
  pids_of sidecar_watchdog.py > "$R/P4/wd.txt"; log "P4 v2 watchdog: $(tr '\n' ' ' < "$R/P4/wd.txt")"
  bash "$SOAK" "$H2" > "$R/P4/soak.txt" 2>&1; log "P4 soak_check rc=$?"
  docker logs "$H2" 2>&1 | grep -m1 'GPU KV cache size' > "$R/P4/kv.txt"; restarts "$R/P4/restart.txt"
  if [[ -f $HERE/qt1/qt1_arm.sh ]]; then
    log "P4 QT1 P7 h2-REF captures (2 x ~15 s; cap 300 s)"
    timeout 300 bash "$HERE/qt1/qt1_arm.sh" h2ref "$R/P4/qt1" > "$R/P4/qt1.log" 2>&1; log "P4 QT1 rc=$?"
  fi
  hold 30 || return 1
  bound || { log "P4 h2-REF not bound at end"; return 1; }
  log "P4 done: bound, watchdog $(python3 -c 'import json;print(json.load(open("'"$M3"'/logs/sidecar_watchdog.status.json")).get("trip"))')"
}

P5(){
  log "P5 cutover back to Pg61"
  local t0=$(date +%s)
  PS4='+$(date +%s.%N) ' bash -x "$CU" > "$R/P5/trace.log" 2>&1; local rc=$?
  log "P5 cutover rc=$rc in $(( $(date +%s) - t0 )) s"
  ( cd "$LW" && sha256sum -c --quiet BUILD-SHA256SUMS ) && log "P5 BUILD=OK" || log "P5 BUILD=BAD"
  WORK=$LW DS41_HOME=$GR/home bash "$LAUNCH" verify > "$R/P5/verify.log" 2>&1; log "P5 verify rc=$?"
  kvline > "$R/P5/kv.txt"; if grep -q "$KV_EXPECT" "$R/P5/kv.txt"; then echo "A_KV_P5 PASS" >> "$R/gates.txt"; else echo "A_KV_P5 FAIL" >> "$R/gates.txt"; log "P5 KV MISMATCH (gate FAIL)"; fi
  cp "$LW/soak-start.json" "$R/P5/" 2>/dev/null; restarts "$R/P5/restart.txt"
  [[ $rc == 0 ]] && bound
}

recover(){
  log "RECOVER after rc=$1"
  if running "$H2"; then bash "$CU" >> "$R/recover.log" 2>&1
  elif ! running "$SRV"; then WORK=$LW DS41_HOME=$GR/home bash "$LAUNCH" down >> "$R/recover.log" 2>&1
       WORK=$LW MODEL_DIR=$MODEL DS41_HOME=$GR/home bash "$LAUNCH" up >> "$R/recover.log" 2>&1; fi
  if bound; then rm -f "$M3/logs/MAINTENANCE"; log "RECOVER ok (lane serving)"
  else log "RECOVER FAILED — lane dark, MAINTENANCE kept, needs hands"; own "RECOVER FAILED — lane DARK, needs hands; $R"; fi
}

P6(){
  isolate_off; log "ISOLATE off"
  [[ -e $M3/logs/MAINTENANCE ]] && log "P6 WARN: MAINTENANCE present" || log "P6 MAINTENANCE clear"
  curl -s -m 10 http://127.0.0.1:30006/metrics > "$R/P6/metrics.txt"
  own "R1 END (lane = $(running $SRV && echo Pg61 || echo DARK)); receipts $R; soak clock $(cat "$LW/soak-start.json" 2>/dev/null)"
  log "CARD R1 END (lane = $(running $SRV && echo Pg61 || echo DARK))"
  # offline fp64 reference on the bring-up capture, lane serving again (CPU container, no network, nice)
  if [[ -s $R/P3/tap.jsonl ]]; then
    cp "$HERE/peer3_cos_ref.py" "$R/P3/"
    nice -n 10 docker run --rm --name r1-cosref --network none --cpus 16 -v $MODEL:/model:ro -v $LW/tree/fastpath:/fp:ro \
      -v "$R/P3":/cap --entrypoint python3 vllm/vllm-openai:v0.31.0 /cap/peer3_cos_ref.py --cap /cap/tap --model /model \
      --rowmap /fp/hook/rowmap-mix-v1-h255.json --ref /fp/peer3/tests --out /cap/cos.json --max-per-layer 10 \
      > "$R/P3/cosref.log" 2>&1; log "P6 bring-up cosine ref rc=$? $(grep -o '"C[123]_[a-z_]*": \[[^]]*' "$R/P3/cosref.log" | tr '\n' ' ')"
  fi
}

main(){
  log "START R=$R"
  local rc=0
  P0 || { log "PREFLIGHT REFUSED — nothing touched"; exit 1; }
  shm_start   # after P0 (advisor consult-3 Fix A: its subshell matches the runner pgrep)
  abort_check && { P2 || rc=2; } || rc=2
  [[ $rc == 0 ]] && { abort_check && { P3 || rc=3; } || rc=3; }
  [[ $rc == 0 ]] && { abort_check && { P4 || rc=4; } || rc=4; }
  [[ $rc == 0 ]] && { P5 || rc=5; }
  [[ $rc != 0 ]] && recover "$rc"
  P6; log "END rc=$rc"; exit $rc
}
main "$@"
