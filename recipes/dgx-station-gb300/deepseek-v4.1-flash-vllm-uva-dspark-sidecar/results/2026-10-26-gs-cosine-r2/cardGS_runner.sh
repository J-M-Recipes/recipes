#!/usr/bin/env bash
# cardGS_runner.sh — Card GS: v25 static reference check boots for cosine-gate R2, then re-judge every archived boot.
# James 2026-10-24 08:0x "I agree": new gate R2 registered (judge_r2.py) BEFORE re-judging; bar from v25 static's own noise;
# two more static check boots AFTER the 48 h v25m soak ends (2026-10-26 07:12:57 CDT); publish whatever it says.
# Sequence: wait for soak end -> record soak result -> MAINTENANCE flag (Mac monitor stays quiet <= 90 min) ->
#   K1, K2 = check-mode boots of rowmap-static-v1 (same launch as GV/GV2 K) -> restore v25m v1 production + fail-stop
#   watchdog -> clear flag -> judge_r2.py over refs {SC0 armA, K1, K2} and cands {G285, GMIX, GV v25m, GV2 v25m2}.
set -uo pipefail
DS=$HOME/dsv41; D=$DS/m3; SC=$D/sc2; GV=$D/gv; R=$GV/results-cardGS
U=http://127.0.0.1:30006; M=dsv41-flash-uva
NAME1=dsv41-vllm-1017-v25m-MIXMAP; MAP1=rowmap-mixed-ar-v1.json; MAPS=rowmap-static-v1.json
H=$D/el8/deepseek-v4.1-flash/m3/hook; LAUNCH=$D/launch_sidecar.sh; WD=$D/sidecar_watchdog.py; SHM=/dev/shm/vllm_peer_tier2
GATE=$D/sc3/sc0_gate.py; GPROMPTS=$D/sc3/sc-gate-prompts-v1.jsonl; LEDGER=$DS/results/ledger.md
SOAK_END_EPOCH=$(date -d '2026-10-26 07:13:30 CDT' +%s)
FF=' ERROR [0-9]{2}-[0-9]{2} |Traceback \(most recent call last\)|ValueError:|RuntimeError:|OutOfMemoryError|EngineCore failed'
SC2R=$SC/results-cardSC2; LB=$DS/1017/lilbench/app; BENCH=$DS/bench; PROMPTS=$BENCH/perclass_prompts.jsonl
[[ -e $R/runner.log ]] && { echo "ABORT: $R/runner.log exists" >&2; exit 5; }
mkdir -p "$R"
exec >> "$R/runner.log" 2>&1
source <(sed -n '/^log(){/,/^# ================= preflight/p' "$SC/cardSC2_runner.sh" | sed '$d')
other_runners(){ ps -eo pid,args | grep -E '^ *[0-9]+ bash .*(runner\.sh|perclass_run\.sh)' | grep -vE "cardGS_runner.sh|cardSC2_runner.sh" | grep -v grep | cut -c1-120 | tr '\n' '|'; }
arm_wd(){ setsid nohup python3 "$WD" --server "$1" --mode enforce --interval 1 --stall 2 \
    --status "$D/logs/sidecar_watchdog.status.json" --log "$D/logs/sidecar_watchdog.log" > /dev/null 2>&1 < /dev/null &
  echo $! > "$D/logs/sidecar_watchdog.pid"; sleep 3
  kill -0 "$(cat "$D/logs/sidecar_watchdog.pid")" && log "fail-stop watchdog armed pid $(cat "$D/logs/sidecar_watchdog.pid") on $1" || log "WARN: watchdog did not start"; }

log "===== CARD GS ARMED: waiting for soak end ($(date -d @$SOAK_END_EPOCH '+%F %T %Z')) ====="
while (( $(date +%s) < SOAK_END_EPOCH )); do sleep 60; done
log "===== CARD GS START ====="
# ---- soak result (48 h v25m, 2026-10-24 07:12:57 -> 2026-10-26 07:12:57)
python3 - "$D" > "$R/SOAK-RESULT.json" <<'PY'
import json,sys,re
D=sys.argv[1]
st=json.load(open(f"{D}/logs/sidecar_watchdog.status.json"))
log=open(f"{D}/logs/sidecar_watchdog.log",errors="replace").read().splitlines()
since=[i for i,l in enumerate(log) if "2026-10-24 07:12:54" in l and "start v2" in l]
tail=log[since[-1]:] if since else log
trips=[l for l in tail if re.search(r" TRIP | stopped .*fail closed",l)]
print(json.dumps({"watchdog_trip":st.get("trip"),"server_up":st.get("server_up"),"timeouts_words2":(st.get("words") or [None]*7)[2],
  "waits":(st.get("words") or [None]*7)[6],"trip_lines_since_soak_start":trips}))
PY
up=$(docker ps --format '{{.Names}}' | grep -c "^$NAME1$"); bound=$(curl -s -m 5 $U/v1/models | grep -c $M)
log "SOAK RESULT: lane_up=$up bound=$bound $(cat "$R/SOAK-RESULT.json")"
docker logs --since 2026-10-24T12:12:00Z "$NAME1" > "$R/soak-serverlog.log" 2>&1; log "SOAK ERRCOUNT $(grep -cE "$FF" "$R/soak-serverlog.log")"
o=$(other_runners); [[ -n $o ]] && { log "REFUSE: another runner is live: $o — lane left as is"; exit 4; }
g=$(gpu_foreign); [[ -n ${g// /} ]] && { log "REFUSE foreign GPU container:$g — lane left as is"; exit 4; }
for f in "$H/$MAPS" "$H/$MAP1" "$GATE" "$GPROMPTS" "$GV/judge_r2.py"; do [[ -s $f ]] || { log "ABORT: missing $f"; exit 5; }; done
{ for f in "$0" "$GV/judge_r2.py" "$GATE" "$GPROMPTS" "$H/$MAPS" "$H/mega_peer_hook.py"; do echo "sha1 $(sha1sum "$f" | cut -c1-12) $f"; done; } > "$R/instruments.txt"

touch "$D/logs/MAINTENANCE"; echo "cardGS $(date '+%F %T %Z')" > "$D/logs/MAINTENANCE"
WP=$(cat "$D/logs/sidecar_watchdog.pid" 2>/dev/null); [[ -n $WP ]] && kill "$WP" 2>/dev/null && log "fail-stop watchdog $WP stopped (soak complete; GS check boots)"
echo "$(date '+%F %T %Z') watchdog stopped by cardGS after the 48 h soak (re-armed after restore)" >> "$D/logs/sidecar_watchdog.log"
log "LANE DARK begins (GS static check boots)"
for k in K1 K2; do
  KN=dsv41-vllm-GS-$k-EXP
  stop_all; drop_caches
  docker rm -f "$KN" "$KN-rtx" >/dev/null 2>&1 || true
  t0=$(date +%s); since=$(date -u +%FT%TZ)
  if NAME=$KN ROWMAP=$MAPS bash "$LAUNCH" sidecar >> "$R/sidecar-$k.txt" 2>&1 && \
     REPLAY_OVERLAY=1 EXTRA="" MEGA_PEER_CHECK=100000 MEGA_COLD_TRT=1 NAME=$KN ROWMAP=$MAPS bash "$LAUNCH" server >> "$R/sidecar-$k.txt" 2>&1 && \
     wait_bind "$KN" 45 "$since" "" "$KN-rtx"; then
    log "boot->ready $k $(( $(date +%s)-t0 ))s"
    python3 "$GATE" sendjsonl $U/v1 $M "$GPROMPTS" > "$R/gate-send-$k.log" 2>&1 || log "GATE $k: some prompts errored"
    sleep 5
    docker logs "$KN" > "$R/serverlog-$k.log" 2>&1; cp "$D/logs/peer_server2.log" "$R/peer_server2-$k.log" 2>/dev/null
    log "FACTS $k: rowmap40=$(grep -c "MEGA_PEER rowmap /w/$MAPS: 40 layers" "$R/serverlog-$k.log") check=$(grep -c 'peer mode 2, check 100000' "$R/serverlog-$k.log")"
    python3 "$GATE" judge "$R/serverlog-$k.log" "$R/peer_server2-$k.log" > "$R/gate-$k.json" 2>&1; log "GATE(R1) $k rc=$?"
  else
    log "$k BOOT FAILED: $(tail -3 "$R/sidecar-$k.txt" | tr '\n' '|')"; docker logs "$KN" > "$R/boot-$k-FAIL.log" 2>&1
  fi
  docker stop "$KN" "$KN-rtx" >/dev/null 2>&1; docker rename "$KN" "$KN-DONE" 2>/dev/null; docker rename "$KN-rtx" "$KN-DONE-rtx" 2>/dev/null
done
ROWMAP=$MAP1
if boot_side F "$NAME1" 1 "--kv-cache-memory 9126805504" 45; then
  log "LANE DARK ends (v25m v1 bound)"; health F "$NAME1"; arm_wd "$NAME1"
else log "RESTORE FAILED — lane dark, needs hands (RESTORE-v25.md)"; fi
rm -f "$D/logs/MAINTENANCE"
A=()
A+=("ref:v25-static-SC0=$D/logs/dsv41-vllm-SC0-armA-ours-EXP.log")
[[ -s $R/serverlog-K1.log ]] && A+=("ref:v25-static-GS-K1=$R/serverlog-K1.log,$R/peer_server2-K1.log")
[[ -s $R/serverlog-K2.log ]] && A+=("ref:v25-static-GS-K2=$R/serverlog-K2.log,$R/peer_server2-K2.log")
A+=("cand:285-99-static-h285-SC3b=$D/results-cardSC3b/serverlog-G285.log")
A+=("cand:mix-v1-RonanLabs-SC3=$D/results-cardSC3/serverlog-GMIX.log")
A+=("cand:v25m-GV=$GV/results-cardGV/serverlog-K.log,$GV/results-cardGV/peer_server2-K.log")
A+=("cand:v25m2-L39static-GV2=$GV/results-cardGV2/serverlog-K.log,$GV/results-cardGV2/peer_server2-K.log")
python3 "$GV/judge_r2.py" "$R" "${A[@]}" > "$R/judge.stdout" 2>&1; log "R2 judge rc=$?"
grep -E '^\| v25m-GV ' "$R/R2-VERDICT.md" | cut -c1-200 | while read -r l; do log "R2 v25m: $l"; done
log "R2 bar: $(head -1 "$R/R2-VERDICT.md")"
grep -hE ' (SOAK|GATE|FACTS|boot->ready|BOUND|FAILED|LANE|R2|HEALTH|fail-stop|WARN|REFUSE|ABORT) ' "$R/runner.log" > "$R/SUMMARY.txt"
{ echo; echo "## $(date '+%F %H:%M %Z') — Card GS: v25m 48 h soak result + cosine gate R2 re-judge"; grep 'SOAK RESULT\|SOAK ERRCOUNT' "$R/runner.log" | cut -c25-; cat "$R/R2-VERDICT.md"; echo "Receipts: $R"; } >> "$LEDGER"
log "===== CARD GS END (lane = $NAME1) ====="
