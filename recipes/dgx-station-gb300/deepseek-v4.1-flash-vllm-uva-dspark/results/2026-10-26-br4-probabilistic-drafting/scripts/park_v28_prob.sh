#!/usr/bin/env bash
# park_v28_prob.sh — create the GB300-only recipe's new reference container v28 = v26 Seeded Adaptive + probabilistic
# drafting (Card BR4: P PASS, 2026-10-26). Boot it once on :30006 (lane dark), verify facts + smoke, STOP-AND-KEEP it as
# dsv41-vllm-1017-v28-prob-BOUND-REF, then restore the serving v27 sidecar pair + fail-stop watchdog. ~20 min dark.
# The GB300-only lane is NOT the serving lane; this is a reference/park, not a routing change. 2026-10-26 Milo (James: "go").
set -uo pipefail
D=~/dsv41/m3; L=$D/logs/park-v28.log; mkdir -p $D/logs; exec >> "$L" 2>&1
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
own(){ echo "- $(date '+%F %T %Z') $*" >> "$D/LANE-OWNER.txt"; }
U=http://127.0.0.1:30006; M=dsv41-flash-uva
V27=dsv41-vllm-1017-v27-block-BOUND-REF; V27R=$V27-rtx
NEW=dsv41-vllm-1017-v28-prob-BOUND-REF
LAUNCH=$D/br3/launch_br3.sh; SPEC=',"draft_sample_method":"probabilistic"'
WDLOG=$D/logs/sidecar_watchdog.v27.log; WARGV=$D/logs/park-v28-wd-argv.txt
wd_pids(){ ps -eo pid=,comm=,args= | awk '$2=="python3" && index($0,"sidecar_watchdog.py") {print $1}' | tr '\n' ' '; }
up(){ docker ps --format '{{.Names}}' | grep -qx "$1"; }
bound(){ curl -s -m 2 $U/v1/models | grep -q $M; }
arm_wd(){ local -a A=(); local i p pp; mapfile -t A < "$WARGV"
  for ((i=0;i<${#A[@]};i++)); do [[ ${A[$i]} == --log && $((i+1)) -lt ${#A[@]} ]] && A[$((i+1))]=$WDLOG; done
  ( cd "$D" || exit 1; setsid nohup "${A[@]}" > /dev/null 2>&1 < /dev/null & ); sleep 4
  p=$(wd_pids | awk '{print $NF}'); pp=$(ps -o ppid= -p "${p:-0}" 2>/dev/null | tr -d ' '); [[ -n $p ]] && echo "$p" > "$D/logs/sidecar_watchdog.pid"
  log "watchdog armed pid ${p:-NONE} ppid ${pp:-?}: $(tr '\0' ' ' < /proc/${p:-0}/cmdline 2>/dev/null | cut -c1-200)"; }
restore(){ log "RESTORE v27 pair: $1"
  up "$NEW" && docker stop -t 60 "$NEW" >/dev/null
  sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory' 2>/dev/null || true; sleep 10
  local t0=$(date +%s) since=$(date -u +%FT%TZ) i
  docker start "$V27R" >/dev/null; for ((i=0;i<36;i++)); do docker logs --since "$since" "$V27R" 2>&1 | grep -qE 'serving|prepared 40 layers' && break; sleep 5; done
  log "sidecar $V27R running=$(docker inspect -f '{{.State.Running}}' $V27R) after $(( $(date +%s)-t0 ))s"
  docker start "$V27" >/dev/null
  for i in $(seq 1 270); do bound && break; up "$V27" || { log "RESTORE: $V27 died — NEEDS HANDS"; own "park v28: RESTORE FAILED, lane dark, needs hands"; rm -f $D/logs/MAINTENANCE; exit 9; }; sleep 10; done
  bound || { log "RESTORE: no bind in 45 min — NEEDS HANDS"; own "park v28: RESTORE did not bind, lane dark, needs hands"; rm -f $D/logs/MAINTENANCE; exit 9; }
  kv=$(docker logs --since "$since" "$V27" 2>&1 | tr '\r' '\n' | grep -m1 -oE 'GPU KV cache size: [0-9,]+ tokens'); log "LANE DARK ends: $V27 bound after $(( $(date +%s)-t0 ))s; $kv"
  arm_wd; rm -f $D/logs/MAINTENANCE; }

log "===== PARK v28-prob START ====="
[[ -z $(ps -eo args | grep -E 'bash .*runner\.sh' | grep -v grep) ]] || { log "ABORT: runner live"; exit 4; }
up "$V27" && up "$V27R" || { log "ABORT: v27 pair not up"; exit 5; }
docker ps -a --format '{{.Names}}' | grep -qx "$NEW" && { log "ABORT: $NEW exists"; exit 5; }
[[ "$(sha1sum $LAUNCH | cut -c1-8)" == 8066045c ]] || { log "ABORT: launch_br3.sh sha != 8066045c"; exit 5; }
WP=$(wd_pids | awk '{print $1}'); [[ -n $WP ]] || { log "ABORT: no live watchdog"; exit 5; }
tr '\0' '\n' < /proc/$WP/cmdline > "$WARGV"; grep -qx -- "$V27" "$WARGV" || { log "ABORT: watchdog argv not v27"; exit 5; }
own "CLAIM park v28-prob: booting $NEW (v26seed launch + draft_sample_method probabilistic, Card BR4 P PASS) once to verify, then stop-and-keep as the GB300-only reference; v27 pair stopped-kept meanwhile (~20 min dark)"
echo "park-v28 $(date '+%F %T %Z')" > $D/logs/MAINTENANCE
kill "$WP"; sleep 3; [[ -z $(wd_pids) ]] || { log "ABORT: watchdog alive"; rm -f $D/logs/MAINTENANCE; exit 6; }
docker stop -t 60 "$V27" >/dev/null; docker stop -t 30 "$V27R" >/dev/null; log "LANE DARK begins (v27 pair stopped-kept)"
sleep 15; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' '); log "GPU used after stop: $u MiB"
t0=$(date +%s); since=$(date -u +%FT%TZ)
NAME=$NEW RUN=~/dsv41/ar/soak ROWMAP=rowmap-mixed-ar-v1.json SPEC_EXTRA="$SPEC" bash $LAUNCH || { log "launch failed"; restore "launch failed"; exit 3; }
ok=0; for i in $(seq 1 150); do bound && { ok=1; break; }; up "$NEW" || break; sleep 10; done
if [[ $ok != 1 ]]; then docker logs "$NEW" > $D/logs/park-v28-boot-FAIL.log 2>&1; docker rename "$NEW" "$NEW-PARK-FAIL"; restore "v28 did not bind"; own "park v28 FAILED (no bind); v27 restored"; exit 3; fi
docker logs "$NEW" > $D/logs/park-v28-boot.log 2>&1
A=$(docker inspect -f '{{json .Args}}' "$NEW"); echo "$A" > $D/logs/park-v28-args.json
kv=$(tr '\r' '\n' < $D/logs/park-v28-boot.log | grep -m1 -oE 'GPU KV cache size: [0-9,]+ tokens')
log "BOUND $NEW after $(( $(date +%s)-t0 ))s | $kv | prob_in_args=$(grep -c draft_sample_method <<<"$A") block_in_args=$(grep -c rejection_sample_method <<<"$A") | rehome=$(tr '\r' '\n' < $D/logs/park-v28-boot.log | grep -c 'rehome done layers=40') GO=$(tr '\r' '\n' < $D/logs/park-v28-boot.log | grep -c 'ADAPT GO seen')"
r1=$(curl -s -m 120 $U/v1/chat/completions -H 'content-type: application/json' -d '{"model":"dsv41-flash-uva","messages":[{"role":"user","content":"Reply with exactly: v28 ok"}],"max_tokens":128,"reasoning_effort":"low"}')
r2=$(curl -s -m 120 $U/v1/chat/completions -H 'content-type: application/json' -d '{"model":"dsv41-flash-uva","messages":[{"role":"user","content":"What time is it on the box? Use the tool."}],"tools":[{"type":"function","function":{"name":"shell","description":"run a shell command","parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}}}],"max_tokens":256,"chat_template_kwargs":{"thinking":false}}')
log "SMOKE low: $(python3 -c 'import json,sys;d=json.loads(sys.argv[1]);m=d["choices"][0]["message"];print(repr((m.get("content") or "")[:40]), d["usage"].get("completion_tokens"))' "$r1" 2>&1 | cut -c1-120) | tool: $(python3 -c 'import json,sys;d=json.loads(sys.argv[1]);print([c["function"]["name"] for c in (d["choices"][0]["message"].get("tool_calls") or [])])' "$r2" 2>&1 | cut -c1-80)"
verdict=OK; [[ $kv == *"2,481,694"* ]] || verdict="KV-MISMATCH"; grep -q draft_sample_method <<<"$A" || verdict="NO-PROB-KEY"; grep -q '"finish_reason"' <<<"$r1" || verdict="SMOKE-FAIL"
log "PARK facts verdict: $verdict"
docker stop -t 60 "$NEW" >/dev/null; log "stopped-and-kept $NEW"
[[ $verdict == OK ]] || docker rename "$NEW" "$NEW-PARK-$verdict"
restore "v28 parked ($verdict)"
own "PARKED v28-prob = $NEW (stopped-kept; $kv; facts $verdict): the GB300-only recipe's reference is now v28 (v26 Seeded Adaptive + probabilistic drafting, Card BR4 P PASS). Lane serving = v27 sidecar pair + watchdog (restored). v26seed-BOUND-REF kept as its rollback."
{ echo; echo "## $(date '+%F %H:%M %Z') — GB300-only reference → v28-prob ($NEW, stopped-kept). Card BR4 P PASS: probabilistic drafting +10.3% tok/step, C1 +5–16% / C8 +8–19% by class at temp 1.0, fidelity at floor, KV 2,481,694 (−0.56%, draft-logits cache). PB (+block) +1.5% over P but FAILED two temp-0 parity gates (shell +3.6% vs 2.6% bar; turns 9/12 vs 6–8) → candidate, not promoted. :30006 serving stays v27 sidecar. Park facts: $verdict."; } >> ~/dsv41/results/ledger.md
log "===== PARK v28-prob END (running=$(docker ps --format '{{.Names}}' | grep -E '^dsv41-' | tr '\n' ' ') wd=$(wd_pids)) ====="
