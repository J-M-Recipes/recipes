#!/usr/bin/env bash
# cardAR6_runner.sh — Card AR6 v2 (REGISTERED 2026-10-24, AR6-SPEC-v2.md): seed map (H1) + queue-rate dose-response (H2)
# on hook v26ar.2 (G2 telemetry: PIN_STEP_RING=65536, PIN_SWAP_CUDA_EVENTS=1), lane :30006, GB300.
# Detached: setsid nohup bash cardAR6_runner.sh > /dev/null 2>&1 < /dev/null & echo $! > results-cardAR6.pid  (see DEPLOY-AR6.md)
# Header/conventions cloned from ar_runner.sh / ar3_runner.sh (paths, ctrl(), launch via launcher, boot-pin greps) and the
# card runners (cardSC5/cardGS: log, stop_all, drop_caches, wait_bind on /v1/models, strays, foreign-runner refusal,
# stop-and-keep, ledger append, START/END markers).
# ARM PLAN (spec §Arms, exactly; DECISIONS in DEPLOY-AR6.md):
#   preflight -> stop sidecar watchdog -> stop v25m pair (keep)
#   Boot 1 / Arm 0: ROWMAP=rowmap-static-v1.json, frozen boot
#       A0-frozen   held-out mixed C1 (ar_mixed_v2 HELDOUT=1, MIX_S) = H1 control + first-request tok/s
#       B0 x REPS   shift passes (shipped policy)  = T90 baseline + G3 churn base (static map)
#     stop+keep
#   Boot 2 / Arm A: ROWMAP=rowmap-mixed-ar-v1.json (seed), frozen boot
#       A-frozen    held-out mixed C1
#       rounds r=1..REPS: B0 (=A-adaptive), B1, B2   (interleaved; B cells compare against same-boot B0)
#       B3 x B3_REPS ONLY if B2 fails the G3 churn sub-gate (ar6_verdict.py --b3-check; default skip)
#       third-domain pass x1 (agent -> zh/math -> agent) at the provisional winner's knobs (B2 if none), non-gating
#       G4: Arm A fixed seed map (restore+freeze): parity F1,F2 (double capture) + gauntlet;
#           winner (if any): adapt under winner knobs (mixed C1), gauntlet live, parity W1 live, W2 frozen
#       dump counts
#     stop+keep -> restore v25m pair (docker start) -> re-arm watchdog -> verdict -> renames w/ suffix -> ledger -> END
# CELL PROTOCOL per pass (spec + W3): freeze -> restore_map -> warm traffic until pending=0,hot_moved=0 (>=1 drain)
#   -> baseline + shipped knobs + unfreeze -> agent PHASE_S (precondition, part of the pass)
#   -> AT THE SHIFT (written in-process by ar6_pass.py): baseline + cell knobs + unfreeze -> JP/DE PHASE_S -> agent PHASE_S
#   -> freeze. Gated shifts: agent->JP and JP->agent. Every ctrl write is atomic (tmp+mv) and logged as "CTRL";
#   every boundary goes to $R/phases.jsonl (wall clock) so the verdict cuts time-based windows.
# Refuses: an existing results dir, another live *runner.sh, foreign GPU containers, pre-existing AR6 containers,
#   Card GS not ENDed, v25m not serving, hook/harness sha1 mismatch. Abort: touch $R/ABORT (between cells) or kill -TERM.
#   Any exit after the lane was touched runs restore (v25m docker start + watchdog) exactly once.
set -uo pipefail
AR=$HOME/dsv41/ar
DS=$HOME/dsv41; D=$DS/m3
R=$AR/results-cardAR6
U=http://127.0.0.1:30006; M=dsv41-flash-uva
IMAGE=vllm/vllm-openai:deepseekv41-flash-0909
GB300=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55; RTX=GPU-d07eb6e6-7053-42a7-6811-7bb833266182
HOOK=$HOME/pin-hot-experts/v26ar2/hook
HOOK_V26AR=$HOME/pin-hot-experts/v26ar/hook              # rowmap-static-v1.json source (AR0/AR1/AR3 boots)
M3HOOK=$D/el8/deepseek-v4.1-flash/m3/hook                      # rowmap-mixed-ar-v1.json source (v25m)
MAP0=rowmap-static-v1.json; MAPA=rowmap-mixed-ar-v1.json
PROMPTS=$HOME/pin-hot-experts/e2c/parity-prompts.json
LAUNCH=$AR/launch_ar6.sh; PASS=$AR/ar6_pass.py; VERDICT=$AR/ar6_verdict.py
V25M=dsv41-vllm-1017-v25m-MIXMAP
WD=$D/sidecar_watchdog.py
N0=dsv41-vllm-1017-v26ar2-AR6-ARM0; NA=dsv41-vllm-1017-v26ar2-AR6-ARMA
GDIR=$HOME/pin-hot-experts/round7/gauntlet
GIMG=sha256:00d577a6a63281e15336029d5bcee4e9a2cf182214a4f20ba6111b1c8e79893d
LEDGER=$DS/results/ledger.md
GS_LOG=$D/gv/results-cardGS/runner.log
FF=' ERROR [0-9]{2}-[0-9]{2} |Traceback \(most recent call last\)|ValueError:|RuntimeError:|OutOfMemoryError|EngineCore failed'
PHASE_S=${PHASE_S:-90}; MIX_S=${MIX_S:-180}; REPS=${REPS:-3}
B3_FORCE=${B3_FORCE:-auto}; B3_REPS=${B3_REPS:-3}; MEM_MIN=${MEM_MIN:-380}; GS_CHECK=${GS_CHECK:-1}; WAIT_PID=${WAIT_PID:-}
K_SHIP='"drain_n": 64, "swaps_per_drain": 2, "swaps_per_barrier": 8, "hyst": 1.5, "min_abs": 8, "halflife": 500'
knobs(){ case $1 in
  B0) echo "$K_SHIP";;
  B1) echo '"drain_n": 32, "swaps_per_drain": 2, "swaps_per_barrier": 8, "hyst": 1.5, "min_abs": 8, "halflife": 500';;
  B2) echo '"drain_n": 16, "swaps_per_drain": 2, "swaps_per_barrier": 8, "hyst": 1.5, "min_abs": 8, "halflife": 500';;
  B3) echo '"drain_n": 64, "swaps_per_drain": 8, "swaps_per_barrier": 8, "hyst": 1.5, "min_abs": 8, "halflife": 500';;
  *) return 1;; esac; }
[[ -e $R/runner.log ]] && { echo "ABORT: $R/runner.log exists (refuse to mix runs; move it aside)" >&2; exit 5; }
mkdir -p "$R"
exec >> "$R/runner.log" 2>&1
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
memavail(){ awk '/MemAvailable/{printf "%.1f",$2/1048576}' /proc/meminfo; }
drop_caches(){ sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory' 2>/dev/null || true; }
stop_all(){ local n; for n in $(docker ps --format '{{.Names}}' | grep -E '^dsv41-vllm-'); do log "stop-and-keep $n"; docker stop "$n" >/dev/null; done; sleep 5; }
gpu_foreign(){ local c out=""; for c in $(docker ps --format '{{.Names}}' | grep -vE '^dsv41-vllm-'); do
    docker inspect "$c" --format '{{json .HostConfig.DeviceRequests}} {{json .HostConfig.Devices}}' 2>/dev/null | grep -qiE 'gpu|nvidia' && out="$out $c"; done; echo "$out"; }
any_foreign(){ docker ps --format '{{.Names}}' | grep -vE '^dsv41-vllm-' | tr '\n' ' '; }
other_runners(){ ps -eo pid,args | grep -E '^ *[0-9]+ bash .*(runner\.sh|perclass_run\.sh)' | grep -v "cardAR6_runner.sh" | grep -v grep | cut -c1-120 | tr '\n' '|'; }
archive_busy(){ ps -eo args | grep -E 'ark_bf1017\.sh|archive_1017\.sh|archive_sums|rsync .*miloark|sha256sum|hf download|docker pull|docker save' | grep -v grep | head -3 | cut -c1-120 | tr '\n' '|'; }
wait_quiet(){ local stage=$1 i a f
  for ((i=0;i<60;i++)); do
    a=$(archive_busy); f=$(any_foreign)
    [[ -z $a && -z ${f// /} ]] && { [[ $i -gt 0 ]] && log "QUIET for $stage after $((i*30))s"; return 0; }
    (( i % 10 == 0 )) && log "WAIT $stage: archive=[${a}] foreign=[${f}]"
    sleep 30
  done; log "WAIT $stage: still busy after 30 min"; return 1; }
strays(){ log "strays: $(ps -eo pid,rss,args --sort=-rss | sed -n 2,5p | awk '{printf "%s %.1fGiB %s %s|", $1, $2/1048576, $3, $4}') MemAvailable=$(memavail)GiB"; }
is_up(){ [[ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" == true ]]; }
rtx_used(){ nvidia-smi --id=$RTX --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' '; }
wait_bind(){ # name maxmin since [nofastfail] [sidecar]
  local n=$1 maxmin=$2 since=$3 nff=${4:-} side=${5:-} i t0 lf=$R/.wb-$1.log; t0=$(date +%s)
  for ((i=0;i<maxmin*6;i++)); do
    curl -s -m 2 $U/v1/models 2>/dev/null | grep -q "$M" && { log "BOUND $n after $(( $(date +%s)-t0 ))s"; return 0; }
    is_up "$n" || { log "DIED $n after $(( $(date +%s)-t0 ))s"; return 1; }
    if [[ -n $side ]] && ! is_up "$side"; then log "DIED sidecar $side after $(( $(date +%s)-t0 ))s"; return 1; fi
    docker logs --since "$since" "$n" > "$lf" 2>&1
    if [[ -z $nff ]] && grep -qE "$FF" "$lf"; then
      log "FASTFAIL $n after $(( $(date +%s)-t0 ))s: $(tr '\r' '\n' < "$lf" | grep -E "$FF" | tail -3 | cut -c1-240 | tr '\n' '|')"; return 1; fi
    (( i % 30 == 0 )) && log "  $n waiting $((i/6)) min: MemAvail=$(memavail)GiB last=[$(tr '\r' '\n' < "$lf" | grep -E 'PIN_HOT|ADAPT|Autotun|Capturing|KV cache size|Loading safetensors|Loading weights|compile' | tail -1 | cut -c1-160)]"
    sleep 10
  done; log "TIMEOUT $n ($maxmin min)"; return 1; }
errcount(){ # tag name since — same narrow regex as the fast-fail
  docker logs --since "$3" "$2" > "$R/postlog-$1.log" 2>&1; grep -cE "$FF" "$R/postlog-$1.log" > "$R/errcount-$1.txt"
  log "ERRCOUNT $1 $(cat "$R/errcount-$1.txt") (since $3)"; }
health(){ local tag=$1 code mc
  mc=$(curl -s -o /dev/null -w '%{http_code}' -m 10 $U/v1/models)
  code=$(curl -s -o "$R/health-$tag-chat.json" -w '%{http_code}' -m 120 -H 'Content-Type: application/json' $U/v1/chat/completions \
    -d '{"model":"'$M'","messages":[{"role":"user","content":"Reply with the single word OK."}],"max_tokens":8,"temperature":0,"chat_template_kwargs":{"thinking":false}}')
  log "HEALTH $tag: models_http=$mc chat_http=$code content=[$(python3 -c "import json;print(((json.load(open('$R/health-$tag-chat.json'))['choices'][0]['message'] or {}).get('content') or '')[:40])" 2>/dev/null)] sidecar=$(docker inspect -f '{{.State.Running}}' "$V25M-rtx" 2>/dev/null)"; }
arm_wd(){ setsid nohup python3 "$WD" --server "$1" --mode enforce --interval 1 --stall 2 \
    --status "$D/logs/sidecar_watchdog.status.json" --log "$D/logs/sidecar_watchdog.log" > /dev/null 2>&1 < /dev/null &
  echo $! > "$D/logs/sidecar_watchdog.pid"; sleep 3
  kill -0 "$(cat "$D/logs/sidecar_watchdog.pid")" && log "fail-stop watchdog armed pid $(cat "$D/logs/sidecar_watchdog.pid") on $1" || log "WARN: watchdog did not start"; }
gauntlet(){ local tag=$1
  rm -rf "$R/gauntlet-$tag"
  bg bash -c 'cd "$1" && shift && exec timeout 1200 python3 run_suite.py "$@"' _ "$GDIR" --base-url $U/v1 --model $M --output "$R/gauntlet-$tag" --tag "ar6-$tag" --cache-salt "ar6-$tag" --sandbox-image "$GIMG" --workers 4 > "$R/gauntlet-log-$tag.txt" 2>&1 || log "gauntlet $tag nonzero rc"
  log "GAUNTLET $tag: $(python3 -c "import json;d=json.load(open('$R/gauntlet-$tag/summary.json'));print(d.get('success_count'),'/',d.get('task_count'))" 2>/dev/null)"; }
# ---- AR conventions (ar3_runner.sh): RUN = the boot's /ar mount; ctrl() atomic tmp+mv then 3 s (ctrl poll is 1 s, mtime-gated)
RUN=""; BOOT="-"; NAME=""
ph(){ printf '{"t": %s, "ev": "%s", "boot": "%s"%s}\n' "$(date +%s.%N)" "$1" "$BOOT" "${2:+, $2}" >> "$R/phases.jsonl"; }
ctrl(){ printf '%s' "$1" > "$RUN/ctrl.tmp" && mv "$RUN/ctrl.tmp" "$RUN/ctrl.json" && touch "$RUN/ctrl.json"
  ph ctrl "\"ctrl\": $1${2:+, \"why\": \"$2\"}"; log "CTRL $BOOT ${2:-} $1"; sleep 3; }
parity(){ bg python3 "$AR/e2c_parity.py" "$1" "$PROMPTS" "$RUN" > "$RUN/parity-$1.log" 2>&1 || log "WARN parity $1 failed"; ph parity "\"tag\": \"$1\""; }
cmpp(){ python3 "$AR/e2c_compare_parity.py" "$RUN/parity-$1.json" "$RUN/parity-$2.json" "$1 vs $2" > "$RUN/cmp-$1-vs-$2.json" 2>&1
  log "PARITY $1 vs $2: $(python3 -c "import json;d=json.load(open('$RUN/cmp-$1-vs-$2.json'));print(d['exact'],'/',d['n'],[m['id'] for m in d['first_misses']])" 2>&1 | tail -1)"; }
bg(){ "$@" & local p=$!; wait $p; }   # long children run in background so TERM/INT traps fire immediately (bash defers traps during a foreground child)
check_abort(){ [[ -e $R/ABORT ]] && { log "ABORT file seen ($(cat "$R/ABORT" 2>/dev/null | head -1))"; exit 7; }; return 0; }

LANE_TOUCHED=0; RESTORED=0; FINISHED=0
restore_lane(){ # idempotent: AR6 containers stop-and-keep, v25m pair docker start (flags kept), watchdog re-armed
  [[ $RESTORED == 1 ]] && return 0; RESTORED=1
  local n since t0
  for n in "$N0" "$NA"; do is_up "$n" && { log "stop-and-keep $n"; docker stop -t 60 "$n" >/dev/null; }; done
  stop_all; drop_caches; strays
  BOOT=restore; ph restore_start
  t0=$(date +%s); since=$(date -u +%FT%TZ)
  docker start "$V25M-rtx" >/dev/null 2>&1 || log "WARN docker start $V25M-rtx rc=$?"
  for ((i=0;i<36;i++)); do docker logs --since "$since" "$V25M-rtx" > "$R/.restore-rtx.log" 2>&1
    grep -qE 'serving|prepared 40 layers' "$R/.restore-rtx.log" && break; sleep 5; done
  log "RESTORE sidecar $V25M-rtx running=$(docker inspect -f '{{.State.Running}}' "$V25M-rtx" 2>/dev/null) after $(( $(date +%s)-t0 ))s: $(grep -m1 -E 'serving|prepared 40 layers' "$R/.restore-rtx.log" | cut -c1-120)"
  docker start "$V25M" >/dev/null 2>&1 || log "WARN docker start $V25M rc=$?"
  if wait_bind "$V25M" 45 "$since" nofastfail "$V25M-rtx"; then
    log "LANE DARK ends ($V25M bound after $(( $(date +%s)-t0 ))s)"
    docker logs --since "$since" "$V25M" > "$R/restore-v25m.log" 2>&1
    log "RESTORE FACTS rowmap40=$(grep -c "MEGA_PEER rowmap /w/$MAPA: 40 layers" "$R/restore-v25m.log") peer2=$(grep -c 'peer mode 2, check 0' "$R/restore-v25m.log") kvdecl=$(docker inspect -f '{{json .Args}}' "$V25M" | grep -c 9126805504)"
    health restore; arm_wd "$V25M"
    [[ -n ${LANE_OWNER_SAVE:-} && -f $LANE_OWNER_SAVE ]] && { cp "$LANE_OWNER_SAVE" "$D/LANE-OWNER.txt"; echo "- $(date '+%F %T %Z') cardAR6 done; lane restored to $V25M (docker start) + fail-stop watchdog re-armed." >> "$D/LANE-OWNER.txt"; }
  else
    log "RESTORE FAILED — lane dark, needs hands (DEPLOY-AR6.md §Rollback)"
    echo "OWNER: cardAR6 RESTORE FAILED $(date '+%F %T %Z') — lane dark; see $R/runner.log" > "$D/LANE-OWNER.txt"
  fi
  rm -f "$D/logs/MAINTENANCE"; ph restore_end; }
on_exit(){ local rc=$?
  pkill -P $$ 2>/dev/null
  if [[ $FINISHED == 0 ]]; then
    log "EXIT rc=$rc before END (lane_touched=$LANE_TOUCHED)"
    if [[ $LANE_TOUCHED == 1 ]]; then restore_lane
      for n in "$N0" "$NA"; do docker ps -a --format '{{.Names}}' | grep -qx "$n" && docker rename "$n" "$n-ABORTED" 2>/dev/null && log "kept $n-ABORTED (stopped)"; done
    fi
    log "===== CARD AR6 END (ABORTED rc=$rc; lane = $(docker ps --format '{{.Names}}' | grep -E '^dsv41-' | tr '\n' ' ')) ====="
  fi; }
trap on_exit EXIT
trap 'log "SIGNAL received"; exit 143' TERM INT HUP

cell_reset(){ # freeze -> restore_map -> warm traffic until drained (>=1 drain) — baseline/knobs come next
  ctrl '{"freeze": true}' reset
  ctrl '{"restore_map": true}' reset
  ph drain_start
  bg python3 "$PASS" drain "$RUN" 80 > "$RUN/.drain.out" 2>&1; local rc=$?
  log "RESET $BOOT: $(tail -1 "$RUN/.drain.out")"; ph drain_end "\"rc\": $rc"
  return $rc; }
pass(){ # cell rep [domain2=jpde]
  local cell=$1 rep=$2 dom2=${3:-jpde} tag since rc kn
  check_abort
  kn=$(knobs "$cell") || { log "BUG unknown cell $cell"; return 1; }
  tag=$cell-r$rep; [[ $dom2 == third ]] && tag=$cell-third
  since=$(date -u +%FT%TZ)
  ph cell_start "\"cell\": \"$cell\", \"rep\": $rep, \"tag\": \"$tag\", \"knobs\": {$kn}"
  if ! cell_reset; then log "CELL $BOOT $tag SKIPPED: restore did not drain"; ph cell_skip "\"cell\": \"$cell\", \"rep\": $rep, \"tag\": \"$tag\""; return 1; fi
  ctrl "{\"baseline\": true, $K_SHIP, \"freeze\": false}" precondition
  log "PASS $BOOT $tag start: agent ${PHASE_S}s (shipped) -> $dom2 ${PHASE_S}s -> agent ${PHASE_S}s; shift knobs {$kn}"
  bg python3 "$PASS" pass "$tag" "$RUN" "$R/phases.jsonl" --boot "$BOOT" --cell "$cell" --rep "$rep" --phase-s "$PHASE_S" \
    --order "agent,$dom2,agent" --shift-at 1 --shift-ctrl "{\"baseline\": true, $kn, \"freeze\": false}" > "$RUN/pass-$tag.log" 2>&1; rc=$?
  log "CTRL $BOOT shift(in-process) $(grep -m1 '^\[ar6\] CTRL' "$RUN/pass-$tag.log" | cut -c1-200)"
  [[ $rc == 0 ]] && log "PASS $BOOT $tag done: $(tail -1 "$RUN/pass-$tag.log" | cut -c1-300)" || log "PASS $BOOT $tag FAILED rc=$rc: $(tail -2 "$RUN/pass-$tag.log" | tr '\n' '|' | cut -c1-300)"
  ctrl '{"freeze": true}' post-pass
  errcount "$BOOT-$tag" "$NAME" "$since"
  ph cell_end "\"cell\": \"$cell\", \"rep\": $rep, \"tag\": \"$tag\", \"rc\": $rc"
  return $rc; }
mixed_frozen(){ # tag — held-out mixed C1 on the frozen BOOT map (ctrl freeze already true from boot)
  local tag=$1 since; since=$(date -u +%FT%TZ)
  check_abort
  ph mixed_start "\"tag\": \"$tag\""
  bg env HELDOUT=1 python3 "$AR/ar_mixed_v2.py" "$tag" "$RUN" 1 "$MIX_S" > "$RUN/mixed-$tag.log" 2>&1 || log "WARN mixed $tag nonzero rc"
  ph mixed_end "\"tag\": \"$tag\""
  log "MIXED $BOOT $tag: $(tail -1 "$RUN/mixed-$tag.log" | cut -c1-330)"
  errcount "$BOOT-$tag" "$NAME" "$since"; }
boot_ar(){ # boot-tag name rowmap rundir
  local tag=$1 name=$2 map=$3 run=$4 since t0 ma
  BOOT=$tag; NAME=$name; RUN=$run; mkdir -p "$RUN"
  check_abort
  wait_quiet "boot-$tag" || log "WARN: host not quiet before boot $tag — proceeding (lane already dark)"
  stop_all; drop_caches; strays
  ma=$(memavail); if awk -v m="$ma" -v t="$MEM_MIN" 'BEGIN{exit !(m < t)}'; then log "ABORT boot $tag: MemAvailable ${ma} GiB < $MEM_MIN after stop+drop_caches (stray?)"; return 1; fi
  docker rm -f "$name" >/dev/null 2>&1 || true   # preflight guaranteed no AR6 container pre-existed; clears only this run's half-made one
  rm -f "$RUN/GO" "$RUN/ctrl.json" "$RUN/telemetry.jsonl" "$RUN/swaps.jsonl" "$RUN/steps.bin"
  t0=$(date +%s); since=$(date -u +%FT%TZ); ph boot_start "\"name\": \"$name\", \"rowmap\": \"$map\""
  HOOK=$HOOK NAME=$name RUN=$RUN ROWMAP=$map PIN_STEP_RING=65536 PIN_SWAP_CUDA_EVENTS=1 bash "$LAUNCH" >> "$R/launch-$tag.txt" 2>&1 || { log "LAUNCH FAILED $tag: $(tail -2 "$R/launch-$tag.txt" | tr '\n' '|')"; return 1; }
  log "SERVER $tag launched: $(tail -1 "$R/launch-$tag.txt")"
  if ! wait_bind "$name" 40 "$since"; then docker logs "$name" > "$R/boot-$tag-FAIL.log" 2>&1; return 1; fi
  log "boot->ready $tag $(( $(date +%s)-t0 ))s ($name)"
  docker logs --since "$since" "$name" > "$R/boot-$tag.log" 2>&1
  tr '\r' '\n' < "$R/boot-$tag.log" | grep -E "PIN_HOT|pin-hot|atomic|swap barrier|v26ar.2 telemetry|rowmap|KV cache size|Maximum concurrency|Autotuning process|Loaded [0-9]+ configs" | cut -c1-300 > "$RUN/boot-pin.log"
  local ok=1 k
  for k in "atomic counter ready" "atomic mode ready" "swap barrier" "v26ar.2 telemetry step_ring=65536" "cuda_events=1" "rowmap=/w/$map"; do
    grep -qF "$k" "$RUN/boot-pin.log" || { log "BOOT $tag missing line: $k"; ok=0; }; done
  grep -q "hook FAILED" "$RUN/boot-pin.log" && { log "BOOT $tag: PIN_HOT hook FAILED to install"; ok=0; }
  log "FACTS $tag: $(grep -m1 'GPU KV cache size' "$RUN/boot-pin.log" | sed 's/.*\] //' | cut -c1-80) | autotune_starts=$(grep -c 'Autotuning process starts' "$RUN/boot-pin.log") | $(grep -m1 'v26ar.2 telemetry' "$RUN/boot-pin.log" | sed 's/.*v26ar.2/v26ar.2/' | cut -c1-160)"
  [[ $ok == 1 ]] || { log "ARM $tag INVALID (hook wiring incomplete) — skipping this boot's cells"; return 1; }
  touch "$RUN/GO"; ph go; sleep 10
  grep -q '"baseline GO"' "$RUN/telemetry.jsonl" 2>/dev/null || log "WARN $tag: GO baseline telemetry line not seen yet"
  local sb; sb=$(stat -c %s "$RUN/steps.bin" 2>/dev/null || echo 0)
  [[ $sb -gt 0 ]] && log "STEPRING $tag steps.bin ${sb} B after GO+10s" || log "WARN $tag: steps.bin not growing (G2 will be UNMEASURED)"
  ph boot_ready "\"name\": \"$name\", \"rowmap\": \"$map\""
  return 0; }
stop_keep(){ local n=$1; is_up "$n" && { log "stop-and-keep $n"; docker stop -t 60 "$n" >/dev/null; }; ph stopped "\"name\": \"$n\""; }

# ================= preflight =================
log "===== CARD AR6 START (phase_s=$PHASE_S mix_s=$MIX_S reps=$REPS b3=$B3_FORCE/$B3_REPS hook=$HOOK maps=$MAP0,$MAPA) ====="
if [[ -n $WAIT_PID ]]; then log "WAIT for pid $WAIT_PID (Card GS) to exit"; while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done; log "pid $WAIT_PID gone"; fi
if [[ $GS_CHECK == 1 && -f $GS_LOG ]] && ! grep -q '===== CARD GS END' "$GS_LOG"; then log "REFUSE: Card GS has no END marker in $GS_LOG"; exit 4; fi
log "LANE-OWNER.txt: $(tr '\n' ' ' < "$D/LANE-OWNER.txt" 2>/dev/null | cut -c1-400)"
o=$(other_runners); [[ -n $o ]] && { log "REFUSE: another runner is live: $o"; exit 4; }
g=$(gpu_foreign); [[ -n ${g// /} ]] && { log "REFUSE foreign GPU container:$g"; exit 4; }
is_up "$V25M" && is_up "$V25M-rtx" || { log "ABORT: v25m pair not running (expected after Card GS restore) — lane untouched"; exit 5; }
curl -s -m 3 $U/v1/models | grep -q $M || { log "ABORT: :30006 not bound — lane untouched"; exit 5; }
x=$(docker ps -a --format '{{.Names}}' | grep -E '^dsv41-vllm-1017-v26ar2-AR6-' | tr '\n' ' '); [[ -n ${x// /} ]] && { log "ABORT: AR6 containers already exist: $x"; exit 5; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || { log "ABORT: $IMAGE missing"; exit 5; }
docker image inspect "$GIMG" >/dev/null 2>&1 || { log "ABORT: gauntlet sandbox image missing"; exit 5; }
chk(){ [[ -s $1 ]] || { log "ABORT: missing $1"; exit 5; }; [[ -z ${2:-} || "$(sha1sum "$1" | cut -c1-8)" == "$2" ]] || { log "ABORT: sha1 $(sha1sum "$1" | cut -c1-12) != $2 for $1"; exit 5; }; }
chk "$HOOK/e4_adaptive.py" 7ea55cbd; chk "$HOOK/pin_hot_experts_hook.py" 69bf0777; chk "$HOOK/sitecustomize.py" e65a93d7
chk "$AR/ar_shift_v2.py" a6cb41e0; chk "$AR/ar_mixed_v2.py" ce5231c1; chk "$AR/e2c_parity.py" 3de78e07; chk "$AR/e2c_compare_parity.py" f253aca8
chk "$PASS"; chk "$VERDICT"; chk "$LAUNCH"; chk "$PROMPTS"; chk "$GDIR/run_suite.py"; chk "$WD"
chk "$HOOK_V26AR/$MAP0"; chk "$M3HOOK/$MAPA"
cp -n "$HOOK_V26AR/$MAP0" "$HOOK/$MAP0" 2>/dev/null; cp -n "$M3HOOK/$MAPA" "$HOOK/$MAPA" 2>/dev/null
cmp -s "$HOOK_V26AR/$MAP0" "$HOOK/$MAP0" || { log "ABORT: $HOOK/$MAP0 differs from $HOOK_V26AR/$MAP0"; exit 5; }
cmp -s "$M3HOOK/$MAPA" "$HOOK/$MAPA" || { log "ABORT: $HOOK/$MAPA differs from $M3HOOK/$MAPA"; exit 5; }
{ for f in "$0" "$PASS" "$VERDICT" "$LAUNCH" "$AR/ar_shift_v2.py" "$AR/ar_mixed_v2.py" "$AR/e2c_parity.py" "$AR/e2c_compare_parity.py" "$PROMPTS" \
    "$HOOK/e4_adaptive.py" "$HOOK/pin_hot_experts_hook.py" "$HOOK/sitecustomize.py" "$HOOK/$MAP0" "$HOOK/$MAPA" "$GDIR/run_suite.py" "$WD"; do
    echo "sha1 $(sha1sum "$f" | cut -c1-12) $f"; done
  echo "knobs B0={$(knobs B0)} B1={$(knobs B1)} B2={$(knobs B2)} B3={$(knobs B3)}"; } > "$R/instruments.txt"
log "preflight OK: MemAvailable=$(memavail)GiB RTX=$(rtx_used)MiB (sha1s in instruments.txt)"
log "lane before: $(docker ps --format '{{.Names}}' | grep -E '^dsv41-' | tr '\n' ' ')"
wait_quiet preflight || { log "ABORT: host never quiet; lane untouched"; exit 6; }

# ================= lane dark: watchdog off, v25m pair stopped-and-kept =================
LANE_TOUCHED=1
LANE_OWNER_SAVE=$R/LANE-OWNER.before.txt; cp "$D/LANE-OWNER.txt" "$LANE_OWNER_SAVE" 2>/dev/null
echo "OWNER: cardAR6 runner pid $$ since $(date '+%F %T %Z') — lane :30006 dark (v26ar2 AR6 boots); restores $V25M at END. Receipts $R" > "$D/LANE-OWNER.txt"
echo "cardAR6 $(date '+%F %T %Z')" > "$D/logs/MAINTENANCE"
WP=$(cat "$D/logs/sidecar_watchdog.pid" 2>/dev/null)
[[ -n $WP ]] && kill "$WP" 2>/dev/null && log "fail-stop watchdog $WP stopped (cardAR6)"
pkill -f "sidecar_watchdog.py --server $V25M" 2>/dev/null; sleep 2
pgrep -f sidecar_watchdog.py >/dev/null && { log "ABORT: sidecar watchdog still alive — refusing to stop v25m under a fail-stop watchdog"; exit 6; }
echo "$(date '+%F %T %Z') watchdog stopped by cardAR6 (re-armed after restore)" >> "$D/logs/sidecar_watchdog.log"
mv "$D/logs/sidecar_watchdog.log" "$D/logs/sidecar_watchdog.log.pre-cardAR6-$(date +%Y%m%d-%H%M%S)"
BOOT=pre; ph lane_dark
log "LANE DARK begins (stopping v25m pair, kept)"
docker stop -t 60 "$V25M" "$V25M-rtx" >/dev/null 2>&1; log "stop-and-keep $V25M $V25M-rtx"

# ================= Boot 1 / Arm 0: static map =================
R0=$R/boot0
if boot_ar 0 "$N0" "$MAP0" "$R0"; then
  sinceB=$(date -u +%FT%TZ)
  mixed_frozen A0-frozen-heldout-c1
  log "G4 reference: static fixed-map parity double capture (frozen boot map)"
  parity Arm0-S1; parity Arm0-S2; cmpp Arm0-S1 Arm0-S2
  for r in $(seq 1 "$REPS"); do pass B0 "$r"; done
  errcount boot0 "$N0" "$sinceB"
  BOOT=0; ctrl '{"freeze": true}' end-arm0
else log "ARM 0 FAILED TO BOOT/VALIDATE — continuing to Arm A (H1 control will be MISSING)"; fi
BOOT=0; stop_keep "$N0"

# ================= Boot 2 / Arm A: seed map =================
RA=$R/bootA
if boot_ar A "$NA" "$MAPA" "$RA"; then
  sinceB=$(date -u +%FT%TZ)
  mixed_frozen A-frozen-heldout-c1
  for r in $(seq 1 "$REPS"); do pass B0 "$r"; pass B1 "$r"; pass B2 "$r"; done
  b3=SKIP
  case $B3_FORCE in
    1) b3=RUN; log "B3 forced (B3_FORCE=1)";;
    0) log "B3 skipped (B3_FORCE=0)";;
    *) l=$(python3 "$VERDICT" "$R" --b3-check 2> "$R/b3-check.err" | grep -m1 '^B3='); log "B3 CHECK: ${l:-B3=SKIP (check produced no line; default skip)}"
       [[ ${l:-} == B3=RUN* ]] && b3=RUN;;
  esac
  if [[ $b3 == RUN ]]; then for r in $(seq 1 "$B3_REPS"); do pass B3 "$r"; done; fi
  W=$(python3 "$VERDICT" "$R" --pick-winner 2> "$R/pick-winner.err" | sed -n 's/^WINNER=//p' | head -1); W=${W:-NONE}
  log "WINNER (provisional, for third-domain + G4): $W"; ph winner "\"winner\": \"$W\""
  TC=$W; [[ $TC == NONE ]] && TC=B2
  pass "$TC" 1 third
  # ---- G4: Arm A fixed seed map (double capture) + gauntlet; winner live + frozen
  check_abort
  ph g4_start "\"winner\": \"$W\""
  if cell_reset; then
    log "G4 ArmA fixed seed map (frozen, restored)"
    parity ArmA-F1; parity ArmA-F2; cmpp ArmA-F1 ArmA-F2
    if [[ -s $R0/parity-Arm0-S1.json ]]; then
      python3 "$AR/e2c_compare_parity.py" "$R0/parity-Arm0-S1.json" "$RUN/parity-ArmA-F1.json" "Arm0-S1 vs ArmA-F1" > "$RUN/cmp-Arm0-S1-vs-ArmA-F1.json" 2>&1
      log "PARITY Arm0-S1 vs ArmA-F1 (static vs seed, cross-boot): $(python3 -c "import json;d=json.load(open('$RUN/cmp-Arm0-S1-vs-ArmA-F1.json'));print(d['exact'],'/',d['n'],[m['id'] for m in d['first_misses']])" 2>&1 | tail -1)"
    else log "WARN: no Arm0-S1 capture — G4 parity ArmA vs static will be MISSING"; fi
    gauntlet ArmA
  else log "G4 ArmA: restore did not drain — G4 ArmA MISSING"; fi
  if [[ $W != NONE ]]; then
    ctrl "{\"baseline\": true, $(knobs "$W"), \"freeze\": false}" "g4-$W"
    bg python3 "$AR/ar_mixed_v2.py" "g4-$W-adapt" "$RUN" 1 "$MIX_S" > "$RUN/mixed-g4-$W-adapt.log" 2>&1 || log "WARN g4 adapt traffic nonzero rc"
    log "G4 $W adapt: $(tail -1 "$RUN/mixed-g4-$W-adapt.log" | cut -c1-200) | $(tail -1 "$RUN/telemetry.jsonl" | cut -c1-200)"
    gauntlet "$W"
    parity "$W-W1"
    ctrl '{"freeze": true}' "g4-$W-freeze"
    parity "$W-W2"
    cmpp ArmA-F1 "$W-W1"; cmpp ArmA-F1 "$W-W2"; cmpp "$W-W1" "$W-W2"
  else log "G4 winner: none (H2 provisional FAIL) — Arm A G4 only"; fi
  ph g4_end
  ctrl '{"freeze": true, "dump": "post-AR6-armA"}' dump; sleep 10
  log "DUMP $(ls -1 "$RUN"/counts-post-AR6-armA-*.npz 2>/dev/null | tail -1)"
  errcount bootA "$NA" "$sinceB"
else log "ARM A FAILED TO BOOT/VALIDATE — H1/H2 MISSING"; fi
BOOT=A; stop_keep "$NA"

# ================= restore, verdict, renames, ledger =================
restore_lane
python3 "$VERDICT" "$R" > "$R/VERDICT.txt" 2> "$R/verdict.err"; vrc=$?
SUF=$(python3 -c "import json;print(json.load(open('$R/verdict.json'))['suffix'])" 2>/dev/null); SUF=${SUF:-AR6-NOVERDICT}
log "VERDICT rc=$vrc $(python3 -c "import json;d=json.load(open('$R/verdict.json'));print(d['result'],'|',d['suffix'],'|',json.dumps({k:v[0] for k,v in d['gates'].items()}))" 2>&1 | tail -1 | cut -c1-900)"
for n in "$N0" "$NA"; do docker ps -a --format '{{.Names}}' | grep -qx "$n" && docker rename "$n" "$n-$SUF" 2>/dev/null && log "kept $n-$SUF (stopped)"; done
grep -hE ' (PASS|CTRL|RESET|CELL|MIXED|GAUNTLET|PARITY|WINNER|B3 CHECK|FACTS|STEPRING|SERVER|boot->ready|BOUND|FASTFAIL|DIED|TIMEOUT|ERRCOUNT|HEALTH|RESTORE|LANE|VERDICT|DUMP|WARN|ABORT|REFUSE|kept|fail-stop|stop-and-keep) ' "$R/runner.log" > "$R/SUMMARY.txt"
{ echo; echo "## $(date '+%F %H:%M %Z') — Card AR6 v2: seed map (H1) + queue-rate dose-response (H2), hook v26ar.2, :30006 — $(python3 -c "import json;d=json.load(open('$R/verdict.json'));print(d['result'], d['suffix'])" 2>/dev/null)"
  cat "$R/VERDICT.txt" 2>/dev/null; echo "Receipts: $R. NOT promoted — James decides (AR5 soak still owed for any H2 winner)."; } >> "$LEDGER"
FINISHED=1
log "FINAL dsv41 running=[$(docker ps --format '{{.Names}}' | grep -E '^dsv41-' | tr '\n' ' ')] bound=$(curl -s -m 3 $U/v1/models | grep -c $M)"
log "===== CARD AR6 END (lane = $V25M) ====="
