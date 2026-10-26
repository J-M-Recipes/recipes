#!/usr/bin/env bash
# rollback-v27.sh — roll the :30006 sidecar lane back from v27 (block verification) to the kept v25m-kvtier pair.
#
# Run ON THE STATION (it drives docker and the local watchdog). Idempotent: re-running after a partial run skips
# the steps that are already done (FROM already stopped/renamed, TO already serving, watchdog already on TO).
#
#   bash scripts/rollback-v27.sh [--dry-run] [--from <server>] [--to <server>]
#
#   --from   pair being rolled back   (default dsv41-vllm-1017-v27-block-BOUND-REF   + -rtx)
#   --to     kept pair to bring back  (default dsv41-vllm-1017-v25m-kvtier-RETIRED-REF + -rtx)
#   --dry-run  print every command (and every wait) without running anything; needs no docker.
#
# Steps (the promotion in results/2026-10-26-br1-block-verification/raw/promote-v27.log, reversed):
#   0  refuse if a card runner is live (pgrep -f runner.sh) or TO does not exist
#   1  kill the fail-stop watchdog (pid file, fallback pgrep) and confirm it is gone — it would trip on the stop
#   2  docker stop -t 60 FROM (server), then -t 30 FROM-rtx (sidecar); confirm :30006 is dark
#   3  rename FROM -> FROM-ROLLED-BACK-<ts> and FROM-rtx -> FROM-ROLLED-BACK-<ts>-rtx (stopped and KEPT)
#   4  docker start TO-rtx; wait <= 3 min for 'serving' or 'prepared 40 layers' in its log (this start only)
#   5  docker start TO; wait <= 45 min for /v1/models to list the model (hash-hit boot binds in ~8 min)
#   6  print the KV line (expect 'GPU KV cache size: 4,400,013 tokens'); one chat smoke (reasoning_effort low)
#      and one tool-call smoke
#   7  re-arm the watchdog from the argv captured at promotion (--server -> TO, --log -> sidecar_watchdog.rollback.log),
#      double-forked via setsid nohup; check PPID 1; write the pid file
#   8  append a LANE-OWNER line and a ledger line
# Step 7 runs even when a step-6 check failed, so a bound lane is never left without its fail-stop guard.
# Any failure: exit 4 and print 'NEEDS HANDS'. Exit 3 = refused before touching anything. Exit 2 = usage.
#
# Environment overrides (defaults are the reference box layout):
#   DSV41_HOME (~/dsv41)  BASE_URL (http://127.0.0.1:30006)  MODEL_NAME (dsv41-flash-uva)
#   WD_ARGV ($DSV41_HOME/m3/logs/promote-v27-wd-argv.txt)  KV_EXPECT (4,400,013)
#   API_KEY_FILE (unset; if set, its contents are sent as a Bearer token — the key never lives in this repo)
#   SIDECAR_WAIT_S (180)  BIND_WAIT_MIN (45)
set -uo pipefail

FROM=dsv41-vllm-1017-v27-block-BOUND-REF
TO=dsv41-vllm-1017-v25m-kvtier-RETIRED-REF
DRY=0
usage(){ sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'; }
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run) DRY=1 ;;
    --from) FROM=${2:?--from needs a container name}; shift ;;
    --to) TO=${2:?--to needs a container name}; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac; shift
done
[[ $FROM != "$TO" ]] || { echo "--from and --to are the same container ($FROM)" >&2; exit 2; }

DS=${DSV41_HOME:-$HOME/dsv41}; D=$DS/m3; LOGS=$D/logs
U=${BASE_URL:-http://127.0.0.1:30006}; M=${MODEL_NAME:-dsv41-flash-uva}
WD_ARGV=${WD_ARGV:-$LOGS/promote-v27-wd-argv.txt}
WD_PIDFILE=$LOGS/sidecar_watchdog.pid
WD_LOG_NEW=$LOGS/sidecar_watchdog.rollback.log
OWNER=$D/LANE-OWNER.txt; LEDGER=$DS/results/ledger.md
KV_EXPECT=${KV_EXPECT:-4,400,013}
SIDECAR_WAIT_S=${SIDECAR_WAIT_S:-180}; BIND_WAIT_MIN=${BIND_WAIT_MIN:-45}
TS=$(date -u +%Y%m%dT%H%M%SZ)
FROM_RB=$FROM-ROLLED-BACK-$TS
RLOG=$LOGS/rollback-v27-$TS.log
FAILS=""; NEWWD=""

# ---------------------------------------------------------------- helpers
log(){ local l; l="$(date '+%F %T %Z') $*"; echo "$l"; [[ $DRY == 1 ]] || echo "$l" >> "$RLOG"; }
fail(){ FAILS="$FAILS | $*"; log "FAIL: $*"; }
die(){ log "FAIL: $*"; own "ROLLBACK $FROM -> $TO FAILED at: $* — NEEDS HANDS (log $RLOG)"
  log "NEEDS HANDS: $* (manual fallback: README ## Rollback)"; exit 4; }
# run: mutating command — printed ('+') in dry-run, executed and logged otherwise ('?' lines are read-only probes)
run(){ if [[ $DRY == 1 ]]; then echo "+ $*"; return 0; fi; log "+ $*"; "$@" >> "$RLOG" 2>&1; }
own(){ if [[ $DRY == 1 ]]; then echo "+ echo '- <date> $*' >> $OWNER"; else mkdir -p "$(dirname "$OWNER")"; echo "- $(date '+%F %T %Z') $*" >> "$OWNER"; fi; }
auth=(); if [[ -n ${API_KEY_FILE:-} && -r ${API_KEY_FILE:-} ]]; then auth=(-H "Authorization: Bearer $(cat "$API_KEY_FILE")"); fi
running(){ # dry-run assumes the pre-rollback state: FROM pair running, TO pair stopped
  if [[ $DRY == 1 ]]; then echo "? docker inspect -f '{{.State.Running}}' $1" >&2; [[ $1 == "$FROM" || $1 == "$FROM-rtx" ]]; return; fi
  [[ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" == true ]]; }
exists(){ if [[ $DRY == 1 ]]; then echo "? docker inspect $1" >&2; return 0; fi; docker inspect "$1" >/dev/null 2>&1; }
bound(){ if [[ $DRY == 1 ]]; then echo "? curl -s -m 3 $U/v1/models | grep -q \"\\\"$M\\\"\"" >&2; return 1; fi
  curl -s -m 3 "${auth[@]+"${auth[@]}"}" "$U/v1/models" 2>/dev/null | grep -q "\"$M\""; }
wd_pids(){ # python watchdog processes only (a bash parent can carry the same argv text)
  if [[ $DRY == 1 ]]; then echo "? pgrep -f 'sidecar_watchdog\\.py'  (python3 processes only)" >&2; return 0; fi
  local p; for p in $(pgrep -f 'sidecar_watchdog\.py' 2>/dev/null); do
    [[ "$(ps -o comm= -p "$p" 2>/dev/null)" == python3* ]] && printf '%s ' "$p"; done; }

[[ $DRY == 1 ]] && echo "# DRY RUN — nothing below is executed. FROM=$FROM(+-rtx) TO=$TO(+-rtx) ts=$TS"
if [[ $DRY == 0 ]]; then
  mkdir -p "$LOGS" || { echo "cannot create $LOGS — NEEDS HANDS" >&2; exit 4; }
  LOCK=$LOGS/.rollback-v27.lock
  mkdir "$LOCK" 2>/dev/null || { echo "another rollback holds $LOCK (rmdir it if stale) — refusing" >&2; exit 3; }
  trap 'rmdir "$LOCK" 2>/dev/null' EXIT
fi
log "===== ROLLBACK $FROM -> $TO START (dry_run=$DRY) ====="

# ---------------------------------------------------------------- 0 preflight
if [[ $DRY == 1 ]]; then echo "? pgrep -af 'runner\\.sh'   (refuse if any, except an orphaned bash whose only child is the watchdog)"
else
  runners=""
  for p in $(pgrep -f 'runner\.sh' 2>/dev/null); do
    [[ $p == "$$" ]] && continue
    # the M2-era '( cd && setsid nohup ... & )' form left an orphaned bash carrying a runner's argv as the
    # watchdog's parent; it is not a runner (see results/2026-10-26-br1-block-verification/scripts/cardBR1_runner.sh)
    kids=$(pgrep -P "$p" 2>/dev/null | tr '\n' ' ')
    if [[ -n $kids && $(echo $kids | wc -w) -eq 1 ]] && ps -o args= -p $kids 2>/dev/null | grep -q sidecar_watchdog.py; then
      log "note: ignoring pid $p (orphaned bash holding the watchdog, not a runner)"; continue; fi
    runners="$runners $p:$(ps -o args= -p "$p" 2>/dev/null | cut -c1-120)"
  done
  if [[ -n $runners ]]; then log "REFUSE: a card runner is live —$runners"; log "lane untouched"; exit 3; fi
fi
command -v docker >/dev/null 2>&1 || [[ $DRY == 1 ]] || die "docker not on PATH"
exists "$TO" && exists "$TO-rtx" || { log "REFUSE: rollback target $TO / $TO-rtx does not exist — lane untouched"; exit 3; }

# ---------------------------------------------------------------- 1 watchdog off
wd_old=""
if [[ $DRY == 1 ]]; then echo "? cat $WD_PIDFILE   (verify /proc/<pid>/cmdline names sidecar_watchdog.py)"
elif [[ -s $WD_PIDFILE ]]; then
  p=$(tr -dc '0-9' < "$WD_PIDFILE")
  if [[ -n $p ]] && tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q sidecar_watchdog.py; then wd_old=$p
  else log "pid file $WD_PIDFILE -> '${p:-}' is not a live watchdog; falling back to pgrep"; fi
fi
targets="$wd_old $(wd_pids)"
# do not kill a watchdog that is already guarding TO (idempotent re-run after step 7)
keep=""; for p in $targets; do
  if [[ $DRY == 0 ]] && tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q -- "--server $TO "; then keep="$keep $p"; fi; done
for p in $(echo $targets | tr ' ' '\n' | sort -u); do
  [[ " $keep " == *" $p "* ]] && continue
  run kill "$p"
done
[[ $DRY == 1 ]] && echo "+ kill <watchdog pid>   (then poll 10 s; kill -9 if still alive)"
if [[ $DRY == 0 ]]; then
  for i in 1 2 3 4 5 6 7 8 9 10; do
    left=""; for p in $(wd_pids); do [[ " $keep " == *" $p "* ]] || left="$left $p"; done
    [[ -z $left ]] && break; sleep 1; done
  for p in $left; do run kill -9 "$p"; done; sleep 1
  left=""; for p in $(wd_pids); do [[ " $keep " == *" $p "* ]] || left="$left $p"; done
  [[ -z $left ]] || die "watchdog still alive after kill -9:$left"
  log "watchdog off (was: ${targets// /} ; kept on TO: ${keep:-none})"
fi

# ---------------------------------------------------------------- 2 stop FROM pair (keep)
if exists "$FROM"; then
  if running "$FROM"; then run docker stop -t 60 "$FROM" || die "docker stop $FROM"; else log "$FROM already stopped"; fi
else log "$FROM not present (already renamed by a previous run?) — skip stop"; fi
if exists "$FROM-rtx"; then
  if running "$FROM-rtx"; then run docker stop -t 30 "$FROM-rtx" || die "docker stop $FROM-rtx"; else log "$FROM-rtx already stopped"; fi
else log "$FROM-rtx not present — skip stop"; fi
if [[ $DRY == 0 ]]; then
  ! running "$FROM" && ! running "$FROM-rtx" 2>/dev/null || die "FROM pair still running after stop"
  if ! running "$TO" && bound; then die "$U still lists $M with FROM stopped and TO not running — something else holds the port"; fi
fi

# ---------------------------------------------------------------- 3 rename FROM -> FROM-ROLLED-BACK-<ts> (kept)
if exists "$FROM"; then run docker rename "$FROM" "$FROM_RB" || die "docker rename $FROM"; fi
if exists "$FROM-rtx"; then run docker rename "$FROM-rtx" "$FROM_RB-rtx" || die "docker rename $FROM-rtx"; fi

# ---------------------------------------------------------------- 4 sidecar first
if running "$TO-rtx" && [[ $DRY == 0 ]] && docker logs "$TO-rtx" 2>&1 | tr '\r' '\n' | grep -qE 'serving|prepared 40 layers'; then
  log "$TO-rtx already running and has logged serving — skip start"
else
  [[ $DRY == 0 ]] && running "$TO" && die "$TO is running without its sidecar $TO-rtx — stop it by hand first"
  since=$(date -u +%FT%TZ)
  run docker start "$TO-rtx" || die "docker start $TO-rtx"
  if [[ $DRY == 1 ]]; then echo "? poll ${SIDECAR_WAIT_S}s: docker logs --since $since $TO-rtx | grep -E 'serving|prepared 40 layers'"
  else ok=0
    for ((t=0; t<SIDECAR_WAIT_S; t+=5)); do
      docker logs --since "$since" "$TO-rtx" 2>&1 | tr '\r' '\n' | grep -qE 'serving|prepared 40 layers' && { ok=1; break; }
      running "$TO-rtx" || die "sidecar $TO-rtx exited during start: $(docker logs --tail 5 "$TO-rtx" 2>&1 | tr '\n' '|' | cut -c1-300)"
      sleep 5; done
    [[ $ok == 1 ]] || die "sidecar $TO-rtx did not log serving/prepared 40 layers in ${SIDECAR_WAIT_S}s"
    log "sidecar $TO-rtx: $(docker logs --since "$since" "$TO-rtx" 2>&1 | tr '\r' '\n' | grep -E 'prepared 40 layers|serving' | tail -2 | tr '\n' '|')"
  fi
fi

# ---------------------------------------------------------------- 5 server, wait for bind
SRV_SINCE=$(date -u +%FT%TZ)
if [[ $DRY == 0 ]] && running "$TO" && bound; then log "$TO already running and bound — skip start"; SRV_SINCE=""
else
  run docker start "$TO" || die "docker start $TO"
  if [[ $DRY == 1 ]]; then echo "? poll ${BIND_WAIT_MIN} min every 10 s: curl -s -m 3 $U/v1/models | grep \"$M\"  (abort if $TO or $TO-rtx exits)"
  else t0=$(date +%s); ok=0
    for ((i=0; i<BIND_WAIT_MIN*6; i++)); do
      bound && { ok=1; break; }
      running "$TO" || die "$TO exited during boot: $(docker logs --tail 5 "$TO" 2>&1 | tr '\n' '|' | cut -c1-300)"
      running "$TO-rtx" || die "sidecar $TO-rtx exited during server boot"
      (( i % 30 == 0 )) && log "  waiting for bind $((i/6)) min"
      sleep 10; done
    [[ $ok == 1 ]] || die "$TO did not bind in $BIND_WAIT_MIN min"
    log "BOUND $TO after $(( $(date +%s)-t0 ))s"
  fi
fi

# ---------------------------------------------------------------- 6 KV line + smokes (soft: watchdog is re-armed first)
if [[ $DRY == 1 ]]; then echo "? docker logs $TO | grep 'GPU KV cache size' | tail -1   (expect $KV_EXPECT tokens)"
else
  kv=$(docker logs ${SRV_SINCE:+--since "$SRV_SINCE"} "$TO" 2>&1 | grep -o 'GPU KV cache size: [0-9,]* tokens' | tail -1)
  log "FACTS: ${kv:-<no KV line>}"
  [[ $kv == "GPU KV cache size: $KV_EXPECT tokens" ]] || fail "KV line '${kv:-none}' != $KV_EXPECT (EXTRA/--kv-cache-memory lost?)"
fi
CHAT='{"model":"'$M'","messages":[{"role":"user","content":"Reply with the single word OK."}],"max_tokens":64,"temperature":0,"reasoning_effort":"low"}'
TOOL='{"model":"'$M'","messages":[{"role":"user","content":"Use the shell tool to run: uname -a"}],"max_tokens":512,"temperature":0,"reasoning_effort":"low","tool_choice":"auto","tools":[{"type":"function","function":{"name":"shell","description":"Run a shell command","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}]}'
if [[ $DRY == 1 ]]; then
  echo "+ curl -s -m 180 $U/v1/chat/completions -d '<chat: \"Reply with the single word OK.\", reasoning_effort low>'   (expect non-empty content)"
  echo "+ curl -s -m 300 $U/v1/chat/completions -d '<tools=[shell], tool_choice auto>'   (expect tool_calls[0].function.name == shell)"
else
  r=$(curl -s -m 180 "${auth[@]+"${auth[@]}"}" -H 'Content-Type: application/json' "$U/v1/chat/completions" -d "$CHAT")
  c=$(printf '%s' "$r" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(((d["choices"][0]["message"] or {}).get("content") or "").strip()[:60])' 2>/dev/null)
  log "SMOKE low: '${c}' $(printf '%s' "$r" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("usage"))' 2>/dev/null)"
  [[ -n $c ]] || fail "chat smoke returned no content: $(printf '%s' "$r" | cut -c1-200)"
  r=$(curl -s -m 300 "${auth[@]+"${auth[@]}"}" -H 'Content-Type: application/json' "$U/v1/chat/completions" -d "$TOOL")
  tc=$(printf '%s' "$r" | python3 -c 'import json,sys;d=json.load(sys.stdin);print([t["function"]["name"] for t in (d["choices"][0]["message"].get("tool_calls") or [])])' 2>/dev/null)
  log "SMOKE tool: ${tc:-<parse error>}"
  [[ $tc == *"'shell'"* ]] || fail "tool smoke: no shell tool call ($(printf '%s' "$r" | cut -c1-200))"
fi

# ---------------------------------------------------------------- 7 re-arm the fail-stop watchdog on TO
ARGS=()
if [[ -r $WD_ARGV ]]; then
  # one arg per line (/proc/<pid>/cmdline | tr '\0' '\n') or a single space-separated line
  if [[ $(grep -c . "$WD_ARGV") -le 1 ]]; then read -r -a ARGS < "$WD_ARGV"
  else while IFS= read -r a || [[ -n $a ]]; do [[ -n $a ]] && ARGS+=("$a"); done < "$WD_ARGV"; fi
elif [[ $DRY == 1 ]]; then
  echo "# (dry-run: $WD_ARGV not readable here; using the argv recorded in raw/promote-v27.log)"
  ARGS=(python3 "$D/sidecar_watchdog.py" --server "$FROM" --mode enforce --interval 1 --stall 2
        --status "$LOGS/sidecar_watchdog.status.json" --log "$LOGS/sidecar_watchdog.v27.log")
fi
if [[ ${#ARGS[@]} -eq 0 ]]; then
  fail "no captured watchdog argv at $WD_ARGV — lane serving WITHOUT the fail-stop guard"
else
  NA=(); prev=""; haslog=0
  for a in "${ARGS[@]}"; do
    [[ $a == "~/"* ]] && a=$HOME/${a#\~/}
    case $prev in --server) a=$TO ;; --log) a=$WD_LOG_NEW; haslog=1 ;; esac
    NA+=("$a"); prev=$a; done
  [[ $haslog == 1 ]] || NA+=(--log "$WD_LOG_NEW")
  printf '%s\n' "${NA[@]}" | grep -qx -- --server || NA+=(--server "$TO")
  have=""; [[ $DRY == 0 ]] && for p in $(wd_pids); do
    tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q -- "--server $TO " && have=$p; done
  if [[ -n $have ]]; then log "watchdog already armed on $TO (pid $have) — skip"; NEWWD=$have
  elif [[ $DRY == 1 ]]; then echo "+ ( cd $D; setsid nohup ${NA[*]} > /dev/null 2>&1 < /dev/null & )"
    echo "? pgrep watchdog on --server $TO; echo <pid> > $WD_PIDFILE; ps -o ppid= -p <pid>   (must be 1)"
  else
    log "re-arm watchdog: ${NA[*]}"
    ( cd "$D" || exit 1; setsid nohup "${NA[@]}" > /dev/null 2>&1 < /dev/null & )   # double fork: ';' so only setsid is backgrounded
    sleep 3
    for p in $(wd_pids); do tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q -- "--server $TO " && NEWWD=$p; done
  fi
  if [[ $DRY == 0 ]]; then
    if [[ -n $NEWWD ]] && kill -0 "$NEWWD" 2>/dev/null; then
      echo "$NEWWD" > "$WD_PIDFILE"
      pp=$(ps -o ppid= -p "$NEWWD" 2>/dev/null | tr -d ' ')
      log "watchdog armed pid $NEWWD ppid $pp on $TO (log $WD_LOG_NEW)"
      [[ $pp == 1 ]] || fail "watchdog PPID $pp != 1 (not detached)"
    else fail "watchdog did not start on $TO — lane serving WITHOUT the fail-stop guard"; NEWWD=""; fi
  fi
fi

# ---------------------------------------------------------------- 8 LANE-OWNER + ledger
if [[ -z $FAILS ]]; then
  own "ROLLBACK $FROM -> $TO done: lane = $TO + $TO-rtx, watchdog ${NEWWD:-<dry-run>}; $FROM kept as $FROM_RB (+ -rtx). Log $RLOG"
  line="- $(date '+%F %T %Z') ROLLBACK v27 -> $TO (scripts/rollback-v27.sh): bound, ${kv:-KV n/a}, chat+tool smoke OK, watchdog ${NEWWD:-n/a} PPID 1. FROM kept as $FROM_RB."
else
  own "ROLLBACK $FROM -> $TO finished WITH FAILURES —${FAILS} — NEEDS HANDS (log $RLOG)"
  line="- $(date '+%F %T %Z') ROLLBACK v27 -> $TO (scripts/rollback-v27.sh) NEEDS HANDS:${FAILS}"
fi
if [[ $DRY == 1 ]]; then echo "+ echo '<ledger line>' >> $LEDGER"
else mkdir -p "$(dirname "$LEDGER")"; echo "$line" >> "$LEDGER"; fi

if [[ -n $FAILS ]]; then log "NEEDS HANDS:${FAILS}"; log "===== ROLLBACK END (FAILED) ====="; exit 4; fi
log "===== ROLLBACK $FROM -> $TO END (lane = $TO $TO-rtx  watchdog=${NEWWD:-dry-run}) ====="
exit 0
