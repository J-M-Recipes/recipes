#!/usr/bin/env bash
# cardKV3_runner.sh — Card KV3 FULL BATTERY (KV3-SPEC-DRAFT.md §SMOKE RESULT "Still owed"): host (Grace) prefix-KV tier
# (vLLM OffloadingConnector, native backend, 48 GiB) vs no-tier control, bracketed K0 -> K1 -> K0R, lane :30006, GB300.
# Detached: cd ~/dsv41/m3/kv3 && setsid nohup bash cardKV3_runner.sh >/dev/null 2>&1 </dev/null & echo $! > cardKV3.pid
# Conventions cloned from cardAR6_runner.sh.DRAFT (log/ph helpers, stop_all, wait_bind, drop_caches, MEM_MIN gate after
# stop+drop, foreign-runner refusal, bg() trap pattern, idempotent restore + EXIT/TERM traps, stop-and-keep renames with
# verdict suffix, START/END markers, ledger append, verdict.json gates{}). DECISIONS: DEPLOY-KV3.md.
# ARM PLAN:
#   preflight (refusals leave the lane untouched) -> stop v26seed-BOUND-REF (keep; it runs with NO watchdog)
#   per arm X in K0 K1 K0R:  stop_all + drop_caches + MEM_MIN gate -> launch_kv3.sh sidecar -> launch_kv3.sh server
#       (EXTRA = KV declare; K1 adds the tier flags) -> arm KV3 watchdog (enforce) -> wait_bind -> boot facts
#       cells: (a) held-out mixed C1 x6 (ar_mixed_v2 HELDOUT=1, MIX_S)  (b) cold-64K TTFT x3, first dropped
#              (c) forced-length warm replay r1,r2  (d) 1M-class needle x1
#              (e) K1 only: in-window evict-replay probe + gauntlet;  K0/K0R: evictctl (same docs, no reset; info only)
#              (f) MemAvailable sampler (boot + cells), watchdog timeouts/trips, FF errcount, 'GPU KV cache size'
#       -> kill watchdog -> stop pair (keep)
#   END: restore v26seed-BOUND-REF (docker start, wait_bind, GO + ctrl unfreeze; NO watchdog) -> verdict -> renames -> ledger
# Abort: touch $R/ABORT (checked between cells/reps/boots) or kill -TERM <pid>. Any exit after the lane was touched
#   runs restore exactly once (EXIT/TERM trap), renames KV3 containers -ABORTED.
set -uo pipefail
HM=${KV3_HOME:-~}   # desk dry-run override only; production = ~
DS=$HM/dsv41; D=$DS/m3; K=$D/kv3
R=$K/results-cardKV3
U=http://127.0.0.1:30006; M=dsv41-flash-uva
IMAGE=vllm/vllm-openai@sha256:3af6a45b060d681aa6254200f87859d1bb3c1f962afb3d15540b4b31d4a41063   # = launch_kv3.sh default (af7f9488)
MODEL=${KV3_MODEL_DIR:-/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170}
GB300=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55; RTX=GPU-d07eb6e6-7053-42a7-6811-7bb833266182
M3HOOK=$D/el8/deepseek-v4.1-flash/m3/hook
ROWMAP=rowmap-mixed-ar-v1.json
LAUNCH=$D/launch_kv3.sh
CELLS=$K/kv3_cells.py; VERDICT=$K/kv3_verdict.py
MIXED=$DS/ar/ar_mixed_v2.py
WD=$D/sidecar_watchdog.py
WDLOG=$D/logs/sidecar_watchdog.kv3.log; WDSTAT=$D/logs/sidecar_watchdog.status.json; WDPID=$D/logs/sidecar_watchdog.kv3.pid
REF=dsv41-vllm-1017-v26seed-BOUND-REF
SOAK=$DS/ar/soak
N_K0=dsv41-vllm-1017-KV3-K0; N_K1=dsv41-vllm-1017-KV3-K1; N_K0R=dsv41-vllm-1017-KV3-K0R
ALLN="$N_K0 $N_K0-rtx $N_K1 $N_K1-rtx $N_K0R $N_K0R-rtx"
KVDECL="--kv-cache-memory 9126805504"
TIER="--kv-offloading-size 48 --kv-offloading-backend native"
KV_EXPECT=4,400,013
GDIR=$HM/pin-hot-experts/round7/gauntlet
GIMG=sha256:00d577a6a63281e15336029d5bcee4e9a2cf182214a4f20ba6111b1c8e79893d
LEDGER=$DS/results/ledger.md
FF=' ERROR [0-9]{2}-[0-9]{2} |Traceback \(most recent call last\)|ValueError:|RuntimeError:|OutOfMemoryError|EngineCore failed'
# sha1 prefixes (8) — override via env only after re-reviewing the file (set-but-empty disables a check: desk dry-run only)
SHA_LAUNCH=${SHA_LAUNCH-5c02229c}; SHA_MIXED=${SHA_MIXED-ce5231c1}; SHA_WD=${SHA_WD-b17d9103}
SHA_CELLS=${SHA_CELLS-7a96fc2f}; SHA_VERDICT=${SHA_VERDICT-dc36c55f}
MIX_S=${MIX_S:-120}; C1_REPS=${C1_REPS:-6}; COLD_N=${COLD_N:-3}; COLD_TOK=${COLD_TOK:-64000}
WARM_TOK=${WARM_TOK:-8000}; WARM_MAX=${WARM_MAX:-256}; NEEDLE_TOK=${NEEDLE_TOK:-950000}; EVICT_LENS=${EVICT_LENS:-8205,33010}
MEM_MIN=${MEM_MIN:-380}; MEM_GUARD=${MEM_GUARD:-30}; BOOT_MIN=${BOOT_MIN:-40}; WAIT_PID=${WAIT_PID:-}
ARMS=${ARMS:-"K0 K1 K0R"}

if [[ -e $R ]]; then echo "$(date '+%F %T %Z') REFUSE: $R exists (refuse to mix runs; move it aside)" | tee -a "$K/cardKV3.refused.log" >&2; exit 5; fi
mkdir -p "$R"
exec >> "$R/runner.log" 2>&1
exec 3>> "$R/runner.log"   # log() always writes here: a trap firing inside `bg ... > cell.log` inherits that redirect on fd 1
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*" >&3; }
MEMINFO=${MEMINFO:-/proc/meminfo}
memavail(){ awk '/MemAvailable/{printf "%.1f",$2/1048576}' "$MEMINFO"; }
drop_caches(){ sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory' 2>/dev/null || true; }
stop_all(){ local n; for n in $(docker ps --format '{{.Names}}' | grep -E '^dsv41-vllm-'); do log "stop-and-keep $n"; docker stop -t 60 "$n" >/dev/null; done; sleep 5; }
gpu_foreign(){ local c out=""; for c in $(docker ps --format '{{.Names}}' | grep -vE '^dsv41-vllm-'); do
    docker inspect "$c" --format '{{json .HostConfig.DeviceRequests}} {{json .HostConfig.Devices}}' 2>/dev/null | grep -qiE 'gpu|nvidia' && out="$out $c"; done; echo "$out"; }
any_foreign(){ docker ps --format '{{.Names}}' | grep -vE '^dsv41-vllm-' | tr '\n' ' '; }
other_runners(){ ps -eo pid,args | grep -E '^ *[0-9]+ bash .*(runner\.sh|perclass_run\.sh)' | grep -v "cardKV3_runner.sh" | grep -v grep | cut -c1-120 | tr '\n' '|'; }
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
exists(){ docker ps -a --format '{{.Names}}' | grep -qx "$1"; }
rtx_used(){ nvidia-smi --id=$RTX --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' '; }
oom_count(){ { sudo -n dmesg 2>/dev/null || journalctl -k --no-pager -q 2>/dev/null; } | grep -ciE 'oom-kill|out of memory: kill' ; }
wait_bind(){ # name maxmin since [nofastfail] [sidecar]
  local n=$1 maxmin=$2 since=$3 nff=${4:-} side=${5:-} i t0 lf=$R/.wb-$1.log; t0=$(date +%s)
  for ((i=0;i<maxmin*6;i++)); do
    curl -s -m 2 $U/v1/models 2>/dev/null | grep -q "$M" && { log "BOUND $n after $(( $(date +%s)-t0 ))s"; return 0; }
    is_up "$n" || { log "DIED $n after $(( $(date +%s)-t0 ))s"; return 1; }
    if [[ -n $side ]] && ! is_up "$side"; then log "DIED sidecar $side after $(( $(date +%s)-t0 ))s"; return 1; fi
    docker logs --since "$since" "$n" > "$lf" 2>&1
    if [[ -z $nff ]] && grep -qE "$FF" "$lf"; then
      log "FASTFAIL $n after $(( $(date +%s)-t0 ))s: $(tr '\r' '\n' < "$lf" | grep -E "$FF" | tail -3 | cut -c1-240 | tr '\n' '|')"; return 1; fi
    (( i % 30 == 0 )) && log "  $n waiting $((i/6)) min: MemAvail=$(memavail)GiB last=[$(tr '\r' '\n' < "$lf" | grep -E 'MEGA_PEER|Offload|offload|Autotun|Capturing|KV cache size|Loading safetensors|Loading weights|compile' | tail -1 | cut -c1-160)]"
    sleep 10
  done; log "TIMEOUT $n ($maxmin min)"; return 1; }
errcount(){ # tag name since — same narrow regex as the fast-fail
  docker logs --since "$3" "$2" > "$R/postlog-$1.log" 2>&1; grep -cE "$FF" "$R/postlog-$1.log" > "$R/errcount-$1.txt"
  log "ERRCOUNT $1 $(cat "$R/errcount-$1.txt") (since $3)"; }
health(){ local tag=$1 side=${2:-} code mc
  mc=$(curl -s -o /dev/null -w '%{http_code}' -m 10 $U/v1/models)
  code=$(curl -s -o "$R/health-$tag-chat.json" -w '%{http_code}' -m 120 -H 'Content-Type: application/json' $U/v1/chat/completions \
    -d '{"model":"'$M'","messages":[{"role":"user","content":"Reply with the single word OK."}],"max_tokens":8,"temperature":0,"chat_template_kwargs":{"thinking":false}}')
  log "HEALTH $tag: models_http=$mc chat_http=$code content=[$(python3 -c "import json;print(((json.load(open('$R/health-$tag-chat.json'))['choices'][0]['message'] or {}).get('content') or '')[:40])" 2>/dev/null)]${side:+ sidecar=$(docker inspect -f '{{.State.Running}}' "$side" 2>/dev/null)}"; }
gauntlet(){ local tag=$1
  rm -rf "$R/gauntlet-$tag"
  bg bash -c 'cd "$1" && shift && exec timeout 1200 python3 run_suite.py "$@"' _ "$GDIR" --base-url $U/v1 --model $M --output "$R/gauntlet-$tag" --tag "kv3-$tag" --cache-salt "kv3-$tag" --sandbox-image "$GIMG" --workers 4 > "$R/gauntlet-log-$tag.txt" 2>&1 || log "gauntlet $tag nonzero rc"
  log "GAUNTLET $tag: $(python3 -c "import json;d=json.load(open('$R/gauntlet-$tag/summary.json'));print(d.get('success_count'),'/',d.get('task_count'))" 2>/dev/null)"; }
BOOT="-"; NAME=""; A=""
ph(){ printf '{"t": %s, "ev": "%s", "boot": "%s"%s}\n' "$(date +%s.%N)" "$1" "$BOOT" "${2:+, $2}" >> "$R/phases.jsonl"; }
setphase(){ printf '%s\n' "$1" > "$R/.phase.tmp" && mv "$R/.phase.tmp" "$R/.phase"; }
bg(){ "$@" & local p=$!; wait $p; }   # long children in background so TERM/INT traps fire immediately
check_abort(){ [[ -e $R/ABORT ]] && { log "ABORT file seen ($(head -1 "$R/ABORT" 2>/dev/null))"; exit 7; }; return 0; }

# ---- MemAvailable sampler (5 s): "<epoch> <phase> <GiB>" -> $R/mem.tsv; live guard writes ABORT below MEM_GUARD
mem_sampler(){ local m p g=0
  while :; do m=$(memavail); p=$(cat "$R/.phase" 2>/dev/null || echo -); echo "$(date +%s) $p $m" >> "$R/mem.tsv"
    if [[ $g == 0 ]] && awk -v m="$m" -v t="$MEM_GUARD" 'BEGIN{exit !(m < t)}'; then
      g=1; echo "MEM_GUARD: MemAvailable ${m} GiB < ${MEM_GUARD} during $p" > "$R/ABORT"; log "MEM_GUARD tripped: ${m} GiB during $p (ABORT written)"; fi
    sleep 5; done; }

# ---- KV3 watchdog (one per boot; enforce). Killed BEFORE the pair is stopped (else it trips on the sidecar stop).
arm_wd(){ [[ -s $WDLOG ]] && mv "$WDLOG" "$R/watchdog-stale-$(date +%H%M%S).log"
  rm -f "$WDSTAT"
  setsid nohup python3 "$WD" --server "$1" --mode enforce --interval 1 --stall 2 \
    --status "$WDSTAT" --log "$WDLOG" > /dev/null 2>&1 < /dev/null &
  echo $! > "$WDPID"; sleep 3
  kill -0 "$(cat "$WDPID")" 2>/dev/null && log "KV3 watchdog armed pid $(cat "$WDPID") on $1 (enforce, stall 2 s)" || log "WARN: KV3 watchdog did not start on $1"; }
wd_collect(){ # arm -> $R/<arm>/wd.json ; must run BEFORE kill_wd
  local arm=$1 alive=false p; p=$(cat "$WDPID" 2>/dev/null)
  [[ -n $p ]] && kill -0 "$p" 2>/dev/null && alive=true
  python3 - "$WDSTAT" "$WDLOG" "$alive" "$R/$arm/wd.json" <<'PY' 2>/dev/null || echo '{"error": "wd_collect failed"}' > "$R/$arm/wd.json"
import json, sys
stat, logf, alive, out = sys.argv[1:5]
d = {"alive_at_collect": alive == "true", "status": None, "timeouts": None, "trips": None}
try:
    st = json.load(open(stat)); d["status"] = st
    if st.get("words") is not None and st.get("timeouts_base") is not None:
        d["timeouts"] = st["words"][2] - st["timeouts_base"]
except Exception as e:
    d["status_error"] = repr(e)
try:
    L = open(logf).read().splitlines()
    d["trips"] = sum(1 for l in L if " TRIP " in l); d["log_lines"] = len(L); d["trip_lines"] = [l for l in L if " TRIP " in l][:5]
except Exception as e:
    d["log_error"] = repr(e)
json.dump(d, open(out, "w"), indent=1)
PY
  log "WATCHDOG $arm: $(python3 -c "import json;d=json.load(open('$R/$arm/wd.json'));print('alive',d.get('alive_at_collect'),'timeouts',d.get('timeouts'),'trips',d.get('trips'))" 2>&1 | tail -1)"; }
kill_wd(){ local p; p=$(cat "$WDPID" 2>/dev/null)
  [[ -n $p ]] && kill "$p" 2>/dev/null && log "KV3 watchdog $p stopped"
  pkill -f "sidecar_watchdog.py --server dsv41-vllm-1017-KV3-" 2>/dev/null; sleep 2
  pgrep -f "sidecar_watchdog.py --server dsv41-vllm-1017-KV3-" >/dev/null && { log "WARN: KV3 watchdog still alive; kill -9"; pkill -9 -f "sidecar_watchdog.py --server dsv41-vllm-1017-KV3-"; }
  rm -f "$WDPID"; return 0; }
stop_pair(){ local n=$1
  kill_wd
  is_up "$n" && { log "stop-and-keep $n"; docker stop -t 60 "$n" >/dev/null; }
  is_up "$n-rtx" && { log "stop-and-keep $n-rtx"; docker stop -t 30 "$n-rtx" >/dev/null; }
  [[ -n $A && -s $WDLOG ]] && mv "$WDLOG" "$A/watchdog.log"
  ph stopped "\"name\": \"$n\""; return 0; }

LANE_TOUCHED=0; RESTORED=0; FINISHED=0; SAMPLER=""
restore_lane(){ # idempotent: KV3 pairs stop-and-keep, v26seed-BOUND-REF docker start (Grace-only, NO watchdog), GO + unfreeze
  [[ $RESTORED == 1 ]] && return 0; RESTORED=1
  local n since t0
  BOOT=restore; setphase restore; ph restore_start
  kill_wd
  for n in $N_K0 $N_K1 $N_K0R; do is_up "$n" && { log "stop-and-keep $n"; docker stop -t 60 "$n" >/dev/null; }
    is_up "$n-rtx" && { log "stop-and-keep $n-rtx"; docker stop -t 30 "$n-rtx" >/dev/null; }; done
  stop_all; drop_caches; strays
  t0=$(date +%s); since=$(date -u +%FT%TZ)
  rm -f "$SOAK/GO"   # GO is re-touched after bind (mtime = post-bind, as boot_ar did for its RUN dir)
  docker start "$REF" >/dev/null 2>&1 || log "WARN docker start $REF rc=$?"
  if wait_bind "$REF" 45 "$since" nofastfail; then
    log "LANE DARK ends ($REF bound after $(( $(date +%s)-t0 ))s)"
    docker logs --since "$since" "$REF" > "$R/restore-ref.log" 2>&1
    touch "$SOAK/GO"
    printf '%s' '{"freeze": false, "baseline": true}' > "$SOAK/ctrl.tmp" && mv "$SOAK/ctrl.tmp" "$SOAK/ctrl.json" && touch "$SOAK/ctrl.json"
    log "CTRL restore GO touched + $SOAK/ctrl.json = $(cat "$SOAK/ctrl.json")"
    log "RESTORE FACTS kvdecl=$(docker inspect -f '{{json .Args}}' "$REF" | grep -c 9126805504) kvline=[$(tr '\r' '\n' < "$R/restore-ref.log" | grep -m1 -oE 'GPU KV cache size: [0-9,]+ tokens')] watchdog=none(by design)"
    health restore
    [[ -n ${LANE_OWNER_SAVE:-} && -f $LANE_OWNER_SAVE ]] && { cp "$LANE_OWNER_SAVE" "$D/LANE-OWNER.txt"; echo "- $(date '+%F %T %Z') cardKV3 done; lane restored to $REF (docker start, Grace-only, no watchdog)." >> "$D/LANE-OWNER.txt"; }
  else
    log "RESTORE FAILED — lane dark, needs hands (DEPLOY-KV3.md §Rollback)"
    echo "OWNER: cardKV3 RESTORE FAILED $(date '+%F %T %Z') — lane dark; see $R/runner.log" > "$D/LANE-OWNER.txt"
  fi
  rm -f "$D/logs/MAINTENANCE"; ph restore_end; }
on_exit(){ local rc=$?
  exec >> "$R/runner.log" 2>&1   # undo any cell-level redirect the signal arrived under
  pkill -P $$ 2>/dev/null
  if [[ $FINISHED == 0 ]]; then
    log "EXIT rc=$rc before END (lane_touched=$LANE_TOUCHED)"
    if [[ $LANE_TOUCHED == 1 ]]; then restore_lane
      python3 "$VERDICT" "$R" > "$R/VERDICT.txt" 2> "$R/verdict.err"; log "partial verdict rc=$? -> VERDICT.txt (not ledgered)"
      for n in $ALLN; do exists "$n" && docker rename "$n" "$n-ABORTED" 2>/dev/null && log "kept $n-ABORTED (stopped)"; done
    fi
    log "===== CARD KV3 END (ABORTED rc=$rc; lane = $(docker ps --format '{{.Names}}' | grep -E '^dsv41-' | tr '\n' ' ')) ====="
  fi; }
trap on_exit EXIT
trap 'log "SIGNAL received"; exit 143' TERM INT HUP

boot_kv3(){ # arm name extra
  local arm=$1 name=$2 extra=$3 since t0 ma
  BOOT=$arm; NAME=$name; A=$R/$arm; mkdir -p "$A"
  check_abort
  setphase "$arm:preboot"
  wait_quiet "boot-$arm" || log "WARN: host not quiet before boot $arm — proceeding (lane already dark)"
  stop_all; drop_caches; strays
  ma=$(memavail); if awk -v m="$ma" -v t="$MEM_MIN" 'BEGIN{exit !(m < t)}'; then log "ABORT boot $arm: MemAvailable ${ma} GiB < $MEM_MIN after stop+drop_caches (stray?)"; return 1; fi
  docker rm -f "$name" "$name-rtx" >/dev/null 2>&1 || true   # preflight guaranteed none pre-existed; clears only this run's half-made pair
  t0=$(date +%s); since=$(date -u +%FT%TZ); echo "$since" > "$A/boot_since.txt"
  setphase "$arm:boot"; ph boot_start "\"name\": \"$name\", \"extra\": \"$extra\""
  log "BOOT $arm: $name EXTRA=[$extra] ROWMAP=$ROWMAP MemAvail=${ma}GiB"
  bg env NAME="$name" ROWMAP="$ROWMAP" EXTRA="$extra" bash "$LAUNCH" sidecar >> "$R/launch-$arm.txt" 2>&1 || { log "LAUNCH sidecar FAILED $arm: $(tail -2 "$R/launch-$arm.txt" | tr '\n' '|')"; return 1; }
  log "SIDECAR $arm serving: $(grep -m1 '^serving' "$D/logs/peer_server2.log" 2>/dev/null | cut -c1-120)"
  bg env NAME="$name" ROWMAP="$ROWMAP" EXTRA="$extra" bash "$LAUNCH" server >> "$R/launch-$arm.txt" 2>&1 || { log "LAUNCH server FAILED $arm: $(tail -2 "$R/launch-$arm.txt" | tr '\n' '|')"; return 1; }
  log "SERVER $arm launched: $(tail -1 "$R/launch-$arm.txt")"
  arm_wd "$name"
  if ! wait_bind "$name" 40 "$since" "" "$name-rtx"; then docker logs "$name" > "$A/boot-FAIL.log" 2>&1; return 1; fi
  log "boot->ready $arm $(( $(date +%s)-t0 ))s ($name)"
  echo "$(date +%s) $arm:boot $(memavail)" >> "$R/mem.tsv"   # guarantees >=1 boot sample even on a fast boot
  docker logs --since "$since" "$name" > "$A/boot.log" 2>&1
  tr '\r' '\n' < "$A/boot.log" | grep -E 'KV cache size|Maximum concurrency|Offload|offload|cudaHostRegister|MEGA_PEER rowmap|kv_transfer|KVConnector' | cut -c1-300 > "$A/boot-facts.log"
  local kv conn pinw args
  kv=$(grep -m1 -oE 'GPU KV cache size: [0-9,]+ tokens' "$A/boot-facts.log" | grep -oE '[0-9,]+' | head -1)
  conn=$(grep -ciE 'OffloadingConnector' "$A/boot-facts.log")
  pinw=$(grep -iE 'cudaHostRegister' "$A/boot-facts.log" | grep -ciE 'warn|fail|error|unpinned')
  args=$(docker inspect -f '{{json .Args}}' "$name" 2>/dev/null)
  printf '{"arm": "%s", "name": "%s", "kv_tokens": "%s", "kv_expect": "%s", "connector_lines": %s, "pin_warn_lines": %s, "kvdecl_in_args": %s, "tier_in_args": %s, "boot_s": %s, "bound_at": %s}\n' \
    "$arm" "$name" "${kv:-}" "$KV_EXPECT" "${conn:-0}" "${pinw:-0}" "$(grep -c 9126805504 <<<"$args")" "$(grep -c 'kv-offloading-size' <<<"$args")" "$(( $(date +%s)-t0 ))" "$(date +%s)" > "$A/facts.json"
  log "FACTS $arm: kv=[${kv:-MISSING}] expect=$KV_EXPECT connector_lines=$conn pin_warn=$pinw kvdecl=$(grep -c 9126805504 <<<"$args") tier=$(grep -c 'kv-offloading-size' <<<"$args")"
  [[ ${kv:-} == "$KV_EXPECT" ]] || log "WARN $arm: GPU KV cache size [${kv:-MISSING}] != $KV_EXPECT (G1 will FAIL; cells continue for receipts)"
  health "$arm" "$name-rtx"
  ph boot_ready "\"name\": \"$name\""
  return 0; }

alive(){ is_up "$NAME" && is_up "$NAME-rtx" && return 0; log "ARM $BOOT DEAD ($NAME up=$(is_up "$NAME" && echo y || echo n) rtx up=$(is_up "$NAME-rtx" && echo y || echo n)) — skipping remaining cells"; return 1; }
cell(){ # tag cmd... -> runs with bg, logs one line
  local tag=$1; shift; check_abort; alive || return 1
  ph cell_start "\"cell\": \"$tag\""
  bg "$@" > "$A/$tag.log" 2>&1; local rc=$?
  ph cell_end "\"cell\": \"$tag\", \"rc\": $rc"
  log "CELL $BOOT $tag rc=$rc: $(tail -1 "$A/$tag.log" | cut -c1-330)"; return 0; }
run_cells(){ local arm=$1 sinceC r
  sinceC=$(date -u +%FT%TZ); setphase "$arm:cells"; ph cells_start
  for r in $(seq 1 "$C1_REPS"); do
    cell "c1-r$r" env HELDOUT=1 python3 "$MIXED" "$arm-c1-r$r" "$A" 1 "$MIX_S" || break
  done
  cell cold64k python3 "$CELLS" cold64k --out "$A/cold64k.json" --n "$COLD_N" --tokens "$COLD_TOK"
  cell warm python3 "$CELLS" warm --out "$A/warm.json" --tokens "$WARM_TOK" --max-tokens "$WARM_MAX"
  cell needle1m python3 "$CELLS" needle --out "$A/needle1m.json" --tokens "$NEEDLE_TOK" --timeout 2700
  if [[ $arm == K1 ]]; then
    cell evict python3 "$CELLS" evict --out "$A/evict.json" --lengths "$EVICT_LENS"
    check_abort; alive && { ph cell_start '"cell": "gauntlet"'; gauntlet K1; ph cell_end '"cell": "gauntlet"'; }
  else
    cell evictctl python3 "$CELLS" evictctl --out "$A/evictctl.json" --lengths "$EVICT_LENS"
  fi
  setphase "$arm:post"; ph cells_end
  errcount "$arm" "$NAME" "$sinceC"; errcount "$arm-rtx" "$NAME-rtx" "$sinceC"
  wd_collect "$arm"; }

# ================= preflight (refusals exit 4/5/6 with the lane untouched) =================
log "===== CARD KV3 START (arms=$ARMS mix_s=$MIX_S c1_reps=$C1_REPS cold=$COLD_N x ${COLD_TOK} warm=${WARM_TOK}/${WARM_MAX} needle=$NEEDLE_TOK evict=$EVICT_LENS mem_min=$MEM_MIN) ====="
if [[ -n $WAIT_PID ]]; then log "WAIT for pid $WAIT_PID to exit"; while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done; log "pid $WAIT_PID gone"; fi
log "LANE-OWNER.txt: $(tr '\n' ' ' < "$D/LANE-OWNER.txt" 2>/dev/null | cut -c1-400)"
o=$(other_runners); [[ -n $o ]] && { log "REFUSE: another runner is live: $o"; exit 4; }
g=$(gpu_foreign); [[ -n ${g// /} ]] && { log "REFUSE foreign GPU container:$g"; exit 4; }
x=$(docker ps --format '{{.Names}}' | grep -E '^dsv41-' | grep -vx "$REF" | tr '\n' ' '); [[ -n ${x// /} ]] && { log "REFUSE: unexpected dsv41 containers running: $x"; exit 4; }
pgrep -af sidecar_watchdog.py >/dev/null && { log "REFUSE: a sidecar_watchdog is running ($(pgrep -af sidecar_watchdog.py | cut -c1-160 | tr '\n' '|')) — v26seed runs without one; whose is it?"; exit 4; }
is_up "$REF" || { log "ABORT: $REF not running — lane untouched"; exit 5; }
curl -s -m 3 $U/v1/models | grep -q $M || { log "ABORT: :30006 not bound — lane untouched"; exit 5; }
for n in $ALLN; do exists "$n" && { log "ABORT: KV3 container already exists: $n"; exit 5; }; done
docker image inspect "$IMAGE" >/dev/null 2>&1 || { log "ABORT: $IMAGE missing"; exit 5; }
docker image inspect "$GIMG" >/dev/null 2>&1 || { log "ABORT: gauntlet sandbox image missing"; exit 5; }
[[ -f $MODEL/config.json ]] || { log "ABORT: model missing $MODEL"; exit 5; }
[[ -d $SOAK ]] || { log "ABORT: $SOAK missing (restore needs it for GO/ctrl)"; exit 5; }
chk(){ [[ -s $1 ]] || { log "ABORT: missing $1"; exit 5; }; [[ -z ${2:-} || "$(sha1sum "$1" | cut -c1-8)" == "$2" ]] || { log "ABORT: sha1 $(sha1sum "$1" | cut -c1-12) != $2 for $1"; exit 5; }; }
chk "$LAUNCH" "$SHA_LAUNCH"; chk "$MIXED" "$SHA_MIXED"; chk "$WD" "$SHA_WD"; chk "$CELLS" "$SHA_CELLS"; chk "$VERDICT" "$SHA_VERDICT"
chk "$GDIR/run_suite.py"; chk "$M3HOOK/$ROWMAP"; chk "$M3HOOK/sitecustomize.py"; chk "$D/local/peer_server2_local.py"
grep -q 'VLLM_SERVER_DEV_MODE=1' "$LAUNCH" || { log "ABORT: $LAUNCH lacks VLLM_SERVER_DEV_MODE=1 (/reset_prefix_cache needed for the evict probe)"; exit 5; }
u=$(rtx_used); [[ ${u:-99999} -lt 2048 ]] || { log "REFUSE: RTX busy (${u} MiB) — v26seed is Grace-only, so who holds it?"; exit 4; }
{ for f in "$0" "$LAUNCH" "$MIXED" "$WD" "$CELLS" "$VERDICT" "$GDIR/run_suite.py" "$M3HOOK/$ROWMAP"; do echo "sha1 $(sha1sum "$f" | cut -c1-12) $f"; done
  echo "image $IMAGE"; echo "gauntlet image $GIMG"
  echo "K0/K0R EXTRA=[$KVDECL]"; echo "K1 EXTRA=[$KVDECL $TIER]"
  echo "REF args: $(docker inspect -f '{{json .Args}}' "$REF" 2>/dev/null | cut -c1-2000)"
  echo "soak dir: $(ls -la "$SOAK" 2>/dev/null | tr '\n' '|' | cut -c1-600)"; } > "$R/instruments.txt"
OOM0=$(oom_count); echo "${OOM0:-}" > "$R/oom-start.txt"
log "preflight OK: MemAvailable=$(memavail)GiB RTX=${u}MiB oom_lines=${OOM0:-unreadable} (sha1s in instruments.txt)"
log "lane before: $(docker ps --format '{{.Names}}' | grep -E '^dsv41-' | tr '\n' ' ')"
wait_quiet preflight || { log "ABORT: host never quiet; lane untouched"; exit 6; }
check_abort

# ================= lane dark: v26seed-BOUND-REF stopped-and-kept (it has no watchdog) =================
LANE_TOUCHED=1
LANE_OWNER_SAVE=$R/LANE-OWNER.before.txt; cp "$D/LANE-OWNER.txt" "$LANE_OWNER_SAVE" 2>/dev/null
echo "OWNER: cardKV3 runner pid $$ since $(date '+%F %T %Z') — lane :30006 dark (KV3 K0/K1/K0R boots); restores $REF at END. Receipts $R" > "$D/LANE-OWNER.txt"
echo "cardKV3 $(date '+%F %T %Z')" > "$D/logs/MAINTENANCE"
setphase pre; mem_sampler & SAMPLER=$!
BOOT=pre; ph lane_dark
log "LANE DARK begins (stopping $REF, kept)"
docker stop -t 60 "$REF" >/dev/null 2>&1; log "stop-and-keep $REF (running=$(docker inspect -f '{{.State.Running}}' "$REF" 2>/dev/null))"

# ================= arms =================
for arm in $ARMS; do
  case $arm in
    K0) n=$N_K0; ex="$KVDECL";;
    K1) n=$N_K1; ex="$KVDECL $TIER";;
    K0R) n=$N_K0R; ex="$KVDECL";;
    *) log "BUG unknown arm $arm"; continue;;
  esac
  if boot_kv3 "$arm" "$n" "$ex"; then
    run_cells "$arm"
  else
    log "ARM $arm FAILED TO BOOT — its cells are MISSING (verdict: UNMEASURED -> FAIL)"
    [[ -d $A ]] && wd_collect "$arm"
  fi
  stop_pair "$n"
  check_abort
done

# ================= END: restore, verdict, renames, ledger =================
restore_lane
setphase end; kill "$SAMPLER" 2>/dev/null
OOM1=$(oom_count); echo "${OOM1:-}" > "$R/oom-end.txt"
python3 "$VERDICT" "$R" > "$R/VERDICT.txt" 2> "$R/verdict.err"; vrc=$?
SUF=$(python3 -c "import json;print(json.load(open('$R/verdict.json'))['suffix'])" 2>/dev/null); SUF=${SUF:-KV3-NOVERDICT}
VL=$(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));print(d["result"],"|",d["suffix"],"|",json.dumps(dict((k,v[0]) for k,v in d["gates"].items())))' "$R/verdict.json" 2>&1 | tail -1 | cut -c1-900)
log "VERDICT rc=$vrc $VL"
for n in $ALLN; do exists "$n" && docker rename "$n" "$n-$SUF" 2>/dev/null && log "kept $n-$SUF (stopped)"; done
grep -hE ' (=====|CELL|BOOT|SIDECAR|SERVER|FACTS|HEALTH|GAUNTLET|WATCHDOG|KV3 watchdog|boot->ready|BOUND|FASTFAIL|DIED|TIMEOUT|ERRCOUNT|RESTORE|LANE|CTRL|VERDICT|MEM_GUARD|WARN|ABORT|REFUSE|ARM|kept|stop-and-keep) ' "$R/runner.log" > "$R/SUMMARY.txt"
{ echo; echo "## $(date '+%F %H:%M %Z') — Card KV3 full battery: Grace host prefix-KV tier (OffloadingConnector native 48 GiB) K0/K1/K0R, :30006 — $(python3 -c "import json;d=json.load(open('$R/verdict.json'));print(d['result'], d['suffix'])" 2>/dev/null)"
  cat "$R/VERDICT.txt" 2>/dev/null; echo "Receipts: $R. NOT promoted — James decides (production adoption = relaunch the serving lane with the tier flags)."; } >> "$LEDGER"
FINISHED=1
log "FINAL dsv41 running=[$(docker ps --format '{{.Names}}' | grep -E '^dsv41-' | tr '\n' ' ')] bound=$(curl -s -m 3 $U/v1/models | grep -c $M)"
log "===== CARD KV3 END (lane = $REF) ====="
