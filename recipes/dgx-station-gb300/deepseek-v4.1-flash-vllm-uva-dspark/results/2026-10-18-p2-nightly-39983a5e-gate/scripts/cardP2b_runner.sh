#!/usr/bin/env bash
# cardP2b_runner.sh — Card P2b: isolate the two Card P2 failures (warm-replay r2 −10.9%, per-class fixture acceptance swings) on
#   nightly-39983a5e + pin-hot hook vs v23, same window, bracketed. Armed 2026-10-18 by Milo (hermes milo) subagent claude-opus-5-5.
#   Template: cardP2_runner.sh (same helpers: stop_all, drop_caches, strays, wait_quiet, file-dumped wait_bind, boot_facts, hostmem).
# ARMS (one launch, this order):
#   A = live v23 REF (dsv41-vllm-1017-v23-pinexact-BOUND-REF), no reboot                         [CTL1]
#   B = docker start dsv41-vllm-1017-P2-n39983a5e-hook-EXP-FAIL (FI managed store already filled, 210 entries) — expect ~5 min
#       bind and NO live tune; record bind time, store entries before/after, Autotuning window, Preloaded X/Y, KV tokens, replay line.
#   C = fresh boot launch_p2b_noreplay.sh = P2 launcher + --no-swa-bounded-replay (flag verified CPU-only by argparse on the image),
#       container dsv41-vllm-1017-P2b-n39983a5e-noreplay-EXP, same cache/store mount (live tune allowed; recorded and budgeted 110 min).
#       One off54 retry ONLY on a PIN_HOT host-guard death (P2 rule).
#   D = docker start v23 REF (hash hit) and LEFT SERVING                                          [CTL2]
# CELLS per arm (identical order on every arm):
#   (d) RT C1x3 (kbench_chat_ds.py real-text, same pool as P2) -> decode_agg + median TTFT (= "TTFT 8K")
#   (a) standard replay replay_c.py N=4 ROUNDS=2, r1 then r2 (unchanged v1 instrument, = P2's cell)
#   (b) forced-length replay replay_fl.py (sibling; same transcripts/SYS/TOOLS) N=4 ROUNDS=2 FL=256 max=min=256 ignore_eos,
#       per-arm cache_salt (r1 round0 cold, rest warm), r1 then r2 -> agg tok/s, TPOT, TTFT cold/warm, exact-length check
#   (c) per-class acceptance: agent_fixture_x3.py (8 prompts/class incl. the originals, fixed order prose structured code
#       shell_ops tool_json; per-prompt metrics deltas) + the standard agent_fixture_o.sh run 3x with CAT_ORDER rotations
#   (+) cold 64K probe (cold_prefill_probe.py SIZES=65536 N=2) on every arm (task asks it for C; all arms for a comparator)
# PRE-REGISTERED BARS (CTL = mean of A and D) — evaluated by cardP2b_verdict.py:
#   B is a PROMOTION CANDIDATE iff  FL replay r2 agg tok/s >= CTL − 2%  AND  every class acceptance on agent_fixture_x3
#     (token-weighted) within 2.0 pts of CTL  AND  RT C1 decode_agg (mean of 3) >= CTL − 2%.
#   else C is the candidate iff C meets the same three bars AND TTFT 8K (RT C1 median TTFT, mean of 3) <= 0.45 s.
#   else P2 CLOSES (candidate none).
#   Diagnostics (not gated): standard replay r2 on B vs CTL (does the −10.9% reproduce on a store-loaded restart = tune-luck check);
#     r2 completion-token counts per arm (do extra tokens explain it); FL TPOT; rotation-fixture mean ± spread; cold 64K.
#   Mechanism checks: B must show "Decoder SWA bounded replay:" and "rehome done layers=40"; C must show rehome and NO
#     "Decoder SWA bounded replay:" line — else that arm is INVALID.
# NO PROMOTION by this runner. Stop-and-keep only (B, C get verdict suffixes). Never :30003, never glmf-*/dsf*. Refuses foreign GPU
#   containers / other live runners. Does not touch run_window_T1T2.sh, replay_c.py, agent_fixture_o.sh (run, not edited) or
#   $HOME/pin-hot-experts/hook/.
set -uo pipefail
D=$HOME/dsv41/m3; W=$HOME/dsv41/1017; R=$W/results-cardP2b; mkdir -p "$R"
BK=$D/bench/kbench_chat_ds.py; P=$D/bench/k3pool/long; U=http://127.0.0.1:30006; M=dsv41-flash-uva
REF=dsv41-vllm-1017-v23-pinexact-BOUND-REF
BNAME=dsv41-vllm-1017-P2-n39983a5e-hook-EXP-FAIL
C0=dsv41-vllm-1017-P2b-n39983a5e-noreplay-EXP
C54=dsv41-vllm-1017-P2b-n39983a5e-noreplay-off54-EXP
CNAME=$C0
LAUNCH=$W/launch_p2b_noreplay.sh
IMAGE=vllm/vllm-openai@sha256:a3bc7f09c1e2e72e068021feb75118c92e98c538706caa0c5d1acc0bb9252623
PCACHE=$HOME/dsv41/vllm-cache-p2-39983a5e
LEDGER=$HOME/dsv41/results/ledger.md
exec >> "$R/runner.log" 2>&1
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
memavail(){ awk '/MemAvailable/{printf "%.1f",$2/1048576}' /proc/meminfo; }
drop_caches(){ sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory' 2>/dev/null || true; }
stop_all(){ local n; for n in $(docker ps --format '{{.Names}}' | grep -E '^dsv41-vllm-'); do log "stop-and-keep $n"; docker stop "$n" >/dev/null; done; sleep 5; }
gpu_foreign(){ local c out=""; for c in $(docker ps --format '{{.Names}}' | grep -vE '^dsv41-vllm-'); do
    docker inspect "$c" --format '{{json .HostConfig.DeviceRequests}}' 2>/dev/null | grep -qi gpu && out="$out $c"; done; echo "$out"; }
any_foreign(){ docker ps --format '{{.Names}}' | grep -vE '^dsv41-vllm-' | tr '\n' ' '; }
other_runners(){ ps -eo pid,args | grep -E '^ *[0-9]+ bash .*runner\.sh' | grep -v "cardP2b_runner.sh" | grep -v grep | cut -c1-120 | tr '\n' '|'; }
archive_busy(){ ps -eo args | grep -E 'ark_bf1017\.sh|archive_1017\.sh|archive_sums|rsync .*miloark|sha256sum|hf download|docker pull' | grep -v grep | head -3 | cut -c1-120 | tr '\n' '|'; }
wait_quiet(){ local stage=$1 i a f
  for ((i=0;i<60;i++)); do
    a=$(archive_busy); f=$(any_foreign)
    [[ -z $a && -z ${f// /} ]] && { [[ $i -gt 0 ]] && log "QUIET for $stage after $((i*30))s"; return 0; }
    (( i % 10 == 0 )) && log "WAIT $stage: archive=[${a}] foreign=[${f}]"
    sleep 30
  done; log "WAIT $stage: still busy after 30 min"; return 1; }
strays(){ log "strays: $(ps -eo pid,rss,args --sort=-rss | sed -n 2,5p | awk '{printf "%s %.1fGiB %s %s|", $1, $2/1048576, $3, $4}') MemAvailable=$(memavail)GiB"; }
store_entries(){ find "$PCACHE/fi-autotune-v2" -path '*/entries/*.json' 2>/dev/null | wc -l; }
hostmem(){ # tag phase [container]
  local tag=$1 ph=$2 c=${3:-} f="$R/hostmem-$1-$2.txt" id cg pid
  { echo "date $(date '+%F %T %Z') tag=$tag phase=$ph container=${c:--}"
    grep -E '^(MemTotal|MemFree|MemAvailable|Cached|Shmem|Mlocked|Unevictable|AnonPages|Mapped):' /proc/meminfo | awk '{printf "meminfo %s %.2f GiB\n",$1,$2/1048576}'
    if [[ -n $c ]] && docker ps --format '{{.Names}}' | grep -qx "$c"; then
      pid=$(docker top "$c" -eo pid,rss,args 2>/dev/null | grep 'VLLM::EngineCore' | awk '{print $1}' | head -1)
      if [[ -n $pid ]]; then
        echo "enginecore_pid $pid"
        sudo -n awk '/^[0-9a-f]+-[0-9a-f]+ /{dz=($0 ~ /\/dev\/zero/); if(dz)n++} dz&&/^Size:/{s+=$2} dz&&/^Rss:/{r+=$2} END{printf "pinned_devzero maps=%d size_gib=%.2f rss_gib=%.2f\n",n,s/1048576,r/1048576}' /proc/$pid/smaps
      fi
    fi
  } > "$f" 2>&1
  log "HOSTMEM $tag $ph: $(grep -E 'MemAvailable|pinned_devzero' "$f" | tr '\n' ';' | tr -s ' ' | cut -c1-240)"; }
# wait_bind: dump logs to a FILE then grep (never `docker logs | grep -q` under pipefail). Fast-fail only on hard errors.
wait_bind(){ # name maxmin [since] [nofastfail]
  local n=$1 maxmin=$2 since=${3:-} nff=${4:-} i t0 lf=$R/.wb-$1.log; t0=$(date +%s)
  for ((i=0;i<maxmin*6;i++)); do
    curl -s -m 2 $U/v1/models 2>/dev/null | grep -q "$M" && { log "BOUND $n after $(( $(date +%s)-t0 ))s"; return 0; }
    docker ps --format '{{.Names}}' | grep -qx "$n" || { log "DIED $n after $(( $(date +%s)-t0 ))s"; return 1; }
    if [[ -n $since ]]; then docker logs --since "$since" "$n" > "$lf" 2>&1; else docker logs "$n" > "$lf" 2>&1; fi
    if [[ -z $nff ]] && grep -qE 'ValueError:|Traceback|ERROR|RuntimeError:' "$lf"; then
      log "FASTFAIL $n after $(( $(date +%s)-t0 ))s: $(tr '\r' '\n' < "$lf" | grep -E 'ValueError:|Traceback|ERROR|RuntimeError:' | tail -3 | cut -c1-240 | tr '\n' '|')"; return 1; fi
    (( i % 30 == 0 )) && log "  $n waiting $((i/6)) min: store_entries=$(store_entries) MemAvail=$(memavail)GiB last=[$(tr '\r' '\n' < "$lf" | grep -E 'rehome|Autotun|Preloaded|Capturing|KV cache size|Decoder SWA|Loading safetensors|Loading weights' | tail -1 | cut -c1-160)]"
    sleep 10
  done; log "TIMEOUT $n ($maxmin min)"; return 1; }
boot_facts(){ # tag name [since]
  local tag=$1 name=$2 since=${3:-}
  if [[ -n $since ]]; then docker logs --since "$since" "$name" > "$R/boot-$tag.log" 2>&1; else docker logs "$name" > "$R/boot-$tag.log" 2>&1; fi
  tr '\r' '\n' < "$R/boot-$tag.log" | grep -E "Decoder SWA bounded replay|SWA bounded replay|Using V[12] Model Runner|Auto-enabling VLLM_USE_BREAKABLE|KV cache format|data type to store kv cache|Using FlashInfer (managed )?autotune cache|Managed autotune cache|Preloaded [0-9]+/[0-9]+|Autotuning process (starts|ends)|Loaded [0-9]+ configs|Saved [0-9]+ configs|GPU KV cache size|Available KV cache memory|Graph capturing finished|PIN_HOT installed|PIN_HOT abort|PIN_HOT hook FAILED|rehome layer=9 |rehome done|Model loading took|model loading took|Initializing a V1 LLM engine" | grep -vE 'it/s\]' | cut -c1-400 > "$R/facts-$tag.txt"
  local ts te; ts=$(grep -m1 'Autotuning process starts' "$R/facts-$tag.txt" | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1); te=$(grep 'Autotuning process ends' "$R/facts-$tag.txt" | tail -1 | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1)
  echo "TUNEWINDOW ${ts:-?} ${te:-?}" >> "$R/facts-$tag.txt"; echo "STORE_ENTRIES $(store_entries)" >> "$R/facts-$tag.txt"
  log "FACTS $tag: replay_line=$(grep -c 'Decoder SWA bounded replay:' "$R/facts-$tag.txt") replay_off=$(grep -c 'bounded replay is off\|needs model runner V2' "$R/facts-$tag.txt") rehome=$(grep -c 'rehome done layers=40' "$R/facts-$tag.txt") | autotune: Loaded=$(grep -oE 'Loaded [0-9]+ configs' "$R/facts-$tag.txt" | tail -1) starts=$(grep -c 'Autotuning process starts' "$R/facts-$tag.txt") window=${ts:-?}->${te:-?} managed=$(grep -oE 'Preloaded [0-9]+/[0-9]+' "$R/facts-$tag.txt" | tail -1) store_entries=$(store_entries) | $(grep 'GPU KV cache size' "$R/facts-$tag.txt" | tail -1 | sed 's/.*\] //' | cut -c1-60) | $(grep 'rehome done' "$R/facts-$tag.txt" | tail -1 | sed 's/.*rehome done //' | cut -c1-140)"; }
rt_c1(){ local tag=$1 N O r
  N="p2b-$1-$(date +%s)"; O="$R/rt-$1"; mkdir -p "$O"
  for r in 1 2 3; do python3 $BK $U $M $P/eval-c1-r$r-try1.jsonl $O/c1-r$r.json 1 --warm $P/eval-c1-r$r-try1-warm.jsonl --nonce $N-c1-r$r > "$O/c1-r$r.stdout" 2>&1; done
  log "RT $tag: $(python3 - "$O" <<'PY'
import json,sys
O=sys.argv[1]; out=[]
for k in ('c1-r1','c1-r2','c1-r3'):
    try:
        s=json.load(open(f'{O}/{k}.json'))['summary']; out.append(f"{k} agg={s['decode_agg']} ttft={s['median_ttft_s']} acc={s.get('acceptance_len')}")
    except Exception as e: out.append(f'{k} ERR {e}')
print(' ; '.join(out))
PY
)"; }
std_replay(){ local tag=$1 r
  for r in r1 r2; do
    ( cd $HOME/dsv41 && env TAG="P2b-$tag-$r" N=4 ROUNDS=2 timeout 1200 python3 replay_c.py ) > "$R/replayc-log-$tag-$r.txt" 2>&1 || log "replay_c $tag $r nonzero rc"
    mv -f $HOME/dsv41/replayc-P2b-$tag-$r-n4.json "$R/" 2>/dev/null
    log "REPLAY-STD $tag $r: $(tail -1 "$R/replayc-log-$tag-$r.txt" | cut -c1-260)"
  done; }
fl_replay(){ local tag=$1 r salt="p2b-$1-$(date +%s)"
  for r in r1 r2; do
    ( cd $W && TAG="P2b-$tag-FL-$r" OUT="$R/replayfl-P2b-$tag-$r.json" SALT="$salt" N=4 ROUNDS=2 FL=256 timeout 1200 python3 replay_fl.py ) > "$R/replayfl-log-$tag-$r.txt" 2>&1 || log "replay_fl $tag $r nonzero rc"
    log "REPLAY-FL $tag $r: $(tail -1 "$R/replayfl-log-$tag-$r.txt" | cut -c1-300)"
  done; }
fixtures(){ local tag=$1 k ord
  ( cd $W && TAG="P2b-$tag-x3" OUT="$R/fixx3-P2b-$tag.json" timeout 1200 python3 agent_fixture_x3.py ) > "$R/fixx3-log-$tag.txt" 2>&1 || log "fixture_x3 $tag nonzero rc"
  log "FIXX3 $tag: $(grep -oE '[a-z_]+ +accept=[0-9.]+' "$R/fixx3-log-$tag.txt" | tr -s ' ' | tr '\n' ';')"
  local ORDS=("prose structured code shell_ops tool_json" "structured code shell_ops tool_json prose" "code shell_ops tool_json prose structured")
  for k in 0 1 2; do ord=${ORDS[$k]}
    ( cd $HOME/dsv41 && CAT_ORDER="$ord" timeout 900 bash agent_fixture_o.sh "P2b-$tag-rot$k" ) > "$R/fixrot-log-$tag-rot$k.txt" 2>&1 || log "fixture_o $tag rot$k nonzero rc"
    mv -f $HOME/dsv41/agentfix-P2b-$tag-rot$k.json "$R/" 2>/dev/null
    log "FIXROT $tag rot$k [$ord]: $(grep -oE '[a-z_]+ +[0-9.]+ tok/s +accept= ?[0-9.]+%' "$R/fixrot-log-$tag-rot$k.txt" | tr -s ' ' | tr '\n' ';')"
  done; }
cold64(){ local tag=$1
  BASE_URL=$U/v1 MODEL=$M API_KEY=none SIZES="65536" N=2 THINKING=0 timeout 900 python3 $HOME/dsv41/cold_prefill_probe.py > "$R/cold-$tag.txt" 2>&1 || log "cold $tag nonzero rc"
  log "COLD64K $tag: $(grep PREFILL "$R/cold-$tag.txt" | cut -c1-200 | tr '\n' ';')"; }
measure(){ # tag name since
  local tag=$1 name=$2 since=$3
  log "--- MEASURE $tag ($name) ---"
  rt_c1 "$tag"; std_replay "$tag"; fl_replay "$tag"; fixtures "$tag"; cold64 "$tag"
  curl -s $U/metrics | grep -E '^vllm:(spec_decode|prefix_cache)' > "$R/metrics-$tag.txt" 2>/dev/null
  hostmem "$tag" steady "$name"
  docker logs --since "$since" "$name" > "$R/postlog-$tag.log" 2>&1; grep -cE 'ValueError:|Traceback|ERROR|RuntimeError:' "$R/postlog-$tag.log" > "$R/errcount-$tag.txt"
  log "ERRCOUNT $tag $(cat "$R/errcount-$tag.txt") (since $since)"; log "--- END MEASURE $tag ---"; }
launch_c(){ # offgb name -> 0 bound, 1 fail, 7 host-guard death
  local off=$1 name=$2 t0 ma
  wait_quiet "C-off$off" || return 1
  stop_all; drop_caches; strays; hostmem "C-off$off" pre ""
  ma=$(memavail); if awk -v m="$ma" 'BEGIN{exit !(m < 380)}'; then log "ABORT C off$off: MemAvailable ${ma} GiB < 380 after stop+drop_caches (stray?)"; return 1; fi
  log "store entries before C boot: $(store_entries)"
  t0=$(date +%s); OFFGB=$off bash "$LAUNCH" "$name" || { log "launch failed (off$off)"; return 1; }
  log "launched $name off$off image ${IMAGE##*@} (--no-swa-bounded-replay)"
  if wait_bind "$name" 110; then log "boot->ready C $(( $(date +%s)-t0 ))s (off$off, $name)"; return 0; fi
  docker logs "$name" > "$R/boot-C-off$off-FAIL.log" 2>&1
  local hg; hg=$(tr '\r' '\n' < "$R/boot-C-off$off-FAIL.log" | grep -c 'PIN_HOT abort.*MemAvailable')
  log "BOOTFAIL C off$off host_guard_hits=$hg tail: $(tr '\r' '\n' < "$R/boot-C-off$off-FAIL.log" | grep -E 'PIN_HOT abort|Error:|ERROR|EngineCore failed|OutOfMemory' | tail -3 | cut -c1-220 | tr '\n' '|')"
  docker stop "$name" >/dev/null 2>&1; docker rename "$name" "$name-BOOT-FAIL" && log "kept $name-BOOT-FAIL (stopped)"
  (( hg > 0 )) && return 7; return 1; }

log "===== CARD P2b START (REF=$REF B=$BNAME C=$C0) ====="
g=$(gpu_foreign); [[ -n ${g// /} ]] && { log "REFUSE foreign GPU container:$g"; exit 4; }
o=$(other_runners); [[ -n $o ]] && { log "REFUSE: another runner is live: $o"; exit 4; }
docker ps --format '{{.Names}}' | grep -qx "$REF" || { log "ABORT: $REF not running for A"; exit 5; }
curl -s -m 3 $U/v1/models | grep -q $M || { log "ABORT: REF not bound on :30006"; exit 5; }
docker ps -a --format '{{.Names}}' | grep -qx "$BNAME" || { log "ABORT: $BNAME missing"; exit 5; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || { log "ABORT: image not present"; exit 5; }
[[ -f $LAUNCH ]] && bash -n "$LAUNCH" || { log "ABORT: $LAUNCH missing or syntax"; exit 5; }
[[ -f $W/replay_fl.py && -f $W/agent_fixture_x3.py && -f $W/cardP2b_verdict.py ]] || { log "ABORT: P2b instruments missing"; exit 5; }
for n in "$C0" "$C54"; do docker ps -a --format '{{.Names}}' | grep -qx "$n" && { log "ABORT: $n already exists"; exit 5; }; done
log "preflight OK: MemAvailable=$(memavail)GiB store_entries=$(store_entries) sha replay_fl=$(sha256sum $W/replay_fl.py | cut -c1-8) fixx3=$(sha256sum $W/agent_fixture_x3.py | cut -c1-8) launcher=$(sha256sum $LAUNCH | cut -c1-8)"
wait_quiet A || { log "ABORT: host never quiet; lane untouched (REF serving)"; exit 6; }

# ---- A: live v23 ----
sinceA=$(date -u +%FT%TZ)
boot_facts A "$REF" "$(docker inspect -f '{{.State.StartedAt}}' "$REF")"; strays; hostmem A live "$REF"
measure A "$REF" "$sinceA"

# ---- B: store-loaded restart of the P2 CAND ----
wait_quiet B || log "WARN: host not quiet before B — proceeding"
stop_all; drop_caches; strays; hostmem B pre ""
STORE_B0=$(store_entries); log "store entries before B start: $STORE_B0"
t0=$(date +%s); sinceB=$(date -u +%FT%TZ); docker start "$BNAME" >/dev/null
if wait_bind "$BNAME" 60 "$sinceB"; then
  log "boot->ready B $(( $(date +%s)-t0 ))s (store-loaded restart)"
  boot_facts B "$BNAME" "$sinceB"; hostmem B bound "$BNAME"; log "store entries after B bind: $(store_entries) (before $STORE_B0)"
  grep -q 'Decoder SWA bounded replay:' "$R/facts-B.txt" || log "WARN B: replay boot line ABSENT -> B INVALID"
  grep -q 'rehome done layers=40' "$R/facts-B.txt" || log "WARN B: hook did NOT re-home -> B INVALID"
  measure B "$BNAME" "$sinceB"
else log "B not measured (bind failed)"; docker logs --since "$sinceB" "$BNAME" > "$R/boot-B-FAIL.log" 2>&1; fi

# ---- C: fresh boot, --no-swa-bounded-replay ----
launch_c 60 "$C0"; rc=$?
if (( rc == 7 )); then log "host guard death at off60 -> retry off54"; CNAME=$C54; launch_c 54 "$C54"; rc=$?; fi
if (( rc == 0 )); then
  boot_facts C "$CNAME"; hostmem C bound "$CNAME"; log "store entries after C bind: $(store_entries)"
  grep -q 'Decoder SWA bounded replay:' "$R/facts-C.txt" && log "WARN C: replay boot line PRESENT -> flag ineffective -> C INVALID"
  grep -q 'rehome done layers=40' "$R/facts-C.txt" || log "WARN C: hook did NOT re-home -> C INVALID"
  measure C "$CNAME" "$(docker inspect -f '{{.State.StartedAt}}' "$CNAME")"
else log "C not measured (rc=$rc)"; fi

# ---- D: v23 REF back, left serving ----
wait_quiet D || log "WARN: host not quiet before D — booting REF anyway (lane must return)"
stop_all; drop_caches; strays
t0=$(date +%s); sinceD=$(date -u +%FT%TZ); docker start "$REF" >/dev/null
if wait_bind "$REF" 40 "$sinceD" nofastfail; then log "boot->ready D $(( $(date +%s)-t0 ))s"; boot_facts D "$REF" "$sinceD"; hostmem D bound "$REF"; measure D "$REF" "$sinceD"
else log "D REF FAILED TO BIND — lane dark, needs hands"; fi

# ---- verdict + suffixes (no promotion) ----
python3 $W/cardP2b_verdict.py "$R" > "$R/VERDICT.md" 2> "$R/verdict.err"
SB=$(grep -m1 '^SUFFIX_B' "$R/VERDICT.md" | awk '{print $2}'); SC=$(grep -m1 '^SUFFIX_C' "$R/VERDICT.md" | awk '{print $2}')
log "VERDICT: $(grep -m1 '^\*\*Verdict' "$R/VERDICT.md")"
if docker ps -a --format '{{.Names}}' | grep -qx "$BNAME"; then
  docker ps --format '{{.Names}}' | grep -qx "$BNAME" && docker stop "$BNAME" >/dev/null 2>&1
  docker rename "$BNAME" "dsv41-vllm-1017-P2-n39983a5e-hook-EXP-P2b-${SB:-DONE}" && log "kept dsv41-vllm-1017-P2-n39983a5e-hook-EXP-P2b-${SB:-DONE} (stopped)"
fi
if docker ps -a --format '{{.Names}}' | grep -qx "$CNAME"; then
  docker ps --format '{{.Names}}' | grep -qx "$CNAME" && docker stop "$CNAME" >/dev/null 2>&1
  docker rename "$CNAME" "$CNAME-${SC:-DONE}" && log "kept $CNAME-${SC:-DONE} (stopped)"
fi
grep -hE ' (RT|REPLAY-STD|REPLAY-FL|FIXX3|FIXROT|COLD64K|FACTS|HOSTMEM|ERRCOUNT|boot->ready|BOUND|BOOTFAIL|FASTFAIL|DIED|TIMEOUT|WARN|ABORT|VERDICT|store entries) ' "$R/runner.log" > "$R/SUMMARY.txt"
{ echo; echo "## $(date '+%F %H:%M %Z') — Card P2b: isolating the P2 failures (nightly-39983a5e+hook; A live v23 → B store-loaded P2 CAND → C --no-swa-bounded-replay → D v23 restart) — $(grep -m1 '^\*\*Verdict' "$R/VERDICT.md" 2>/dev/null)"
  grep -vE '^SUFFIX' "$R/VERDICT.md"; echo; echo "Runner summary lines:"; sed 's/^/    /' "$R/SUMMARY.txt"; echo "Receipts: $R (Mac copy: el8-cards-2026-10-18/P2/receipts-P2b/). Not promoted."; } >> "$LEDGER"
log "FINAL dsv41 running=[$(docker ps --format '{{.Names}}' | grep -E '^dsv41-' | tr '\n' ' ')] bound=$(curl -s -m 3 $U/v1/models | grep -c $M)"
log "===== CARD P2b END (lane = $REF) ====="
