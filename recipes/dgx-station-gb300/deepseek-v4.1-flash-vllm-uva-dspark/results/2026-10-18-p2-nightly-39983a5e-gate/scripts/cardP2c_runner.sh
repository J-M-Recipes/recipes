#!/usr/bin/env bash
# cardP2c_runner.sh — Card P2c: bisect the DSpark per-class acceptance drop of config C (nightly-39983a5e + hook, replay off).
#   Armed 2026-10-18 by Milo (hermes milo) subagent claude-opus-5-5. Template: cardP2b_runner.sh (same helpers, same fixture cells).
# ARMS (one launch, this order, same window):
#   A  = live v23 REF (dsv41-vllm-1017-v23-pinexact-BOUND-REF), no reboot                               [CTL1]
#   C  = docker start dsv41-vllm-1017-P2b-n39983a5e-noreplay-EXP-FAIL (store-loaded; second boot of config C)
#   C1 = fresh boot launch_p2c.sh VARIANT=norecord  (config C + #56893 gate forced False via :ro bind-mount of the image's own
#        attention.py, 1 line) -> container dsv41-vllm-1017-P2c-noreplay-norecord-EXP
#   C2 = fresh boot launch_p2c.sh VARIANT=draftsparse (config C + drafter attention_backend FLASHMLA_SPARSE_DSV41; launch-flag probe,
#        desk diff found no second Python-only single-file revert) -> container dsv41-vllm-1017-P2c-noreplay-draftsparse-EXP
#   D  = docker start v23 REF and LEFT SERVING                                                          [CTL2]
# CELLS per arm (identical order every arm, = P2b cells): RT C1x3 (kbench real-text; decode_agg + median TTFT = "TTFT 8K"),
#   agent_fixture_x3.py (8 prompts/class, fixed order), agent_fixture_o.sh x3 with the P2b CAT_ORDER rotations.
# PRE-REGISTERED BAR (CTL = mean of A and D; evaluated by cardP2c_verdict.py): an arm RECOVERS iff fixx3 code AND structured acceptance
#   are within 2.0 pts of CTL AND RT C1 >= CTL − 2% AND TTFT 8K <= 0.45 s. Mechanism checks: C1 in-container gate False + KV tokens moved;
#   C2 no "Using the target model's ... attention backend for the DeepSeek-V4 DSpark drafter" line; all nightly arms no replay line, rehome done.
# NO PROMOTION. Stop-and-keep only; verdict suffixes on C1/C2. Never :30003, never glmf-*/dsf*. Refuses foreign GPU containers / other runners.
set -uo pipefail
D=$HOME/dsv41/m3; W=$HOME/dsv41/1017; R=$W/results-cardP2c; mkdir -p "$R"
BK=$D/bench/kbench_chat_ds.py; P=$D/bench/k3pool/long; U=http://127.0.0.1:30006; M=dsv41-flash-uva
REF=dsv41-vllm-1017-v23-pinexact-BOUND-REF
CNAME=dsv41-vllm-1017-P2b-n39983a5e-noreplay-EXP-FAIL
C1NAME=dsv41-vllm-1017-P2c-noreplay-norecord-EXP
C2NAME=dsv41-vllm-1017-P2c-noreplay-draftsparse-EXP
LAUNCH=$W/launch_p2c.sh
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
other_runners(){ ps -eo pid,args | grep -E '^ *[0-9]+ bash .*runner\.sh' | grep -v "cardP2c_runner.sh" | grep -v grep | cut -c1-120 | tr '\n' '|'; }
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
  tr '\r' '\n' < "$R/boot-$tag.log" | grep -E "Decoder SWA bounded replay|SWA bounded replay|Using V[12] Model Runner|KV cache format|data type to store kv cache|attention backend for the DeepSeek-V4 DSpark drafter|Using .* backend|Using FlashInfer (managed )?autotune cache|Managed autotune cache|Preloaded [0-9]+/[0-9]+|Autotuning process (starts|ends)|Loaded [0-9]+ configs|Saved [0-9]+ configs|GPU KV cache size|Available KV cache memory|Graph capturing finished|PIN_HOT installed|PIN_HOT abort|PIN_HOT hook FAILED|rehome layer=9 |rehome done|Model loading took|model loading took" | grep -vE 'it/s\]' | cut -c1-400 > "$R/facts-$tag.txt"
  local ts te; ts=$(grep -m1 'Autotuning process starts' "$R/facts-$tag.txt" | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1); te=$(grep 'Autotuning process ends' "$R/facts-$tag.txt" | tail -1 | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1)
  echo "TUNEWINDOW ${ts:-?} ${te:-?}" >> "$R/facts-$tag.txt"; echo "STORE_ENTRIES $(store_entries)" >> "$R/facts-$tag.txt"
  log "FACTS $tag: replay_line=$(grep -c 'Decoder SWA bounded replay:' "$R/facts-$tag.txt") drafter_target_backend_line=$(grep -c 'attention backend for the DeepSeek-V4 DSpark drafter' "$R/facts-$tag.txt") rehome=$(grep -c 'rehome done layers=40' "$R/facts-$tag.txt") | autotune: starts=$(grep -c 'Autotuning process starts' "$R/facts-$tag.txt") window=${ts:-?}->${te:-?} managed=$(grep -oE 'Preloaded [0-9]+/[0-9]+' "$R/facts-$tag.txt" | tail -1) store_entries=$(store_entries) | $(grep 'GPU KV cache size' "$R/facts-$tag.txt" | tail -1 | sed 's/.*\] //' | cut -c1-60)"; }
gate_check(){ # tag name -> in-container value of the #56893 gate + mounts
  local tag=$1 name=$2
  { echo "mounts: $(docker inspect "$name" --format '{{range .Mounts}}{{.Source}}->{{.Destination}} {{end}}' | tr ' ' '\n' | grep -c deepseek_v41/attention.py) attention.py override mount(s)"
    docker exec "$name" python3 -c "import vllm.models.deepseek_v41.attention as a; print('GATE', a._use_v41_mxfp8_kv_record())" 2>&1 | grep -E '^GATE|Error' | tail -2
  } > "$R/gate-$tag.txt" 2>&1; log "GATE $tag: $(tr '\n' ' ' < "$R/gate-$tag.txt")"; }
rt_c1(){ local tag=$1 N O r
  N="p2c-$1-$(date +%s)"; O="$R/rt-$1"; mkdir -p "$O"
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
fixtures(){ local tag=$1 k ord
  ( cd $W && TAG="P2c-$tag-x3" OUT="$R/fixx3-P2c-$tag.json" timeout 1200 python3 agent_fixture_x3.py ) > "$R/fixx3-log-$tag.txt" 2>&1 || log "fixture_x3 $tag nonzero rc"
  log "FIXX3 $tag: $(grep -oE '[a-z_]+ +accept=[0-9.]+' "$R/fixx3-log-$tag.txt" | tr -s ' ' | tr '\n' ';')"
  local ORDS=("prose structured code shell_ops tool_json" "structured code shell_ops tool_json prose" "code shell_ops tool_json prose structured")
  for k in 0 1 2; do ord=${ORDS[$k]}
    ( cd $HOME/dsv41 && CAT_ORDER="$ord" timeout 900 bash agent_fixture_o.sh "P2c-$tag-rot$k" ) > "$R/fixrot-log-$tag-rot$k.txt" 2>&1 || log "fixture_o $tag rot$k nonzero rc"
    mv -f $HOME/dsv41/agentfix-P2c-$tag-rot$k.json "$R/" 2>/dev/null
    log "FIXROT $tag rot$k [$ord]: $(grep -oE '[a-z_]+ +[0-9.]+ tok/s +accept= ?[0-9.]+%' "$R/fixrot-log-$tag-rot$k.txt" | tr -s ' ' | tr '\n' ';')"
  done; }
measure(){ # tag name since
  local tag=$1 name=$2 since=$3
  log "--- MEASURE $tag ($name) ---"
  rt_c1 "$tag"; fixtures "$tag"
  curl -s $U/metrics | grep -E '^vllm:(spec_decode|prefix_cache)' > "$R/metrics-$tag.txt" 2>/dev/null
  docker logs --since "$since" "$name" > "$R/postlog-$tag.log" 2>&1; grep -cE 'ValueError:|Traceback|ERROR|RuntimeError:' "$R/postlog-$tag.log" > "$R/errcount-$tag.txt"
  log "ERRCOUNT $tag $(cat "$R/errcount-$tag.txt") (since $since)"; log "--- END MEASURE $tag ---"; }
fresh_arm(){ # tag variant name maxmin
  local tag=$1 var=$2 name=$3 maxmin=$4 t0 ma
  wait_quiet "$tag" || log "WARN: host not quiet before $tag — proceeding"
  stop_all; drop_caches; strays
  ma=$(memavail); if awk -v m="$ma" 'BEGIN{exit !(m < 380)}'; then log "ABORT $tag: MemAvailable ${ma} GiB < 380 after stop+drop_caches (stray?)"; return 1; fi
  log "store entries before $tag boot: $(store_entries)"
  t0=$(date +%s); VARIANT=$var OFFGB=60 bash "$LAUNCH" "$name" || { log "launch failed ($tag)"; return 1; }
  log "launched $name VARIANT=$var"
  if wait_bind "$name" "$maxmin"; then log "boot->ready $tag $(( $(date +%s)-t0 ))s ($name)"; return 0; fi
  docker logs "$name" > "$R/boot-$tag-FAIL.log" 2>&1
  log "BOOTFAIL $tag tail: $(tr '\r' '\n' < "$R/boot-$tag-FAIL.log" | grep -E 'PIN_HOT abort|Error:|ERROR|EngineCore failed|OutOfMemory' | tail -3 | cut -c1-220 | tr '\n' '|')"
  docker stop "$name" >/dev/null 2>&1; docker rename "$name" "$name-BOOT-FAIL" && log "kept $name-BOOT-FAIL (stopped)"
  return 1; }

log "===== CARD P2c START (REF=$REF C=$CNAME C1=$C1NAME C2=$C2NAME) ====="
g=$(gpu_foreign); [[ -n ${g// /} ]] && { log "REFUSE foreign GPU container:$g"; exit 4; }
o=$(other_runners); [[ -n $o ]] && { log "REFUSE: another runner is live: $o"; exit 4; }
docker ps --format '{{.Names}}' | grep -qx "$REF" || { log "ABORT: $REF not running for A"; exit 5; }
curl -s -m 3 $U/v1/models | grep -q $M || { log "ABORT: REF not bound on :30006"; exit 5; }
docker ps -a --format '{{.Names}}' | grep -qx "$CNAME" || { log "ABORT: $CNAME missing"; exit 5; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || { log "ABORT: image not present"; exit 5; }
[[ -f $LAUNCH ]] && bash -n "$LAUNCH" || { log "ABORT: $LAUNCH missing or syntax"; exit 5; }
[[ -f $W/agent_fixture_x3.py && -f $W/cardP2c_verdict.py ]] || { log "ABORT: instruments missing"; exit 5; }
for n in "$C1NAME" "$C2NAME"; do docker ps -a --format '{{.Names}}' | grep -qx "$n" && { log "ABORT: $n already exists"; exit 5; }; done
log "preflight OK: MemAvailable=$(memavail)GiB store_entries=$(store_entries) sha fixx3=$(sha256sum $W/agent_fixture_x3.py | cut -c1-8) launcher=$(sha256sum $LAUNCH | cut -c1-8) override=$(sha256sum $W/p2c-overrides/attention.py.norecord | cut -c1-8)"
wait_quiet A || { log "ABORT: host never quiet; lane untouched (REF serving)"; exit 6; }

# ---- A: live v23 ----
sinceA=$(date -u +%FT%TZ)
boot_facts A "$REF" "$(docker inspect -f '{{.State.StartedAt}}' "$REF")"; strays
measure A "$REF" "$sinceA"

# ---- C: store-loaded restart of config C (second boot) ----
wait_quiet C || log "WARN: host not quiet before C — proceeding"
stop_all; drop_caches; strays
STORE0=$(store_entries); log "store entries before C start: $STORE0"
t0=$(date +%s); sinceC=$(date -u +%FT%TZ); docker start "$CNAME" >/dev/null
if wait_bind "$CNAME" 60 "$sinceC"; then
  log "boot->ready C $(( $(date +%s)-t0 ))s (store-loaded restart)"
  boot_facts C "$CNAME" "$sinceC"; gate_check C "$CNAME"; log "store entries after C bind: $(store_entries) (before $STORE0)"
  grep -q 'Decoder SWA bounded replay:' "$R/facts-C.txt" && log "WARN C: replay line PRESENT -> C INVALID"
  measure C "$CNAME" "$sinceC"
else log "C not measured (bind failed)"; docker logs --since "$sinceC" "$CNAME" > "$R/boot-C-FAIL.log" 2>&1; fi
docker ps --format '{{.Names}}' | grep -qx "$CNAME" && { docker stop "$CNAME" >/dev/null; log "stop-and-keep $CNAME"; }

# ---- C1: fresh boot, #56893 gate forced False ----
if fresh_arm C1 norecord "$C1NAME" 110; then
  boot_facts C1 "$C1NAME"; gate_check C1 "$C1NAME"; log "store entries after C1 bind: $(store_entries)"
  grep -q 'Decoder SWA bounded replay:' "$R/facts-C1.txt" && log "WARN C1: replay line PRESENT -> C1 INVALID"
  grep -q 'GATE False' "$R/gate-C1.txt" || log "WARN C1: in-container gate not False -> C1 INVALID"
  measure C1 "$C1NAME" "$(docker inspect -f '{{.State.StartedAt}}' "$C1NAME")"
  docker stop "$C1NAME" >/dev/null; log "stop-and-keep $C1NAME"
else log "C1 not measured"; fi

# ---- C2: fresh boot, drafter attention backend FLASHMLA_SPARSE_DSV41 ----
if fresh_arm C2 draftsparse "$C2NAME" 60; then
  boot_facts C2 "$C2NAME"; gate_check C2 "$C2NAME"; log "store entries after C2 bind: $(store_entries)"
  grep -q 'attention backend for the DeepSeek-V4 DSpark drafter' "$R/facts-C2.txt" && log "WARN C2: drafter still inherits target backend -> C2 INVALID"
  measure C2 "$C2NAME" "$(docker inspect -f '{{.State.StartedAt}}' "$C2NAME")"
  docker stop "$C2NAME" >/dev/null; log "stop-and-keep $C2NAME"
else log "C2 not measured"; fi

# ---- D: v23 REF back, left serving ----
wait_quiet D || log "WARN: host not quiet before D — booting REF anyway (lane must return)"
stop_all; drop_caches; strays
t0=$(date +%s); sinceD=$(date -u +%FT%TZ); docker start "$REF" >/dev/null
if wait_bind "$REF" 40 "$sinceD" nofastfail; then log "boot->ready D $(( $(date +%s)-t0 ))s"; boot_facts D "$REF" "$sinceD"; measure D "$REF" "$sinceD"
else log "D REF FAILED TO BIND — lane dark, needs hands"; fi

# ---- verdict + suffixes (no promotion) ----
python3 $W/cardP2c_verdict.py "$R" > "$R/VERDICT.md" 2> "$R/verdict.err"
log "VERDICT: $(grep -m1 '^\*\*Verdict' "$R/VERDICT.md")"
for pair in "C1:$C1NAME" "C2:$C2NAME"; do t=${pair%%:*}; n=${pair#*:}
  s=$(grep -m1 "^SUFFIX_$t " "$R/VERDICT.md" | awk '{print $2}')
  if docker ps -a --format '{{.Names}}' | grep -qx "$n"; then
    docker ps --format '{{.Names}}' | grep -qx "$n" && docker stop "$n" >/dev/null 2>&1
    docker rename "$n" "$n-${s:-DONE}" && log "kept $n-${s:-DONE} (stopped)"; fi
done
grep -hE ' (RT|FIXX3|FIXROT|FACTS|GATE|ERRCOUNT|boot->ready|BOUND|BOOTFAIL|FASTFAIL|DIED|TIMEOUT|WARN|ABORT|VERDICT|store entries|kept) ' "$R/runner.log" > "$R/SUMMARY.txt"
{ echo; echo "## $(date '+%F %H:%M %Z') — Card P2c: bisecting config C's DSpark per-class acceptance drop (A live v23 → C restart → C1 #56893 gate False → C2 drafter FLASHMLA_SPARSE_DSV41 → D v23 restart) — $(grep -m1 '^\*\*Verdict' "$R/VERDICT.md" 2>/dev/null)"
  grep -vE '^SUFFIX' "$R/VERDICT.md"; echo; echo "Runner summary lines:"; sed 's/^/    /' "$R/SUMMARY.txt"; echo "Receipts: $R (Mac copy: el8-cards-2026-10-18/P2/receipts-P2c/). Not promoted."; } >> "$LEDGER"
log "FINAL dsv41 running=[$(docker ps --format '{{.Names}}' | grep -E '^dsv41-' | tr '\n' ' ')] bound=$(curl -s -m 3 $U/v1/models | grep -c $M)"
log "===== CARD P2c END (lane = $REF) ====="
