#!/usr/bin/env bash
# cardP2e_runner.sh — Card P2e: is config C's DSpark code/structured acceptance drop the TARGET-side MegaAttn fused o-proj FP8 cast?
#   Armed 2026-10-18 by Milo (hermes milo) subagent claude-opus-5-5. Template: cardP2c_runner.sh (same helpers, same fixture cells,
#   same CAT_ORDER rotations) + P2/P2b window/cold64/gauntlet cells.
# ARMS (one launch, this order, same window):
#   A = live v23 REF (dsv41-vllm-1017-v23-pinexact-BOUND-REF), no reboot                                   [CTL1]
#   E = fresh boot launch_p2e_sparse.sh (config C with target AND drafter attention FLASHMLA_SPARSE_DSV41, fp8_ds_mla,
#       --no-swa-bounded-replay, same store) -> dsv41-vllm-1017-P2e-noreplay-sparse-EXP. New autotune hash expected; bind budget 110 min.
#       One off54 retry ONLY on a PIN_HOT host-guard death.
#   D = docker start v23 REF and LEFT SERVING                                                               [CTL2]
# CELLS per arm (identical order every arm): RT C1x3 (kbench real-text; decode_agg + median TTFT = "TTFT 8K"),
#   agent_fixture_x3.py (8 prompts/class), agent_fixture_o.sh x3 (P2b/P2c CAT_ORDER rotations), then run_window_T1T2.sh
#   (knee r1/r2, knee6, fixture, replay r1/r2), cold 64K probe (cold_prefill_probe.py SIZES=65536 N=2), gauntlet (round7, 12 tasks).
# PRE-REGISTERED BARS (CTL = mean of A and D; evaluated by cardP2e_verdict.py):
#   RECOVERS iff fixx3 code AND structured acceptance within 2.0 pts of CTL.
#   P2f candidate-quality iff RECOVERS AND RT C1 >= CTL-2% AND replay r2 >= CTL-2% AND TTFT 8K <= 0.45 s AND gauntlet 12/12.
# Mechanism checks on E: no "Decoder SWA bounded replay:" line, "rehome done layers=40", no MegaAttn in facts, KV format line fp8_ds_mla.
# NO PROMOTION. Stop-and-keep only; verdict suffix on E. Never :30003, never glmf-*/dsf*. Refuses foreign GPU containers / other runners.
set -uo pipefail
D=$HOME/dsv41/m3; W=$HOME/dsv41/1017; R=$W/results-cardP2e; mkdir -p "$R"
BK=$D/bench/kbench_chat_ds.py; P=$D/bench/k3pool/long; U=http://127.0.0.1:30006; M=dsv41-flash-uva
REF=dsv41-vllm-1017-v23-pinexact-BOUND-REF
ENAME=dsv41-vllm-1017-P2e-noreplay-sparse-EXP
E54=dsv41-vllm-1017-P2e-noreplay-sparse-off54-EXP
LAUNCH=$W/launch_p2e_sparse.sh
IMAGE=vllm/vllm-openai@sha256:a3bc7f09c1e2e72e068021feb75118c92e98c538706caa0c5d1acc0bb9252623
PCACHE=$HOME/dsv41/vllm-cache-p2-39983a5e
GDIR=$HOME/pin-hot-experts/round7/gauntlet
GIMG=sha256:00d577a6a63281e15336029d5bcee4e9a2cf182214a4f20ba6111b1c8e79893d
LEDGER=$HOME/dsv41/results/ledger.md
exec >> "$R/runner.log" 2>&1
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
memavail(){ awk '/MemAvailable/{printf "%.1f",$2/1048576}' /proc/meminfo; }
drop_caches(){ sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory' 2>/dev/null || true; }
stop_all(){ local n; for n in $(docker ps --format '{{.Names}}' | grep -E '^dsv41-vllm-'); do log "stop-and-keep $n"; docker stop "$n" >/dev/null; done; sleep 5; }
gpu_foreign(){ local c out=""; for c in $(docker ps --format '{{.Names}}' | grep -vE '^dsv41-vllm-'); do
    docker inspect "$c" --format '{{json .HostConfig.DeviceRequests}}' 2>/dev/null | grep -qi gpu && out="$out $c"; done; echo "$out"; }
any_foreign(){ docker ps --format '{{.Names}}' | grep -vE '^dsv41-vllm-' | tr '\n' ' '; }
other_runners(){ ps -eo pid,args | grep -E '^ *[0-9]+ bash .*runner\.sh' | grep -v "cardP2e_runner.sh" | grep -v grep | cut -c1-120 | tr '\n' '|'; }
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
store_keys(){ find "$PCACHE/fi-autotune-v2" -mindepth 1 -maxdepth 3 -type d 2>/dev/null | sort; }
hostmem(){ # tag phase [container]
  local tag=$1 ph=$2 c=${3:-} f="$R/hostmem-$1-$2.txt" pid
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
  tr '\r' '\n' < "$R/boot-$tag.log" | grep -E "Decoder SWA bounded replay|SWA bounded replay|Using V[12] Model Runner|KV cache format|data type to store kv cache|attention backend for the DeepSeek-V4 DSpark drafter|Using .* backend|MEGA_ATTN|MegaAttn|SPARSE_DSV41|FlashMLAAttention|Using FlashInfer (managed )?autotune cache|Managed autotune cache|Preloaded [0-9]+/[0-9]+|Autotuning process (starts|ends)|Loaded [0-9]+ configs|Saved [0-9]+ configs|GPU KV cache size|Available KV cache memory|Setting kv cache block size|Graph capturing finished|PIN_HOT installed|PIN_HOT abort|PIN_HOT hook FAILED|rehome layer=9 |rehome done|Model loading took|model loading took" | grep -vE 'it/s\]|non-default args|Initializing a V1 LLM engine' | cut -c1-400 > "$R/facts-$tag.txt"
  local ts te; ts=$(grep -m1 'Autotuning process starts' "$R/facts-$tag.txt" | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1); te=$(grep 'Autotuning process ends' "$R/facts-$tag.txt" | tail -1 | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1)
  echo "TUNEWINDOW ${ts:-?} ${te:-?}" >> "$R/facts-$tag.txt"; echo "STORE_ENTRIES $(store_entries)" >> "$R/facts-$tag.txt"
  # the full config line carries attention_config / cache_dtype; keep only those fields
  tr '\r' '\n' < "$R/boot-$tag.log" | grep -m1 -oE "attention_config=[^)]*\)|AttentionConfig\([^)]*\)" | cut -c1-300 | sed 's/^/CFG /' >> "$R/facts-$tag.txt"
  tr '\r' '\n' < "$R/boot-$tag.log" | grep -m1 -oE "kv_cache_dtype=[^,]*|cache_dtype=[^,]*" | sed 's/^/CFG /' >> "$R/facts-$tag.txt"
  log "FACTS $tag: replay_line=$(grep -c 'Decoder SWA bounded replay:' "$R/facts-$tag.txt") drafter_inherit_line=$(grep -c 'attention backend for the DeepSeek-V4 DSpark drafter' "$R/facts-$tag.txt") mega_lines=$(grep -ciE 'MEGA_ATTN|MegaAttn' "$R/facts-$tag.txt") sparse_lines=$(grep -c 'SPARSE_DSV41' "$R/facts-$tag.txt") kvfmt=[$(grep -oE "Using DeepSeek's [a-z0-9_]+ KV cache format" "$R/facts-$tag.txt" | sort -u | tr '\n' ' ')] rehome=$(grep -c 'rehome done layers=40' "$R/facts-$tag.txt") | autotune: starts=$(grep -c 'Autotuning process starts' "$R/facts-$tag.txt") window=${ts:-?}->${te:-?} managed=$(grep -oE 'Preloaded [0-9]+/[0-9]+' "$R/facts-$tag.txt" | tail -1) saved=$(grep -oE 'Saved [0-9]+ configs' "$R/facts-$tag.txt" | tail -1) store_entries=$(store_entries) | $(grep 'GPU KV cache size' "$R/facts-$tag.txt" | tail -1 | sed 's/.*\] //' | cut -c1-60) | guard: $(grep 'rehome layer=9 ' "$R/facts-$tag.txt" | tail -1 | grep -oE 'host_avail=[0-9.]+GiB' | head -1) done:$(grep 'rehome done' "$R/facts-$tag.txt" | tail -1 | grep -oE 'host_avail=[0-9.]+GiB' | head -1)"; }
rt_c1(){ local tag=$1 N O r
  N="p2e-$1-$(date +%s)"; O="$R/rt-$1"; mkdir -p "$O"
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
  ( cd $W && TAG="P2e-$tag-x3" OUT="$R/fixx3-P2e-$tag.json" timeout 1200 python3 agent_fixture_x3.py ) > "$R/fixx3-log-$tag.txt" 2>&1 || log "fixture_x3 $tag nonzero rc"
  log "FIXX3 $tag: $(grep -oE '[a-z_]+ +accept=[0-9.]+' "$R/fixx3-log-$tag.txt" | tr -s ' ' | tr '\n' ';')"
  local ORDS=("prose structured code shell_ops tool_json" "structured code shell_ops tool_json prose" "code shell_ops tool_json prose structured")
  for k in 0 1 2; do ord=${ORDS[$k]}
    ( cd $HOME/dsv41 && CAT_ORDER="$ord" timeout 900 bash agent_fixture_o.sh "P2e-$tag-rot$k" ) > "$R/fixrot-log-$tag-rot$k.txt" 2>&1 || log "fixture_o $tag rot$k nonzero rc"
    mv -f $HOME/dsv41/agentfix-P2e-$tag-rot$k.json "$R/" 2>/dev/null
    log "FIXROT $tag rot$k [$ord]: $(grep -oE '[a-z_]+ +[0-9.]+ tok/s +accept= ?[0-9.]+%' "$R/fixrot-log-$tag-rot$k.txt" | tr -s ' ' | tr '\n' ';')"
  done; }
window(){ local tag=$1
  ( timeout 2400 bash $HOME/dsv41/run_window_T1T2.sh "P2e-$tag" ) > "$R/nohup-$tag.log" 2>&1 || log "window $tag nonzero rc"
  mv $HOME/dsv41/overnight-T1T2/*"P2e-$tag"[-.]* "$R"/ 2>/dev/null
  log "WINDOW $tag: $(python3 - "$R" "$tag" <<'PY'
import json,sys
R,t=sys.argv[1],sys.argv[2]; o=[]
for r in ('r1','r2'):
    try:
        rows=json.load(open(f'{R}/knee-P2e-{t}-{r}.json'))['rows']; d={}
        for x in rows:
            c,a=(x['conc'],x.get('agg')) if isinstance(x,dict) else (x[0],x[1]); d[int(c)]=a
        o.append(f"knee {r} C1={d.get(1,0):.1f} C8={d.get(8,0):.1f} C16={d.get(16,0):.1f}")
    except Exception as e: o.append(f'knee {r} ERR {e}')
for r in ('r1','r2'):
    try: d=json.load(open(f'{R}/replayc-P2e-{t}-{r}-n4.json')); o.append(f"replay {r} {d.get('agg_tok_s')} acc {d.get('accept')} tok {d.get('tokens')}")
    except Exception as e: o.append(f'replay {r} ERR {e}')
print(' ; '.join(o))
PY
)"; }
cold64(){ local tag=$1
  BASE_URL=$U/v1 MODEL=$M API_KEY=none SIZES="65536" N=2 THINKING=0 timeout 900 python3 $HOME/dsv41/cold_prefill_probe.py > "$R/cold-$tag.txt" 2>&1 || log "cold $tag nonzero rc"
  log "COLD64K $tag: $(grep PREFILL "$R/cold-$tag.txt" | cut -c1-200 | tr '\n' ';')"; }
gauntlet(){ local tag=$1
  rm -rf "$R/gauntlet-$tag"
  (cd "$GDIR" && timeout 1200 python3 run_suite.py --base-url $U/v1 --model $M --output "$R/gauntlet-$tag" --tag "p2e-$tag" --cache-salt "p2e-$tag" --sandbox-image "$GIMG" --workers 4) > "$R/gauntlet-log-$tag.txt" 2>&1 || log "gauntlet $tag nonzero rc"
  log "GAUNTLET $tag: $(python3 -c "import json;d=json.load(open('$R/gauntlet-$tag/summary.json'));print(d.get('success_count'),'/',d.get('task_count'))" 2>/dev/null)"; }
measure(){ # tag name since
  local tag=$1 name=$2 since=$3
  log "--- MEASURE $tag ($name) ---"
  rt_c1 "$tag"; fixtures "$tag"
  curl -s $U/metrics | grep -E '^vllm:(spec_decode|prefix_cache)' > "$R/metrics-$tag.txt" 2>/dev/null
  window "$tag"; cold64 "$tag"; gauntlet "$tag"
  hostmem "$tag" steady "$name"
  docker logs --since "$since" "$name" > "$R/postlog-$tag.log" 2>&1; grep -cE 'ValueError:|Traceback|ERROR|RuntimeError:' "$R/postlog-$tag.log" > "$R/errcount-$tag.txt"
  log "ERRCOUNT $tag $(cat "$R/errcount-$tag.txt") (since $since)"; log "--- END MEASURE $tag ---"; }
launch_e(){ # offgb name -> 0 bound, 1 fail, 7 host-guard death
  local off=$1 name=$2 t0 ma hg
  wait_quiet "E-off$off" || log "WARN: host not quiet before E — proceeding"
  stop_all; drop_caches; strays; hostmem "E-off$off" pre ""
  ma=$(memavail); if awk -v m="$ma" 'BEGIN{exit !(m < 380)}'; then log "ABORT E off$off: MemAvailable ${ma} GiB < 380 after stop+drop_caches (stray?)"; return 1; fi
  log "store entries before E boot: $(store_entries)"; store_keys > "$R/store-keys-before-E-off$off.txt"
  t0=$(date +%s); OFFGB=$off bash "$LAUNCH" "$name" || { log "launch failed (E off$off)"; return 1; }
  log "launched $name off$off"
  if wait_bind "$name" 110; then log "boot->ready E $(( $(date +%s)-t0 ))s ($name off$off)"; store_keys > "$R/store-keys-after-E.txt"; return 0; fi
  docker logs "$name" > "$R/boot-E-off$off-FAIL.log" 2>&1
  hg=$(tr '\r' '\n' < "$R/boot-E-off$off-FAIL.log" | grep -c 'PIN_HOT abort.*MemAvailable')
  log "BOOTFAIL E off$off host_guard_hits=$hg tail: $(tr '\r' '\n' < "$R/boot-E-off$off-FAIL.log" | grep -E 'PIN_HOT abort|Error:|ERROR|EngineCore failed|OutOfMemory' | tail -3 | cut -c1-220 | tr '\n' '|')"
  docker stop "$name" >/dev/null 2>&1; docker rename "$name" "$name-BOOT-FAIL" && log "kept $name-BOOT-FAIL (stopped)"
  (( hg > 0 )) && return 7; return 1; }

log "===== CARD P2e START (REF=$REF E=$ENAME) ====="
g=$(gpu_foreign); [[ -n ${g// /} ]] && { log "REFUSE foreign GPU container:$g"; exit 4; }
o=$(other_runners); [[ -n $o ]] && { log "REFUSE: another runner is live: $o"; exit 4; }
docker ps --format '{{.Names}}' | grep -qx "$REF" || { log "ABORT: $REF not running for A"; exit 5; }
curl -s -m 3 $U/v1/models | grep -q $M || { log "ABORT: REF not bound on :30006"; exit 5; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || { log "ABORT: image not present"; exit 5; }
[[ -f $LAUNCH ]] && bash -n "$LAUNCH" || { log "ABORT: $LAUNCH missing or syntax"; exit 5; }
[[ -f $W/agent_fixture_x3.py && -f $W/cardP2e_verdict.py && -f $HOME/dsv41/run_window_T1T2.sh && -f $GDIR/run_suite.py ]] || { log "ABORT: instruments missing"; exit 5; }
for n in "$ENAME" "$E54"; do docker ps -a --format '{{.Names}}' | grep -qx "$n" && { log "ABORT: $n already exists"; exit 5; }; done
log "preflight OK: MemAvailable=$(memavail)GiB store_entries=$(store_entries) sha fixx3=$(sha256sum $W/agent_fixture_x3.py | cut -c1-8) launcher=$(sha256sum $LAUNCH | cut -c1-8) window=$(sha256sum $HOME/dsv41/run_window_T1T2.sh | cut -c1-8)"
wait_quiet A || { log "ABORT: host never quiet; lane untouched (REF serving)"; exit 6; }

# ---- A: live v23 ----
sinceA=$(date -u +%FT%TZ)
boot_facts A "$REF" "$(docker inspect -f '{{.State.StartedAt}}' "$REF")"; strays
measure A "$REF" "$sinceA"

# ---- E: fresh boot, target + drafter FLASHMLA_SPARSE_DSV41 ----
STORE0=$(store_entries); ECN=$ENAME
launch_e 60 "$ENAME"; rc=$?
if (( rc == 7 )); then log "host guard death at off60 -> retry off54 (P2 rule)"; ECN=$E54; launch_e 54 "$E54"; rc=$?; fi
if (( rc == 0 )); then
  boot_facts E "$ECN"; hostmem E bound "$ECN"; log "store entries after E bind: $(store_entries) (before $STORE0)"
  grep -q 'Decoder SWA bounded replay:' "$R/facts-E.txt" && log "WARN E: replay line PRESENT -> E INVALID"
  grep -q 'rehome done layers=40' "$R/facts-E.txt" || log "WARN E: hook did NOT re-home -> E INVALID"
  grep -qiE 'MEGA_ATTN|MegaAttn' "$R/facts-E.txt" && log "WARN E: MegaAttn mentioned in boot facts — check facts-E.txt"
  measure E "$ECN" "$(docker inspect -f '{{.State.StartedAt}}' "$ECN")"
  docker stop "$ECN" >/dev/null; log "stop-and-keep $ECN"
else log "E not measured (rc=$rc)"; fi

# ---- D: v23 REF back, left serving ----
wait_quiet D || log "WARN: host not quiet before D — booting REF anyway (lane must return)"
stop_all; drop_caches; strays
t0=$(date +%s); sinceD=$(date -u +%FT%TZ); docker start "$REF" >/dev/null
if wait_bind "$REF" 40 "$sinceD" nofastfail; then log "boot->ready D $(( $(date +%s)-t0 ))s"; boot_facts D "$REF" "$sinceD"; measure D "$REF" "$sinceD"
else log "D REF FAILED TO BIND — lane dark, needs hands"; fi

# ---- verdict + suffix (no promotion) ----
python3 $W/cardP2e_verdict.py "$R" > "$R/VERDICT.md" 2> "$R/verdict.err"
log "VERDICT: $(grep -m1 '^\*\*Verdict' "$R/VERDICT.md")"
s=$(grep -m1 "^SUFFIX_E " "$R/VERDICT.md" | awk '{print $2}')
if docker ps -a --format '{{.Names}}' | grep -qx "$ECN"; then
  docker ps --format '{{.Names}}' | grep -qx "$ECN" && docker stop "$ECN" >/dev/null 2>&1
  docker rename "$ECN" "$ECN-${s:-DONE}" && log "kept $ECN-${s:-DONE} (stopped)"; fi
grep -hE ' (RT|FIXX3|FIXROT|WINDOW|COLD64K|GAUNTLET|FACTS|HOSTMEM|ERRCOUNT|boot->ready|BOUND|BOOTFAIL|FASTFAIL|DIED|TIMEOUT|WARN|ABORT|VERDICT|store entries|kept) ' "$R/runner.log" > "$R/SUMMARY.txt"
{ echo; echo "## $(date '+%F %H:%M %Z') — Card P2e: config C with target+drafter on FLASHMLA_SPARSE_DSV41 (A live v23 → E fresh boot → D v23 restart) — $(grep -m1 '^\*\*Verdict' "$R/VERDICT.md" 2>/dev/null)"
  grep -vE '^SUFFIX' "$R/VERDICT.md"; echo; echo "Runner summary lines:"; sed 's/^/    /' "$R/SUMMARY.txt"; echo "Receipts: $R (Mac copy: el8-cards-2026-10-18/P2/receipts-P2e/). Not promoted."; } >> "$LEDGER"
log "FINAL dsv41 running=[$(docker ps --format '{{.Names}}' | grep -E '^dsv41-' | tr '\n' ' ')] bound=$(curl -s -m 3 $U/v1/models | grep -c $M)"
log "===== CARD P2e END (lane = $REF) ====="
