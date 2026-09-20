#!/usr/bin/env bash
# V19C runner (2026-09-20 afternoon) — third C16 fund window for the v19 candidate (v19b p95 2.56 vs bar 2.28; v19a 1.68).
# One boot of the loaded config, mix 8/120 warm (discard) + mix 16/420 ×2, then v18 restored. Derived from v19-async-runner.
# ORIGINAL HEADER FOLLOWS:
# — two decisions in one box session, same-window against live v18.
#  A) v19 candidate, LOADED config: nightly dee37d89 + v15 hook + off54 + fp8_ds_mla + the ddf01704 autotune set already
#     in place (T3b6 saved it; T3b6b loaded it once at 183.3 C1). Question: is 183 a stable draw, and what does the
#     fund harness (the lane's real job) say at C16/C24?  Windows: v19a -> v18ctl9 -> v19b, each = knee ×2 + fund harness.
#  B) --async-scheduling on the v18 launch (0909 image, hook, all v18 flags + --async-scheduling). Hash hit expected.
#     Question: at v18's ~5.8 ms step, does taking the scheduler off the Grace critical path buy anything? (v12 wash.)
#     Windows: ASYNC -> v18ctl10, each = run_window_T1T2 (knee ×2, knee6, fixture, replay ×2).
# Bars: v19 promote-candidate holds if C1 >= 172*1.05 on both v19 boots AND fund C24 warm p50 <= v18ctl9 AND C16 warm
#       p95 <= v18ctl9 + 0.5 s AND tools == 64/64. ASYNC adopt if C1 and C8 >= +2% and replay >= 0; else CLOSED.
# Rules: never :30003 / dsfv-*, never glm53-*, never docker rm, no pull, stop-and-keep, STOP file honoured, v18 restored.
# Written by Milo (Hermes milo profile, claude-fable-5-1 via anthropic) for James Meadlock.
set -uo pipefail
set -f
cd /home/milo/dsv41
OUT=/home/milo/dsv41/overnight-T1T2
D=/home/milo/dsv41/v19c-2026-09-20
IW=/home/milo/iwyzer
PAR=/home/milo/pin-hot-experts/e2c/t3b-parity
mkdir -p "$OUT" "$D" "$PAR"
LEDGER=/home/milo/dsv41/results/ledger.md
CAMP=/home/milo/dsv41/overnight-campaign-2026-09-18.log
STOPF=/home/milo/dsv41/STOP-CAMPAIGN
REF=dsv41-vllm-v18-cgsizes-BOUND-REF
S=/home/milo/pin-hot-experts/scripts
V19C=dsv41-vllm-v19c-nightly-hook-off54-fp8kv-ddf01704-EXP
V19B=dsv41-vllm-v19b-nightly-hook-off54-fp8kv-ddf01704-EXP
ASYNC=dsv41-vllm-ASYNC-v18-async-scheduling-EXP
CACHEROOT=/home/milo/dsv41/vllm-cache/flashinfer_autotune_cache/0.6.18.post1/103a
PROMPTS=/home/milo/pin-hot-experts/scripts/e5_parity_prompts.json
exec >> "$CAMP" 2>&1

log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
led(){ printf '\n## %s — %s\nworker: milo-v19c-runner (nohup) · brief by Milo\n%s\n' "$(date '+%F %H:%M %Z')" "$1" "$2" >> "$LEDGER"; }
safety(){ local up; up=$(docker ps --format '{{.Names}}' | grep -vE '^dsv41-vllm-' || true)
  if [[ -n $up ]]; then log "REFUSE foreign container up: $up"; exit 4; fi
  if docker ps --format '{{.Names}}' | grep -qE 'dsfv-|glm53-'; then log "REFUSE prod lane up"; exit 4; fi
  if pgrep -f "python3 [i]wbench" >/dev/null; then log "REFUSE iwbench already running"; exit 4; fi; }
memcheck(){ local a; a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); log "MemAvailable=${a}GiB"; if (( a < 40 )); then log "MemAvailable<40GiB — abort"; restore_ref; exit 5; fi; }
stop_keep(){ local n; for n in $(docker ps --format '{{.Names}}' | grep '^dsv41-vllm-'); do log "stop-and-keep $n"; docker stop "$n" >/dev/null; done; sleep 3; }
drop_caches(){ sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; }
wait_bind(){ local n=$1 m=$2 i; for ((i=0;i<m*6;i++)); do
    if curl -s -m 2 http://127.0.0.1:30006/v1/models 2>/dev/null | grep -q dsv41-flash-uva; then log "BOUND $n after $((i*10))s"; return 0; fi
    if ! docker ps --format '{{.Names}}' | grep -qx "$n"; then log "DIED $n"; return 1; fi
    if docker logs "$n" 2>&1 | grep -qE 'ValueError:|Traceback|RuntimeError:' ; then log "FASTFAIL $n"; docker logs "$n" 2>&1 | grep -E 'ValueError:|RuntimeError:' | tail -3; return 1; fi
    sleep 10; done; log "TIMEOUT bind $n"; return 1; }
boot_facts(){ local n=$1; docker logs "$n" 2>&1 | grep -E 'KV cache format|PIN_HOT rowmap layer 0 |rehome order|HBM_expert=|GPU KV cache size|Graph capturing finished|autotune cache file|Loaded [0-9]+ configs|Autotuning process ends|Saved [0-9]+|Engram|async' | sed 's/^.*\] //' | cut -c1-160 | head -18; }
power(){ nvidia-smi -i 1 --query-gpu=power.draw,power.limit,clocks.sm,clocks.mem,temperature.gpu,clocks_event_reasons.active --format=csv,noheader; }
stop_check(){ if [[ -f $STOPF ]]; then log "STOP-CAMPAIGN seen"; restore_ref; exit 0; fi; }
restore_ref(){ stop_keep; if ! docker ps --format '{{.Names}}' | grep -qx "$REF"; then log "restore $REF"; docker start "$REF" >/dev/null; wait_bind "$REF" 15 || log "WARN: $REF did not bind"; fi; }
start_ref(){ stop_keep; drop_caches; docker start "$REF" >/dev/null; wait_bind "$REF" 15 || { log "REF failed to bind"; exit 6; }; }
smoke(){ bash /home/milo/dsv41/smoke_vllm.sh 2>&1 | tail -8; }
knee2(){ local t=$1; log "KNEE $t"; bash knee.sh "$t-r1" > "$D/knee-$t-r1.log" 2>&1; bash knee.sh "$t-r2" > "$D/knee-$t-r2.log" 2>&1
  cp -f "knee-$t-r1.json" "knee-$t-r2.json" "$D/" 2>/dev/null; grep -h 'conc= 1:\|conc= 8:\|conc=16:' "$D/knee-$t-r1.log" "$D/knee-$t-r2.log" | tr '\n' ';' | cut -c1-400; echo; }
fund16(){ local t=$1; log "FUND16 $t"; log "power-pre $t: $(power)"
  ( cd "$IW" && export CHARS_PER_TOKEN=3.894 BASE_URL=http://127.0.0.1:30006/v1 MODEL=dsv41-flash-uva API_KEY=none THINKING=0 OUT="$IW/iw-$t.jsonl"
    TAG=$t-warm python3 iwbench.py mix 8 120; TAG=$t-w1 python3 iwbench.py mix 16 420; TAG=$t-w2 python3 iwbench.py mix 16 420 ) > "$D/run-$t.log" 2>&1
  cp -f "$IW/iw-$t.jsonl" "$D/"; log "power-post $t: $(power)"; grep -hE "MIX C16:|WARM agent turn" "$D/run-$t.log" | cut -c1-200; }
fund(){ local t=$1; log "FUND $t"; log "power-pre $t: $(power)"
  ( cd "$IW" && export CHARS_PER_TOKEN=3.894 BASE_URL=http://127.0.0.1:30006/v1 MODEL=dsv41-flash-uva API_KEY=none THINKING=0 OUT="$IW/iw-$t.jsonl"
    TAG=$t-warm python3 iwbench.py mix 8 120; TAG=$t python3 iwbench.py mix 16 420; TAG=$t python3 iwbench.py mix 24 420
    TAG=$t python3 iwbench.py coldload 5; TAG=$t python3 iwbench.py tools 16 ) > "$D/run-$t.log" 2>&1
  cp -f "$IW/iw-$t.jsonl" "$D/"; log "power-post $t: $(power)"
  grep -hE "MIX C(16|24):|WARM agent turn|COLD .* under|TOOLS C16" "$D/run-$t.log" | cut -c1-200; }
window(){ local t=$1; log "WINDOW $t"; log "power-pre $t: $(power)"; bash /home/milo/dsv41/run_window_T1T2.sh "$t" > "$OUT/nohup-$t.log" 2>&1; log "power-post $t: $(power)"; grep -h 'conc=' "$OUT/window-$t.log" | tr '\n' ';' | cut -c1-600; echo; mv "$OUT"/*"$t"* "$D"/ 2>/dev/null; }
parity(){ local t=$1; log "PARITY $t"; python3 $S/e2c_parity.py "$t" "$PROMPTS" "$PAR" > "$PAR/log-$t.txt" 2>&1 || log "parity $t nonzero exit"; }
v19cand(){ local name=$1 tag=$2
  stop_check; stop_keep; drop_caches
  if docker ps -a --format '{{.Names}}' | grep -qx "$name"; then log "REFUSE $name exists"; return 5; fi
  HOOK=1 OFFGB=54 KVDTYPE=fp8_ds_mla NAME=$name bash $S/launch-t3-nightly.sh >/dev/null
  if wait_bind "$name" 40; then
    boot_facts "$name" | tee "$D/boot-$tag.txt"
    if ! grep -q 'Loaded 189' "$D/boot-$tag.txt"; then log "WARN $tag did not load the ddf01704 set (live tune?) — recorded, continuing"; fi
    smoke | tee "$D/smoke-$tag.txt"; knee2 "$tag"; fund "$tag"; parity "$tag"
    led "$tag window complete" "$(grep -h 'conc= 1:' $D/knee-$tag-r1.log | head -1 | cut -c1-100); fund: $(grep -h 'WARM agent turn' $D/run-$tag.log | tail -1 | cut -c1-120)"
    return 0
  else led "$tag boot FAIL" "$(docker logs $name 2>&1 | grep -E 'Error|ValueError|PIN_HOT' | tail -4 | cut -c1-300)"; docker rename "$name" "${name}-BOOT-FAIL"; return 1; fi; }

log "===== V19C START ====="; safety; memcheck
test -d "$CACHEROOT/ddf01704bfd54f378b73e381bec6601ad21c7ce1b07d359ef605429001378c0f" || { log "ddf01704 cache dir missing"; exit 3; }
led "v19c runner start" "Third C16 fund window for v19 (loaded config): v19c mix16 ×2 -> v18ctl11 mix16 ×2. Settles the v19b p95 miss (2.56 vs 2.28 bar)."
stop_check; stop_keep; drop_caches
if docker ps -a --format '{{.Names}}' | grep -qx "$V19C"; then log "REFUSE $V19C exists"; restore_ref; exit 5; fi
HOOK=1 OFFGB=54 KVDTYPE=fp8_ds_mla NAME=$V19C bash $S/launch-t3-nightly.sh >/dev/null
if wait_bind "$V19C" 40; then boot_facts "$V19C" | tee "$D/boot-v19c.txt"; smoke | tee "$D/smoke-v19c.txt"; fund16 v19c
  led "v19c window complete" "$(grep -h 'WARM agent turn' $D/run-v19c.log | tail -2 | cut -c1-120 | tr '\n' ' ')"
else led "v19c boot FAIL" "$(docker logs $V19C 2>&1 | grep -E 'Error|ValueError|PIN_HOT' | tail -4 | cut -c1-300)"; docker rename "$V19C" "${V19C}-BOOT-FAIL"; fi
stop_check; start_ref; log "power-ref: $(power)"; fund16 v18ctl11
restore_ref; log "power-end: $(power)"
log "===== V19C END ====="; led "v19c runner END" "v18 restored on :30006; v19c stopped-and-kept; receipts $D"
