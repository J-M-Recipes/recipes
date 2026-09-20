#!/usr/bin/env bash
# FIDELITY runner (2026-09-20) — the two amber items on the recipe card, run by us.
#  1) GPQA-Diamond (198 q, Idavidrein/gpqa, terms accepted by James) on live v18: reasoning ON, T=0, C16. Sanity anchor.
#  2) Teacher-forced Δlogprob on ~79k positions (scripts/tf_logprob.py, tf_corpus.jsonl): prefill-only prompt_logprobs.
#     Captures: v18 (hook, 0909) -> v14 RETIRED-REF restart (NO hook, same image; hash hit) -> v19c restart (nightly + hook + fp8).
#     Compare v18 vs v14 (isolates the hook's numerics) and v19c vs v14 (nightly image + hook). Speculation cannot enter
#     a prefill-only measurement, so no k=0 boot is needed.
#  Waits for the v19d runner to exit. v18 restored at end. Rules as every runner: stop-and-keep, no rm, no pull, STOP file.
# Written by Milo (Hermes milo profile, claude-fable-5-1 via anthropic) for James Meadlock.
set -uo pipefail
cd /home/milo/dsv41
D=/home/milo/dsv41/fidelity-2026-09-20; mkdir -p "$D"
LEDGER=/home/milo/dsv41/results/ledger.md
CAMP=/home/milo/dsv41/overnight-campaign-2026-09-18.log
STOPF=/home/milo/dsv41/STOP-CAMPAIGN
REF=dsv41-vllm-v18-cgsizes-BOUND-REF
V14=dsv41-vllm-v14-1M-ksched-agent-RETIRED-REF
V19C=dsv41-vllm-v19c-nightly-hook-off54-fp8kv-ddf01704-EXP
S=/home/milo/pin-hot-experts/scripts
exec >> "$CAMP" 2>&1
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
led(){ printf '\n## %s — %s\nworker: milo-fidelity-runner (nohup) · brief by Milo\n%s\n' "$(date '+%F %H:%M %Z')" "$1" "$2" >> "$LEDGER"; }
stop_keep(){ local n; for n in $(docker ps --format '{{.Names}}' | grep '^dsv41-vllm-'); do log "stop-and-keep $n"; docker stop "$n" >/dev/null; done; sleep 3; }
drop_caches(){ sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; }
wait_bind(){ local n=$1 m=$2 i; for ((i=0;i<m*6;i++)); do
    if curl -s -m 2 http://127.0.0.1:30006/v1/models 2>/dev/null | grep -q dsv41-flash-uva; then log "BOUND $n after $((i*10))s"; return 0; fi
    if ! docker ps --format '{{.Names}}' | grep -qx "$n"; then log "DIED $n"; return 1; fi
    sleep 10; done; log "TIMEOUT bind $n"; return 1; }
stop_check(){ if [[ -f $STOPF ]]; then log "STOP-CAMPAIGN seen"; restore_ref; exit 0; fi; }
restore_ref(){ stop_keep; if ! docker ps --format '{{.Names}}' | grep -qx "$REF"; then log "restore $REF"; docker start "$REF" >/dev/null; wait_bind "$REF" 15 || log "WARN: $REF did not bind"; fi; }
start_kept(){ local n=$1; stop_keep; drop_caches; docker start "$n" >/dev/null; wait_bind "$n" 25; }
tf(){ local t=$1; log "TF capture $t"; ( cd "$D" && MAXPOS=4096 CONC=4 python3 $S/tf_logprob.py capture "$t" /home/milo/pin-hot-experts/tf_corpus.jsonl ) > "$D/tf-$t.log" 2>&1; tail -1 "$D/tf-$t.log"; }

for ((i=0;i<540;i++)); do pgrep -f "bash .*v19d-runner" >/dev/null || break; sleep 10; done
log "===== FIDELITY START ====="
if pgrep -f "python3 [i]wbench" >/dev/null; then log "REFUSE iwbench running"; exit 4; fi
led "fidelity runner start" "GPQA-Diamond 198 on v18 (reasoning on, T=0, C16); TF Δlogprob ~79k positions: v18 -> v14 no-hook (ref) -> v19c."
# ---- ensure v18 up ----
stop_check; if ! docker ps --format '{{.Names}}' | grep -qx "$REF"; then start_kept "$REF" || { log "v18 bind fail"; exit 6; }; fi
# ---- 1) GPQA on v18 ----
log "GPQA v18 start"; ( cd "$D" && CONC=16 MAXTOK=16384 TEMP=0 THINKING=1 TAG=v18 OUT="$D/gpqa-v18.jsonl" python3 $S/gpqa_diamond.py /home/milo/pin-hot-experts/gpqa_diamond.csv ) > "$D/gpqa-v18.log" 2>&1
grep -h "GPQA-Diamond n=" "$D/gpqa-v18.log" | cut -c1-260; led "GPQA v18 done" "$(grep -h 'GPQA-Diamond n=' $D/gpqa-v18.log | cut -c1-200)"
# ---- 2) TF captures ----
stop_check; tf v18
stop_check; if start_kept "$V14"; then tf v14nohook; else led "v14 restart FAIL" "$(docker logs $V14 2>&1 | tail -3 | cut -c1-200)"; fi
stop_check; if start_kept "$V19C"; then tf v19c; else led "v19c restart FAIL" "$(docker logs $V19C 2>&1 | tail -3 | cut -c1-200)"; fi
( cd "$D" && for p in "v14nohook v18" "v14nohook v19c" "v18 v19c"; do set -- $p; test -f tf-$1.jsonl && test -f tf-$2.jsonl && python3 $S/tf_logprob.py compare tf-$1.jsonl tf-$2.jsonl; done ) | tee "$D/tf-compare.txt"
restore_ref
log "===== FIDELITY END ====="; led "fidelity runner END" "$(cat $D/tf-compare.txt | cut -c1-300 | tr '\n' ' ') · v18 restored; receipts $D"
