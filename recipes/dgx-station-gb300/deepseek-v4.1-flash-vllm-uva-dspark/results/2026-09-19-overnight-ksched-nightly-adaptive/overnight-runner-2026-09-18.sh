#!/usr/bin/env bash
# OVERNIGHT CAMPAIGN 2026-09-18/19 — detached on-box runner (nohup). Sequence: T1(rest) → T2 → T3 → T3b → T4.
# Why a runner: subagent workers are capped at 15 min; this campaign is ~8 h. Same pattern as campaign_round*.sh.
# Rules: never :30003 / dsfv-*, never glm53-*, never docker rm, no docker pull (nightly is pre-pulled), stop-and-keep,
# STOP file honoured between steps, v18 restored at the end no matter what. Every step appends to the ledger.
# Written by Milo for James Meadlock. Worker measurement window = run_window_T1T2.sh (grok-4.6's, verified on T1).
set -uo pipefail
set -f
cd $BOX_HOME/dsv41
OUT=$BOX_HOME/dsv41/overnight-T1T2
mkdir -p "$OUT" $BOX_HOME/dsv41/overnight-T3 $BOX_HOME/dsv41/overnight-T4
LEDGER=$BOX_HOME/dsv41/results/ledger.md
CAMP=$BOX_HOME/dsv41/overnight-campaign-2026-09-18.log
STOPF=$BOX_HOME/dsv41/STOP-CAMPAIGN
REF=dsv41-vllm-v18-cgsizes-BOUND-REF
S=$BOX_HOME/pin-hot-experts/scripts
exec >> "$CAMP" 2>&1

log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
led(){ printf '\n## %s — %s\nworker: milo-runner (nohup) · brief by Milo\n%s\n' "$(date '+%F %H:%M %Z')" "$1" "$2" >> "$LEDGER"; }
stop_check(){ if [[ -f $STOPF ]]; then log "STOP-CAMPAIGN seen"; restore_ref; exit 0; fi; }
safety(){  # refuse if anything other than our own lane is up
  local up; up=$(docker ps --format '{{.Names}}' | grep -vE '^dsv41-vllm-' || true)
  if [[ -n $up ]]; then log "REFUSE foreign container up: $up"; restore_ref; exit 4; fi
  if docker ps --format '{{.Names}}' | grep -qE 'dsfv-|glm53-'; then log "REFUSE prod lane up"; exit 4; fi
}
memcheck(){ local a; a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); log "MemAvailable=${a}GiB"; if (( a < 40 )); then log "MemAvailable<40GiB — abort night"; restore_ref; exit 5; fi; }
stop_keep(){ local n; for n in $(docker ps --format '{{.Names}}' | grep '^dsv41-vllm-'); do log "stop-and-keep $n"; docker stop "$n" >/dev/null; done; sleep 3; }
drop_caches(){ sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; }
wait_bind(){ # $1 name $2 max_minutes
  local n=$1 m=$2 i; for ((i=0;i<m*6;i++)); do
    if curl -s -m 2 http://127.0.0.1:30006/v1/models 2>/dev/null | grep -q dsv41-flash-uva; then log "BOUND $n after $((i*10))s"; return 0; fi
    if ! docker ps --format '{{.Names}}' | grep -qx "$n"; then log "DIED $n"; return 1; fi
    if docker logs "$n" 2>&1 | grep -qE 'ValueError:|Traceback|RuntimeError:' ; then log "FASTFAIL $n"; docker logs "$n" 2>&1 | grep -E 'ValueError:|RuntimeError:' | tail -3; return 1; fi
    sleep 10; done; log "TIMEOUT bind $n"; return 1; }
boot_facts(){ local n=$1; docker logs "$n" 2>&1 | grep -E 'PIN_HOT rowmap layer 0 |HBM_expert=|GPU KV cache size|Graph capturing finished|autotune cache file|Loaded [0-9]+ configs|Autotuning process ends|ADAPT counter|GO seen|Engram' | sed 's/^.*\] //' | cut -c1-160 | head -14; }
restore_ref(){ stop_keep; if ! docker ps --format '{{.Names}}' | grep -qx "$REF"; then log "restore $REF"; docker start "$REF" >/dev/null; wait_bind "$REF" 15 || log "WARN: $REF did not bind"; fi; }
start_ref(){ stop_keep; drop_caches; docker start "$REF" >/dev/null; wait_bind "$REF" 15 || { log "REF failed to bind"; exit 6; }; }
window(){ # $1 tag
  local t=$1; log "WINDOW $t"; bash $BOX_HOME/dsv41/run_window_T1T2.sh "$t" > "$OUT/nohup-$t.log" 2>&1; grep -h 'conc=' "$OUT/window-$t.log" | tr '\n' ';' | cut -c1-600; echo; }
seed(){ nohup bash $BOX_HOME/dsv41/seed_autotune.sh > $BOX_HOME/dsv41/seed-$1.log 2>&1 & }
smoke(){ bash $BOX_HOME/dsv41/smoke_vllm.sh 2>&1 | tail -8; }

log "===== CAMPAIGN START ====="; safety; memcheck
led "overnight runner start" "T1 window already measured by grok-4.6 (hash hit 62426808, KV 2,406,686, C1 170.6/171.2 C8 615.8/619.2 C16 746/952). Runner continues: v18ctl → T2 → v18ctl2 → T1b → T2b → T3 → T3b → T4."

# ---------- T1/T2 remaining windows ----------
stop_check; start_ref; window v18ctl
stop_check; stop_keep; drop_caches; seed T2
NAME=dsv41-vllm-T2-ksched-5-3-EXP KSCHED='[[1,8,5],[9,24,3]]' bash $S/launch-many-seat.sh >/dev/null
if wait_bind dsv41-vllm-T2-ksched-5-3-EXP 90; then boot_facts dsv41-vllm-T2-ksched-5-3-EXP; window T2; else led "T2 boot FAIL" "$(docker logs dsv41-vllm-T2-ksched-5-3-EXP 2>&1 | grep -E 'Error|ValueError' | tail -3)"; docker rename dsv41-vllm-T2-ksched-5-3-EXP dsv41-vllm-T2-ksched-5-3-EXP-BOOT-FAIL; fi
stop_check; start_ref; window v18ctl2
stop_check; stop_keep; drop_caches; docker start dsv41-vllm-T1-ksched-5-3-1-EXP >/dev/null; wait_bind dsv41-vllm-T1-ksched-5-3-1-EXP 15 && window T1b
stop_check; if docker ps -a --format '{{.Names}}' | grep -qx dsv41-vllm-T2-ksched-5-3-EXP; then stop_keep; drop_caches; docker start dsv41-vllm-T2-ksched-5-3-EXP >/dev/null; wait_bind dsv41-vllm-T2-ksched-5-3-EXP 15 && window T2b; fi
led "T1/T2 windows complete" "receipts in $OUT; verdict by Milo in the morning"

# ---------- T3: nightly, hook off ----------
stop_check; memcheck; stop_keep; drop_caches; seed T3
HOOK=0 bash $S/launch-t3-nightly.sh >/dev/null
T3=dsv41-vllm-T3-nightly-dee37d89-nohook-EXP
if wait_bind $T3 100; then
  boot_facts $T3 | tee $BOX_HOME/dsv41/overnight-T3/boot-T3.txt
  smoke | tee $BOX_HOME/dsv41/overnight-T3/smoke-T3.txt
  window T3; mv "$OUT"/*T3* $BOX_HOME/dsv41/overnight-T3/ 2>/dev/null
  BASE_URL=http://127.0.0.1:30006/v1 MODEL=dsv41-flash-uva API_KEY=none SIZES="8000 32000 128000" N=2 THINKING=0 python3 cold_prefill_probe.py > $BOX_HOME/dsv41/overnight-T3/coldprefill-T3.txt 2>&1
  led "T3 nightly hook-off bound" "$(cat $BOX_HOME/dsv41/overnight-T3/boot-T3.txt | tr '\n' ' ' | cut -c1-500)"
  # ---------- T3b class check (no GPU) ----------
  docker run --rm --entrypoint python3 vllm/vllm-openai:nightly-dee37d89115db4c94a820a79a78a7828e141c910 - > $BOX_HOME/dsv41/overnight-T3/class-check.txt 2>&1 <<'EOF'
import importlib, inspect, subprocess, sys
hits = subprocess.run(["grep","-rl","class TrtLlmMxfp4ExpertsModular","/usr/local/lib/python3.12/dist-packages/vllm"],capture_output=True,text=True).stdout.split()
print("files:", hits)
if not hits: print("CLASS MISSING"); sys.exit(2)
p = hits[0].split("dist-packages/")[1][:-3].replace("/",".")
mod = importlib.import_module(p); cls = getattr(mod, "TrtLlmMxfp4ExpertsModular")
print("module:", p); print("has _invoke_kernel:", hasattr(cls,"_invoke_kernel"))
print("sig:", inspect.signature(cls._invoke_kernel) if hasattr(cls,"_invoke_kernel") else "NONE")
EOF
  cat $BOX_HOME/dsv41/overnight-T3/class-check.txt
  OLD=$(docker run --rm --entrypoint python3 vllm/vllm-openai:deepseekv41-flash-0909 -c 'import importlib,inspect,subprocess;h=subprocess.run(["grep","-rl","class TrtLlmMxfp4ExpertsModular","/usr/local/lib/python3.12/dist-packages/vllm"],capture_output=True,text=True).stdout.split();p=h[0].split("dist-packages/")[1][:-3].replace("/",".");m=importlib.import_module(p);print(inspect.signature(m.TrtLlmMxfp4ExpertsModular._invoke_kernel))' 2>/dev/null)
  NEW=$(grep '^sig:' $BOX_HOME/dsv41/overnight-T3/class-check.txt | sed 's/^sig: //')
  echo "OLD sig: $OLD" >> $BOX_HOME/dsv41/overnight-T3/class-check.txt
  if [[ -n $NEW && "$NEW" == "$OLD" ]]; then
    log "T3b class check PASS — booting hook on nightly"; stop_keep; drop_caches; seed T3b
    HOOK=1 bash $S/launch-t3-nightly.sh >/dev/null; T3B=dsv41-vllm-T3b-nightly-dee37d89-hook-EXP
    if wait_bind $T3B 100; then boot_facts $T3B | tee $BOX_HOME/dsv41/overnight-T3/boot-T3b.txt; smoke > $BOX_HOME/dsv41/overnight-T3/smoke-T3b.txt; window T3b; mv "$OUT"/*T3b* $BOX_HOME/dsv41/overnight-T3/ 2>/dev/null; led "T3b nightly+hook bound" "$(cat $BOX_HOME/dsv41/overnight-T3/boot-T3b.txt | tr '\n' ' ' | cut -c1-500)"
    else led "T3b boot FAIL" "$(docker logs $T3B 2>&1 | grep -E 'Error|ValueError|PIN_HOT' | tail -4 | cut -c1-300)"; docker rename $T3B ${T3B}-BOOT-FAIL; fi
  else led "T3b SKIPPED — hook target changed" "OLD: $OLD | NEW: $NEW | see overnight-T3/class-check.txt"; fi
else led "T3 boot FAIL" "$(docker logs $T3 2>&1 | grep -iE 'Error|ValueError|driver|insufficient' | tail -4 | cut -c1-300)"; docker rename $T3 ${T3}-BOOT-FAIL; fi
stop_check; start_ref; window v18ctl3; mv "$OUT"/*v18ctl3* $BOX_HOME/dsv41/overnight-T3/ 2>/dev/null

# ---------- T4: adaptive unfrozen, C8 gate ----------
stop_check; memcheck; stop_keep; drop_caches; seed T4
bash $S/launch-t4-adaptive-v18.sh >/dev/null; T4=dsv41-vllm-T4-adaptive-d1-unfrozen-C8gate-EXP
if wait_bind $T4 90; then
  boot_facts $T4 | tee $BOX_HOME/dsv41/overnight-T4/boot-T4.txt; smoke > $BOX_HOME/dsv41/overnight-T4/smoke-T4.txt
  touch $BOX_HOME/pin-hot-experts/t4/GO; sleep 20; docker logs $T4 2>&1 | grep -E 'GO seen' | tail -1
  t4win(){ local t=$1; env TAG="$t-warm" N=4 ROUNDS=1 python3 replay_c.py >/dev/null 2>&1; echo "swaps after warm: $(wc -l < $BOX_HOME/pin-hot-experts/t4/swaps.jsonl)"; window "$t"
    for p in 1 2; do python3 $BOX_HOME/pin-hot-experts/scripts/e2c_heldout.py > $BOX_HOME/dsv41/overnight-T4/heldout-$t-p$p.json 2>&1 || cp $BOX_HOME/pin-hot-experts/scripts/e2c_heldout.py /dev/null; echo "swaps after heldout p$p: $(wc -l < $BOX_HOME/pin-hot-experts/t4/swaps.jsonl)"; done
    cp $BOX_HOME/pin-hot-experts/t4/swaps.jsonl $BOX_HOME/dsv41/overnight-T4/swaps-$t.jsonl; mv "$OUT"/*"$t"* $BOX_HOME/dsv41/overnight-T4/ 2>/dev/null; }
  t4win T4
  stop_check; start_ref; window v18ctl4; python3 $BOX_HOME/pin-hot-experts/scripts/e2c_heldout.py > $BOX_HOME/dsv41/overnight-T4/heldout-v18ctl4.json 2>&1; mv "$OUT"/*v18ctl4* $BOX_HOME/dsv41/overnight-T4/ 2>/dev/null
  stop_check; stop_keep; drop_caches; docker start $T4 >/dev/null; wait_bind $T4 15 && t4win T4b
  led "T4 windows complete" "swaps total $(wc -l < $BOX_HOME/pin-hot-experts/t4/swaps.jsonl); receipts overnight-T4/"
else led "T4 boot FAIL" "$(docker logs $T4 2>&1 | grep -E 'Error|ValueError|PIN_HOT|ADAPT' | tail -4 | cut -c1-300)"; docker rename $T4 ${T4}-BOOT-FAIL; fi

restore_ref; log "===== CAMPAIGN END — v18 restored ====="; led "overnight runner END" "v18 restored on :30006; all candidates stopped-and-kept; Milo to verdict."
