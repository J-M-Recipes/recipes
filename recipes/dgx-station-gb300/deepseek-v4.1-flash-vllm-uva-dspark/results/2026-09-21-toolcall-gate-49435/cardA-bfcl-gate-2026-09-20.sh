#!/usr/bin/env bash
# CARD A (2026-09-20 night) — tool-call correctness gate, same window, v19-REF vs v18-RETIRED.
#   BFCL v4 simple_python (399) + multiple (199) via /v1/chat/completions with tools, T=0, thinking off, our AST grader.
#   Plus DSpark acceptance per class from agent_fixture_o (the acceptance half of the promotion rule).
#   Plus (dead-time item) equal-budget GPQA on v18 at 64k so the card's GPQA pair is a pair.
# Order: v19 (live) BFCL -> fixture -> v18 start -> BFCL -> fixture -> GPQA-64k -> v19 restored.
# Promotion rule (James 2026-09-20 19:45 CDT): BFCL exact-match >= v18 - 1 pt AND tool_json/shell acceptance within 2 pts.
# Rules: never :30003 / dsfv-*, never glm53-*, never docker rm, no pull here (Card B pulls), stop-and-keep, STOP file honoured.
# Written by Milo (Hermes milo profile, claude-fable-5-1 via anthropic) for James Meadlock.
set -uo pipefail
cd /home/milo/dsv41
D=/home/milo/dsv41/cardA-2026-09-20; mkdir -p "$D"
BF=/home/milo/pin-hot-experts/bfcl_data
LEDGER=/home/milo/dsv41/results/ledger.md
CAMP=/home/milo/dsv41/overnight-campaign-2026-09-20.log
STOPF=/home/milo/dsv41/STOP-CAMPAIGN
V19=dsv41-vllm-v19-nightly-hook-off54-fp8kv-ddf01704-BOUND-REF
V18=dsv41-vllm-v18-cgsizes-RETIRED-REF
S=/home/milo/pin-hot-experts/scripts
exec >> "$CAMP" 2>&1
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
led(){ printf '\n## %s — %s\nworker: milo-cardA-runner (nohup) · brief by Milo · window approved by James (24 h, 2026-09-20 19:45 CDT)\n%s\n' "$(date '+%F %H:%M %Z')" "$1" "$2" >> "$LEDGER"; }
safety(){ local up; up=$(docker ps --format '{{.Names}}' | grep -vE '^dsv41-vllm-' || true)
  if [[ -n $up ]]; then log "REFUSE foreign container up: $up"; exit 4; fi
  if pgrep -f "python3 [i]wbench" >/dev/null; then log "REFUSE iwbench running"; exit 4; fi; }
stop_keep(){ local n; for n in $(docker ps --format '{{.Names}}' | grep '^dsv41-vllm-'); do log "stop-and-keep $n"; docker stop "$n" >/dev/null; done; sleep 3; }
drop_caches(){ sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; }
wait_bind(){ local n=$1 m=$2 i; for ((i=0;i<m*6;i++)); do
    if curl -s -m 2 http://127.0.0.1:30006/v1/models 2>/dev/null | grep -q dsv41-flash-uva; then log "BOUND $n after $((i*10))s"; return 0; fi
    if ! docker ps --format '{{.Names}}' | grep -qx "$n"; then log "DIED $n"; return 1; fi
    sleep 10; done; log "TIMEOUT bind $n"; return 1; }
power(){ nvidia-smi -i 1 --query-gpu=power.draw,power.limit,clocks.sm,temperature.gpu,clocks_event_reasons.active --format=csv,noheader; }
stop_check(){ if [[ -f $STOPF ]]; then log "STOP-CAMPAIGN seen"; restore_v19; exit 0; fi; }
restore_v19(){ stop_keep; drop_caches; docker start "$V19" >/dev/null; wait_bind "$V19" 25 || log "WARN: $V19 did not bind"; }
start_kept(){ local n=$1; stop_keep; drop_caches; docker start "$n" >/dev/null; wait_bind "$n" 25; }
warm(){ bash /home/milo/dsv41/smoke_vllm.sh 2>&1 | tail -3; bash knee.sh "$1-warm" > "$D/knee-$1-warm.log" 2>&1; grep -h 'conc= 1:' "$D/knee-$1-warm.log" | head -1 | cut -c1-80; }
bfcl(){ local t=$1; log "BFCL $t start"; log "power-pre $t: $(power)"
  ( cd "$D" && CONC=8 MAXTOK=2048 THINKING=0 python3 $S/bfcl_gate.py "$t" "$BF" "$D" ) > "$D/bfcl-$t.log" 2>&1
  log "power-post $t: $(power)"; tail -1 "$D/bfcl-$t.log"; }
fixture(){ local t=$1; log "FIXTURE $t"; CAT_ORDER="prose structured code shell_ops tool_json" bash agent_fixture_o.sh "$t" > "$D/fixture-$t.log" 2>&1; cp -f "agentfix-$t.json" "$D/"
  python3 - "$D/agentfix-$t.json" <<'EOF'
import json,sys; j=json.load(open(sys.argv[1])); r=j.get("res",{})
print("accept:", {k: round(v.get("accept",0),3) for k,v in r.items()}, "tok_s:", {k: round(v.get("tok_s",0),1) for k,v in r.items()})
EOF
}

log "===== CARD A START (BFCL gate v19 vs v18) ====="
safety
test -f "$BF/BFCL_v4_simple_python.json" && test -f "$BF/possible_answer/BFCL_v4_multiple.json" || { log "BFCL data missing at $BF"; exit 3; }
led "Card A start" "BFCL v4 simple_python+multiple (598 cases, T=0, thinking off, our AST grader) on v19-REF then v18-RETIRED, same window; agent_fixture_o acceptance on both; GPQA-Diamond 64k on v18 (equal-budget pair). v19 restored at end."

# ---- v19 (live) ----
stop_check
if ! docker ps --format '{{.Names}}' | grep -qx "$V19"; then start_kept "$V19" || { log "v19 bind fail"; exit 6; }; fi
warm v19 | tee "$D/warm-v19.txt"
bfcl v19; fixture v19 | tee "$D/fixture-v19.txt"
led "Card A v19 done" "$(tail -1 $D/bfcl-v19.log | cut -c1-200) · $(cat $D/fixture-v19.txt | cut -c1-300)"

# ---- v18 (docker start) ----
stop_check
if start_kept "$V18"; then
  warm v18 | tee "$D/warm-v18.txt"
  bfcl v18; fixture v18 | tee "$D/fixture-v18.txt"
  led "Card A v18 done" "$(tail -1 $D/bfcl-v18.log | cut -c1-200) · $(cat $D/fixture-v18.txt | cut -c1-300)"
  # ---- equal-budget GPQA on v18 (dead-time item; ~40 min) ----
  stop_check
  log "GPQA v18 64k start"; ( cd "$D" && CONC=16 MAXTOK=65536 TEMP=0 THINKING=1 TAG=v18-64k OUT="$D/gpqa-v18-64k.jsonl" python3 $S/gpqa_diamond.py /home/milo/pin-hot-experts/gpqa_diamond.csv ) > "$D/gpqa-v18-64k.log" 2>&1
  grep -h "GPQA-Diamond n=" "$D/gpqa-v18-64k.log" | cut -c1-260; led "GPQA v18 64k done" "$(grep -h 'GPQA-Diamond n=' $D/gpqa-v18-64k.log | cut -c1-200)"
else
  led "Card A v18 restart FAIL" "$(docker logs $V18 2>&1 | tail -3 | cut -c1-200)"
fi

# ---- verdict table ----
python3 - "$D" <<'EOF' | tee "$D/VERDICT-cardA.txt"
import json,sys,os; D=sys.argv[1]
def L(p):
    try: return json.load(open(p))
    except Exception as e: return {"err": str(e)}
b19=L(f"{D}/bfcl-v19-summary.json"); b18=L(f"{D}/bfcl-v18-summary.json")
f19=L(f"{D}/agentfix-v19.json").get("res",{}); f18=L(f"{D}/agentfix-v18.json").get("res",{})
print("| metric | v18 | v19 | delta | bar |"); print("|---|---|---|---|---|")
for cat in ("all","simple_python","multiple"):
    a=(b18.get(cat) or {}).get("acc"); b=(b19.get(cat) or {}).get("acc")
    d=None if a is None or b is None else round((b-a)*100,2)
    bar = "" if d is None else ("PASS" if d >= -1.0 else "FAIL")
    print(f"| BFCL {cat} exact-match | {a} | {b} | {d} pt | >= v18-1pt {bar} |")
for c in ("tool_json","shell_ops","prose","structured","code"):
    a=(f18.get(c) or {}).get("accept"); b=(f19.get(c) or {}).get("accept")
    d=None if a is None or b is None else round((b-a)*100,1)
    bar = "" if d is None else (("PASS" if d >= -2.0 else "FAIL") if c in ("tool_json","shell_ops") else "(info)")
    print(f"| accept {c} | {a} | {b} | {d} pt | within 2 pt {bar} |")
nc=lambda s:(s.get("all") or {}).get("no_call"); print(f"| no-call cases | {nc(b18)} | {nc(b19)} | | |")
print(f"| errors | {b18.get('errors')} | {b19.get('errors')} | | |")
EOF

restore_v19
led "Card A END" "$(cat $D/VERDICT-cardA.txt | tr '\n' ' ' | cut -c1-900) · v19 restored on :30006 · receipts $D"
log "===== CARD A END ====="
