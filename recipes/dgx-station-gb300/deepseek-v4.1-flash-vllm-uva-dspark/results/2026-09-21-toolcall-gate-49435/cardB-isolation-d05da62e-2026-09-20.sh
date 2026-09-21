#!/usr/bin/env bash
# CARD B (2026-09-20 night) — isolation boot: vLLM nightly d05da62e (main 2026-09-20 06:00Z, carries #49435 "Fix SM100
# fp8_ds_mla cache scales", #56227 SWA bounded replay, #57604, #56625, #43310) at the exact v19 flags (hook, off54, fp8_ds_mla).
# Question: does v19's tool-shaped logit drift (agent-doc top-1 flips 9.6% vs 0.94% for the v18 hook) follow the fp8_ds_mla
# writer bug fixed by #49435? Instrument: tf_logprob per-class split vs the v14 no-hook capture, run COLD-ONLY (#56227 makes
# post-prefix-hit output non-identical by design; tf capture is one prefill per doc with unique docs, so it is cold by construction).
# Then, if flips drop: BFCL + fixture on the new boot so it can be read against Card A; if not: revert release to v18+corrections.
# New image = new FlashInfer autotune hash => LIVE TUNE (~75 min budget, 110-min bind cap). No cache choreography (nothing to pin yet).
# Pre-step (done by Milo/Grok before launch, NOT here): docker pull of the image with James's window approval; hook class check.
# Rules: never :30003 / dsfv-*, never glm53-*, never docker rm, stop-and-keep, STOP file honoured, v19 restored at end
# (unless VERDICT says revert, which Milo executes separately — this runner never renames REF containers).
# Written by Milo (Hermes milo profile, claude-fable-5-1 via anthropic) for James Meadlock.
set -uo pipefail
set -f
cd /home/milo/dsv41
D=/home/milo/dsv41/cardB-2026-09-20; mkdir -p "$D"
FID=/home/milo/dsv41/fidelity-2026-09-20          # v14nohook / v18 / v19 tf captures live here
BF=/home/milo/pin-hot-experts/bfcl_data
LEDGER=/home/milo/dsv41/results/ledger.md
CAMP=/home/milo/dsv41/overnight-campaign-2026-09-20.log
STOPF=/home/milo/dsv41/STOP-CAMPAIGN
V19=dsv41-vllm-v19-nightly-hook-off54-fp8kv-ddf01704-BOUND-REF
NEWIMG=vllm/vllm-openai:nightly-d05da62e9ccdf8e342b15bf6785d83224cc165af
CAND=dsv41-vllm-v20cand-nightly-d05da62e-hook-off54-fp8kv-EXP
S=/home/milo/pin-hot-experts/scripts
exec >> "$CAMP" 2>&1
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
led(){ printf '\n## %s — %s\nworker: milo-cardB-runner (nohup) · brief by Milo · window approved by James (24 h, 2026-09-20 19:45 CDT)\n%s\n' "$(date '+%F %H:%M %Z')" "$1" "$2" >> "$LEDGER"; }
safety(){ local up; up=$(docker ps --format '{{.Names}}' | grep -vE '^dsv41-vllm-' || true)
  if [[ -n $up ]]; then log "REFUSE foreign container up: $up"; exit 4; fi
  if pgrep -f "python3 [i]wbench" >/dev/null; then log "REFUSE iwbench running"; exit 4; fi
  if ps -eo args | grep -q '^bash /home/milo/pin-hot-experts/scripts/cardA'; then log "REFUSE Card A still running"; exit 4; fi; }
memcheck(){ local a; a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); log "MemAvailable=${a}GiB"; if (( a < 40 )); then log "MemAvailable<40GiB — abort"; restore_v19; exit 5; fi; }
stop_keep(){ local n; for n in $(docker ps --format '{{.Names}}' | grep '^dsv41-vllm-'); do log "stop-and-keep $n"; docker stop "$n" >/dev/null; done; sleep 3; }
drop_caches(){ sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; }
wait_bind(){ local n=$1 m=$2 i; for ((i=0;i<m*6;i++)); do
    if curl -s -m 2 http://127.0.0.1:30006/v1/models 2>/dev/null | grep -q dsv41-flash-uva; then log "BOUND $n after $((i*10))s"; return 0; fi
    if ! docker ps --format '{{.Names}}' | grep -qx "$n"; then log "DIED $n"; return 1; fi
    if docker logs "$n" 2>&1 | grep -qE 'ValueError:|Traceback|RuntimeError:' ; then log "FASTFAIL $n"; docker logs "$n" 2>&1 | grep -E 'ValueError:|RuntimeError:' | tail -3; return 1; fi
    sleep 10; done; log "TIMEOUT bind $n"; return 1; }
boot_facts(){ local n=$1; docker logs "$n" 2>&1 | grep -E 'KV cache format|attention backend|Using .*Backend|PIN_HOT rowmap layer 0 |rehome order|HBM_expert=|GPU KV cache size|Graph capturing finished|autotune cache file|Loaded [0-9]+ configs|Autotuning process|Saved [0-9]+|Engram|vLLM version|FlashInfer|SWA|bounded' | sed 's/^.*\] //' | cut -c1-180 | head -24; }
power(){ nvidia-smi -i 1 --query-gpu=power.draw,power.limit,clocks.sm,temperature.gpu,clocks_event_reasons.active --format=csv,noheader; }
stop_check(){ if [[ -f $STOPF ]]; then log "STOP-CAMPAIGN seen"; restore_v19; exit 0; fi; }
restore_v19(){ stop_keep; drop_caches; docker start "$V19" >/dev/null; wait_bind "$V19" 25 || log "WARN: $V19 did not bind"; }
tf(){ local t=$1; log "TF capture $t"; ( cd "$D" && MAXPOS=4096 CONC=4 python3 $S/tf_logprob.py capture "$t" /home/milo/pin-hot-experts/tf_corpus.jsonl ) > "$D/tf-$t.log" 2>&1; tail -1 "$D/tf-$t.log"; }
bfcl(){ local t=$1; log "BFCL $t start"; log "power-pre $t: $(power)"
  ( cd "$D" && CONC=8 MAXTOK=2048 THINKING=0 python3 $S/bfcl_gate.py "$t" "$BF" "$D" ) > "$D/bfcl-$t.log" 2>&1; log "power-post $t: $(power)"; tail -1 "$D/bfcl-$t.log"; }
fixture(){ local t=$1; log "FIXTURE $t"; CAT_ORDER="prose structured code shell_ops tool_json" bash agent_fixture_o.sh "$t" > "$D/fixture-$t.log" 2>&1; cp -f "agentfix-$t.json" "$D/"; grep -h "accept=" "$D/fixture-$t.log" | cut -c1-100; }
knee2(){ local t=$1; bash knee.sh "$t-r1" > "$D/knee-$t-r1.log" 2>&1; cp -f "knee-$t-r1.json" "$D/"; bash knee.sh "$t-r2" > "$D/knee-$t-r2.log" 2>&1; cp -f "knee-$t-r2.json" "$D/"; grep -h 'conc= 1:\|conc= 8:\|conc=16:' "$D/knee-$t-r2.log" | cut -c1-80; }

log "===== CARD B START (isolation boot d05da62e + #49435) ====="
safety
docker image inspect "$NEWIMG" >/dev/null 2>&1 || { log "IMAGE NOT PRESENT: $NEWIMG — pull is a separate approved step; abort"; exit 3; }
test -f "$FID/tf-v14nohook.jsonl" && test -f "$FID/tf-v19.jsonl" || { log "reference tf captures missing in $FID"; exit 3; }
docker ps -a --format '{{.Names}}' | grep -qx "$CAND" && { log "REFUSE $CAND exists"; exit 5; }
led "Card B start" "Boot $NEWIMG at v19 flags (hook, off54, fp8_ds_mla) as $CAND; live autotune expected (new image hash). Instrument: tf_logprob per-class split vs v14nohook and vs v19 (cold-only by construction). Then BFCL+fixture+knee if the boot binds."

stop_check; stop_keep; drop_caches; memcheck
HOOK=1 OFFGB=54 KVDTYPE=fp8_ds_mla NAME=$CAND IMAGE=$NEWIMG bash $S/launch-t3-nightly.sh > "$D/launch.txt" 2>&1 || { cat "$D/launch.txt"; log "launcher refused"; restore_v19; exit 3; }
cat "$D/launch.txt"
if ! wait_bind "$CAND" 110; then
  docker logs "$CAND" 2>&1 | tail -40 > "$D/boot-fail.txt"
  docker stop "$CAND" >/dev/null 2>&1; docker rename "$CAND" "$CAND-BOOT-FAIL"
  led "Card B BOOT FAIL" "$(grep -E 'Error|error' $D/boot-fail.txt | tail -3 | cut -c1-240 | tr '\n' ' ') · renamed -BOOT-FAIL · v19 restored"
  restore_v19; log "===== CARD B END (boot fail) ====="; exit 6
fi
boot_facts "$CAND" | tee "$D/boot-v20cand.txt"
log "power-post-bind: $(power)"
# hook-rehome sanity: must show HBM_expert / pinned lines, else this is a plain positional boot and the run is void
if ! grep -q "HBM_expert=" "$D/boot-v20cand.txt"; then led "Card B VOID" "hook did not re-home on $NEWIMG (no HBM_expert= line) — v20cand is a positional boot; do not read its numbers"; docker stop "$CAND"; docker rename "$CAND" "$CAND-HOOK-NOFIRE-VOID"; restore_v19; exit 7; fi
bash /home/milo/dsv41/smoke_vllm.sh 2>&1 | tail -6 | tee "$D/smoke-v20cand.txt"

# ---- the discriminator ----
stop_check; tf v20cand
( cd "$D" && python3 $S/tf_logprob.py compare "$FID/tf-v14nohook.jsonl" tf-v20cand.jsonl && python3 $S/tf_logprob.py compare "$FID/tf-v19.jsonl" tf-v20cand.jsonl && python3 $S/tf_logprob.py compare "$FID/tf-v18.jsonl" tf-v20cand.jsonl ) | tee "$D/tf-compare.txt"
( cd "$D" && python3 $S/tf_split.py "$FID/tf-compare-v14nohook-vs-v18.json" "$FID/tf-compare-v14nohook-vs-v19.json" tf-compare-v14nohook-vs-v20cand.json ) | tee "$D/tf-split.txt"
led "Card B TF split" "$(cat $D/tf-split.txt | tr '\n' ' ' | cut -c1-900)"

# ---- speed + correctness on the new boot (so it can be read against Card A) ----
stop_check; knee2 v20cand | tee "$D/knee-v20cand.txt"
stop_check; fixture v20cand | tee "$D/fixture-v20cand.txt"
stop_check; bfcl v20cand
curl -s http://127.0.0.1:30006/metrics | grep spec_decode > "$D/metrics-v20cand.txt"

# ---- verdict ----
python3 - "$D" "$FID" <<'EOF' | tee "$D/VERDICT-cardB.txt"
import json,sys; D,F=sys.argv[1],sys.argv[2]
def split(fn):
    j=json.load(open(fn)); a={"pos":0,"fl":0}
    for k,v in j["per_doc"].items():
        if k.startswith("parity-agent"): a["pos"]+=v["positions"]; a["fl"]+=v["top1_flips"]
    return a["fl"]/a["pos"]*100 if a["pos"] else None, j["top1_flip_rate"]*100
v18=split(f"{F}/tf-compare-v14nohook-vs-v18.json"); v19=split(f"{F}/tf-compare-v14nohook-vs-v19.json"); v20=split(f"{D}/tf-compare-v14nohook-vs-v20cand.json")
print(f"agent-doc top-1 flips vs no-hook: v18 {v18[0]:.2f}%  v19 {v19[0]:.2f}%  v20cand {v20[0]:.2f}%   (corpus-wide: {v18[1]:.2f} / {v19[1]:.2f} / {v20[1]:.2f})")
if v20[0] <= v18[0]*2.0 or v20[0] <= 2.0:
    print("VERDICT CARD B: KV-WRITER — the tool-shaped drift followed #49435; d05da62e at v19 flags is the v20 candidate (needs Card-A-style BFCL/acceptance read + same-window speed pair before promotion)")
elif v20[0] >= v19[0]*0.8:
    print("VERDICT CARD B: IMAGE-KERNELS — drift unchanged on the fixed writer; cause is the nightly's kernels vs 0909. Per James: revert release to v18 + corrections (Milo executes).")
else:
    print("VERDICT CARD B: PARTIAL — flips reduced but not to v18 level; both mechanisms present. Report; no promotion, no revert without James.")
try:
    b=json.load(open(f"{D}/bfcl-v20cand-summary.json")); print(f"BFCL v20cand all={b['all']['acc']} simple={b['simple_python']['acc']} multiple={b['multiple']['acc']} no_call={b['all']['no_call']} errors={b['errors']}")
except Exception as e: print("BFCL v20cand: missing", e)
try:
    r=json.load(open(f"{D}/agentfix-v20cand.json"))["res"]; print("accept v20cand:", {k: r[k]["accept"] for k in ("prose","structured","code","shell_ops","tool_json") if k in r})
except Exception as e: print("fixture v20cand: missing", e)
EOF

restore_v19
led "Card B END" "$(cat $D/VERDICT-cardB.txt | tr '\n' ' ' | cut -c1-900) · $CAND stopped-and-kept · v19 restored on :30006 · receipts $D"
log "===== CARD B END ====="
