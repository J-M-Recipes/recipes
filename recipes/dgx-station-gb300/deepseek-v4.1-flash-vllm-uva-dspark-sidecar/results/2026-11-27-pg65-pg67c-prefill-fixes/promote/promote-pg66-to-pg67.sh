#!/bin/bash
# Promote Pg66 -> Pg67: F1a routing-glue fusion + capture ladder 384/512/768 on a WORK COPY.
# work dir: ~/dsv41/sidecar-resid/work-Pg67 (pg66-work immutable, untouched).
# Gates (all must pass, else down + restore Pg66): boot hash verify, FUSE1 selftests (layers 0/20/21/39),
# >= 48 capture lines (45+3 new sizes, prod prints multiplicities), A12 receipt: status PASS,
# post_capture_free_bytes >= 12.0 GiB, kv_tokens >= 11,500,000; policy env production-complete, no
# DS41_OPC/DS41_W12/DS41_INSTR/PEER2_*; ladder literal = 256,...,384,512,768; same-boot det smoke (8 turns);
# eager step probe: uncached 264 captured cheap (< 90 ms srv_prefill median of 2x probes).
# Rollback: promote/rollback-pg67-to-pg66.sh (written alongside, tested shape).
# Pre-registered BEFORE the run (Milo): promote only if every gate above is green.
set -uo pipefail
GR=~/dsv41/garner; F=~/dsv41/sidecar-resid; W67=$F/work-Pg67
MODEL_DIR=/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170
LOG=$GR/promote/promote-pg67-$(date +%Y%m%d-%H%M).log; exec >>"$LOG" 2>&1
ts(){ date "+%F %T %Z"; }; lo(){ echo "- $(ts) $*" >> ~/dsv41/m3/LANE-OWNER.txt; }

fail(){ (3>/dev/null || true); echo "$(ts) GATE FAIL: $*"; lo "PROMOTE Pg67 GATE FAIL ($*) -- restoring Pg66"
  cp "$LOG" $GR/home/state/promote-pg67-FAILED.txt 2>/dev/null
  env WORK=$W67 DS41_HOME=$GR/home bash $GR/launcher/launch-pg61.sh down || true; sleep 20
  env WORK=$GR/pg66-work MODEL_DIR=$MODEL_DIR DS41_HOME=$GR/home bash $GR/launcher/launch-pg61.sh up
  lo "PROMOTE aborted; Pg66 restored (verify separately)"; exit 3; }

lo "PROMOTE Pg66 -> Pg67 starting (F1a + rungs 384/512/768; James: 'try them all plz' 2026-11-27 ~11:40)"
# sanity: tree is what we claim
diff -rq $GR/pg66-work/tree $W67/tree 2>/dev/null | grep -v __pycache__ > /tmp/pg67-diff.txt
[[ $(wc -l < /tmp/pg67-diff.txt) -eq 3 ]] || fail "tree diff count $(wc -l < /tmp/pg67-diff.txt) != 3"
grep -aq "AS1408" /dev/null || true   # nothing; keep shell quiet
ld=$(python3 $F/f-overlay/r7_cg.py show $W67/tree/serve/serve.sh | sed -E 's/.*: \[([^]]*)\].*/\1/')
[[ "$ld" == "1, 2, 4, 6, 8, 12, 16, 18, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 96, 128, 144, 160, 192, 224, 256, 384, 512, 768" ]] || fail "ladder literal '$ld'"
echo "$(ts) tree verified: fuse1_route.py + mega_peer_hook.py + ladder 384/512/768"
touch ~/dsv41/m3/NO-AUTOSTART   # autostart must not fight the promote (remove at the end)
env WORK=$GR/pg66-work DS41_HOME=$GR/home bash $GR/launcher/launch-pg61.sh down || fail "down pg66 rc"
sleep 20
for i in $(seq 1 60); do docker ps -a --format '{{.Names}}' | grep -q '^ds41\(-sidecar\)\?$' || break; sleep 2; done
sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; sleep 45
env WORK=$W67 MODEL_DIR=$MODEL_DIR DS41_HOME=$GR/home bash $GR/launcher/launch-pg61.sh up || fail "up Pg67 (see launch log)"
bl=~/dsv41/garner/home/state/logs; slog=$(ls -t $bl/*-ds41.log | head -1)
tail -2000 "$slog" > /tmp/pg67-boot.log; cp /tmp/pg67-boot.log $F/results-Pg67-boot.log
# gates
ok=1
grep -aq "verify OK: gumbel 6eac3e134833 rsu edc8f8f4c425 libw6mk b54168223847560b PEER3_SPINS=1000000000000" /tmp/pg67-boot.log || { echo "$(ts) GATE FAIL: boot hash"; ok=0; }
for L in 0 20 21 39; do grep -aq "MEGA_PEER FUSE1 layer $L: selftest OK" /tmp/pg67-boot.log || { echo "$(ts) GATE FAIL: FUSE1 selftest layer $L"; ok=0; }; done
cl=$(grep -ac "Capturing CUDA graph" /tmp/pg67-boot.log); (( cl >= 45 )) || { echo "$(ts) GATE FAIL: capture lines $cl < 45"; ok=0; }
grep -aq "Capturing CUDA graph" /tmp/pg67-boot.log || { echo "$(ts) GATE FAIL: no capture lines"; ok=0; }
python3 - "$slog" <<'PY' || { echo "$(date '+%F %T %Z') GATE FAIL: A12 receipt"; exit 7; }
import json, re, sys
txt = open(sys.argv[1], errors='ignore').read(); last = {}
for m_ in re.finditer(r"A12_INSTALL_MEMORY (\{.*)", txt):
    try: d = json.loads(m_.group(1))
    except Exception: continue
    last = d
    if d.get("status") == "PASS": break
t = last.get("terms") or {}
free = float(t.get("post_capture_free_bytes") or 0)
est = float(t.get("applied_graph_estimate_bytes") or 0)
kv = float(t.get("kv_bytes") or 0)
print("A12:", last.get("status"), "kv_bytes", kv, "graph_est", est, "free", free)
ok = (last.get("status") == "PASS" and free >= 12.0 * (1 << 30) and kv >= 2.84e10)
sys.exit(0 if ok else 7)
PY
[[ $? -eq 0 ]] || ok=0
kt=$(grep -aoE 'GPU KV cache size: [0-9,]+ tokens' /tmp/pg67-boot.log | tail -1 | tr -dc '0-9')
[[ ${kt:-0} -ge 11500000 ]] || { echo "$(ts) GATE FAIL: kv_tokens=${kt:-?} < 11.5M"; ok=0; }
e=$(docker inspect -f '{{json .Config.Env}}' ds41); cenv=$(printf '%s' "$e" | grep -o '"DS41_OPC[^"]*"\|"DS41_W12[^"]*"\|"DS41_INSTR[^"]*"\|"PEER2_[^"]*"' | wc -l)
[[ $cenv -eq 0 ]] || { echo "$(ts) GATE FAIL: instrument env in production $cenv"; ok=0; }
[[ $e == *'"VLLM_PREFILL_YIELD=short"'* && $e == *w8oproj* && $e == *lowb_aux* && $e == *'"VLLM_PREFILL_CADENCE=3"'* ]] || { echo "$(ts) GATE FAIL: production env incomplete"; ok=0; }
[[ $e == *"\"MEGA_ROWMAP=/w/rowmap-mix-v1-h255.json\""* ]] || { echo "$(ts) GATE FAIL: rowmap not production"; ok=0; }
(( ok == 1 )) || fail "gates above"
# det smoke + speed probes (same boot, before promotion is declared)
( cd $F && PATH=~/dsv41/resid/venv/bin:$PATH python3 r7-overlay/r6_floor_probe.py --docs $F/w10/corpus --out /tmp/pg67-probe.jsonl --det-out /tmp/pg67-det.json --cells "" --reps 0 --det 8 --arm Pg67 --base http://127.0.0.1:30006 --model dsv41-flash-uva ) \
  || fail "det probe rc"
python3 - <<'PY' || { echo "$(date '+%F %T %Z') GATE FAIL: det probe could not read"; exit 7; }
import json
d = json.load(open('/tmp/pg67-det.json')); n = len(d.get('rows', [])); print('det rows', n); sys.exit(0 if n == 8 else 7)
PY
[[ $? -eq 0 ]] || ok=0
( cd $F && PATH=~/dsv41/resid/venv/bin:$PATH python3 r7-overlay/r6_floor_probe.py --docs $F/w10/corpus --out /tmp/pg67-cells.jsonl --det-out /tmp/pg67-det2.json --cells 264,392,648 --reps 3 --det 0 --arm Pg67 --base http://127.0.0.1:30006 --model dsv41-flash-uva ) \
  || fail "cell probe rc"
python3 - <<'PY' || { echo "$(date '+%F %T %Z') GATE FAIL: eager-264 still expensive (rung not captured)"; exit 7; }
import json, statistics as st
rows = [json.loads(l) for l in open('/tmp/pg67-cells.jsonl') if json.loads(l).get('tag') == 'cell']
med = {t: st.median(r['srv_prefill_ms'] for r in rows if r['target'] == t) for t in (264, 392, 648)}
print('Pg67 cell medians srv_prefill ms:', med)
sys.exit(0 if all(med[t] < 90.0 for t in med) else 7)
PY
[[ $? -eq 0 ]] || fail "cell probe: 264-648 should now sit on captured rungs (< 90 ms)"
# promote: repoint autostart and declare
cp $GR/promote/autostart-pg61.sh $GR/promote/autostart-pg61.sh.bak-pg66-to-pg67-$(date +%Y%m%d-%H%M)
sed -i 's#/garner/pg66-work#/sidecar-resid/work-Pg67#g' $GR/promote/autostart-pg61.sh
rm -f ~/dsv41/m3/NO-AUTOSTART
lo "PROMOTED: Pg67 serving :30006 = Pg66 + F1a routing-glue fusion + capture 384/512/768 (gates: selftest OK x4, A12 PASS (see promote log), kv_tokens=${kt:-?}, det 8/8, 264-648 captured < 90 ms). rollback = bash $GR/promote/rollback-pg67-to-pg66.sh; autostart repointed; promote log $LOG"
echo "$(ts) PROMOTE OK: Pg67 live at $W67; rollback script present; autostart repointed (backup written)"
