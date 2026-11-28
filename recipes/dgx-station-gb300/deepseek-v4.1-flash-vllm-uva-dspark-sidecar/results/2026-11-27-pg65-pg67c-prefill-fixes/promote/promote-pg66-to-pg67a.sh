#!/bin/bash
# Promote Pg66 -> Pg67a: F1a routing-glue fusion ONLY (NO capture-ladder change; KV must be equal).
# work dir: ~/dsv41/sidecar-resid/work-Pg67a (pg66-work immutable).
# Pre-registered gates (BEFORE run, Milo): boot hash; FUSE1 selftest OK x4; capture lines >= 45 (production);
# A12 status PASS with post_capture_free >= 12.5 GiB; kv_tokens >= 11,550,000; envs production-complete,
# no DS41_OPC/W12/INSTR/PEER2_*; ladder literal == production 256-top; det 8 rows roll out.
# Rollback: promote/rollback-pg67a-to-pg66.sh
set -uo pipefail
GR=~/dsv41/garner; F=~/dsv41/sidecar-resid; W=$F/work-Pg67a
MODEL_DIR=/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170
LOG=$GR/promote/promote-pg67a-$(date +%Y%m%d-%H%M).log; exec >>"$LOG" 2>&1
ts(){ date "+%F %T %Z"; }; lo(){ echo "- $(ts) $*" >> ~/dsv41/m3/LANE-OWNER.txt; }
fail(){ echo "$(ts) GATE FAIL: $*"; lo "PROMOTE Pg67a GATE FAIL ($*) -- restoring Pg66"
  cp "$LOG" $GR/home/state/promote-pg67a-FAILED.txt 2>/dev/null
  env WORK=$W DS41_HOME=$GR/home bash $GR/launcher/launch-pg61.sh down || true; sleep 20
  env WORK=$GR/pg66-work MODEL_DIR=$MODEL_DIR DS41_HOME=$GR/home bash $GR/launcher/launch-pg61.sh up
  lo "PROMOTE aborted; Pg66 restored (verify separately)"; exit 3; }

lo "PROMOTE Pg66 -> Pg67a starting (F1a only, no rung change)"
diff -rq $GR/pg66-work/tree $W/tree 2>/dev/null | grep -v __pycache__ > /tmp/pg67a-diff.txt
[[ $(wc -l < /tmp/pg67a-diff.txt) -eq 2 ]] || fail "tree diff count $(wc -l < /tmp/pg67a-diff.txt) != 2 (want fuse1_route.py + mega_peer_hook.py)"
ld=$(python3 $F/f-overlay/r7_cg.py show $W/tree/serve/serve.sh | sed -E 's/.*: \[([^]]*)\].*/\1/')
[[ "$ld" == "1, 2, 4, 6, 8, 12, 16, 18, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 96, 128, 144, 160, 192, 224, 256" ]] || fail "ladder literal changed: '$ld'"
touch ~/dsv41/m3/NO-AUTOSTART
env WORK=$GR/pg66-work DS41_HOME=$GR/home bash $GR/launcher/launch-pg61.sh down || fail "down pg66 rc"
sleep 20
for i in $(seq 1 60); do docker ps -a --format '{{.Names}}' | grep -q '^ds41\(-sidecar\)\?$' || break; sleep 2; done
sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; sleep 45
env WORK=$W MODEL_DIR=$MODEL_DIR DS41_HOME=$GR/home bash $GR/launcher/launch-pg61.sh up || fail "up Pg67a (see $LOG)"
docker logs ds41 > /tmp/pg67a-boot.log 2>&1; cp /tmp/pg67a-boot.log $F/results-Pg67a-boot.log
ok=1
grep -aq "verify OK: gumbel 6eac3e134833 rsu edc8f8f4c425 libw6mk b54168223847560b PEER3_SPINS=1000000000000" "$LOG" || { echo "$(ts) GATE FAIL: boot hash"; ok=0; }
for L in 0 20 21 39; do grep -aq "MEGA_PEER FUSE1 layer $L: selftest OK" /tmp/pg67a-boot.log || { echo "$(ts) GATE FAIL: FUSE1 selftest layer $L"; ok=0; }; done
cl=$(grep -ac "Capturing CUDA graph" /tmp/pg67a-boot.log); (( cl >= 45 )) || { echo "$(ts) GATE FAIL: capture lines $cl < 45"; ok=0; }
python3 - <<'PY' || { echo "$(date '+%F %T %Z') GATE FAIL: A12 receipt"; exit 7; }
import json, re
txt = open('/tmp/pg67a-boot.log', errors='ignore').read(); last = {}
for m_ in re.finditer(r"A12_INSTALL_MEMORY (\{.*)", txt):
    try: d = json.loads(m_.group(1))
    except Exception: continue
    last = d
    if d.get("status") == "PASS": break
t = last.get("terms") or {}
free = float(t.get("post_capture_free_bytes") or 0); kv = float(t.get("kv_bytes") or 0); est = float(t.get("applied_graph_estimate_bytes") or 0)
print("A12:", last.get("status"), "kv_bytes", kv, "graph_est", est, "free", free)
okp = (last.get("status") == "PASS" and free >= 12.5 * (1 << 30) and kv >= 2.84e10)
exit(0 if okp else 7)
PY
[[ $? -eq 0 ]] || ok=0
kt=$(grep -aoE 'GPU KV cache size: [0-9,]+ tokens' /tmp/pg67a-boot.log | tail -1 | tr -dc '0-9')
[[ ${kt:-0} -ge 11550000 ]] || { echo "$(ts) GATE FAIL: kv_tokens=${kt:-?} < 11.55M (KV must be equal)"; ok=0; }
e=$(docker inspect -f '{{json .Config.Env}}' ds41)
bad=$(printf '%s' "$e" | grep -o '"DS41_OPC[^"]*"\|"DS41_W12[^"]*"\|"DS41_INSTR[^"]*"\|"PEER2_[^"]*"' | wc -l)
[[ $bad -eq 0 ]] || { echo "$(ts) GATE FAIL: instrument env present: $bad"; ok=0; }
[[ $e == *'"VLLM_PREFILL_YIELD=short"'* && $e == *w8oproj* && $e == *lowb_aux* && $e == *'"VLLM_PREFILL_CADENCE=3"'* ]] || { echo "$(ts) GATE FAIL: production env incomplete"; ok=0; }
[[ $e == *"\"MEGA_ROWMAP=/w/rowmap-mix-v1-h255.json\""* ]] || { echo "$(ts) GATE FAIL: not production rowmap"; ok=0; }
(( ok == 1 )) || fail "gates above"
( cd $F && PATH=~/dsv41/resid/venv/bin:$PATH python3 r7-overlay/r6_floor_probe.py --docs $F/w10/corpus --out /tmp/pg67a-probe.jsonl --det-out /tmp/pg67a-det.json --cells "" --reps 0 --det 8 --arm Pg67a --base http://127.0.0.1:30006 --model dsv41-flash-uva ) || fail "det probe rc"
python3 -c "import json; d=json.load(open('/tmp/pg67a-det.json')); n=len(d.get('rows',[])); print('det rows',n); exit(0 if n==8 else 7)" || fail "det rows != 8"
cp $GR/promote/autostart-pg61.sh $GR/promote/autostart-pg61.sh.bak-pg66-to-pg67a-$(date +%Y%m%d-%H%M)
sed -i 's#/garner/pg66-work#/sidecar-resid/work-Pg67a#g' $GR/promote/autostart-pg61.sh
rm -f ~/dsv41/m3/NO-AUTOSTART
lo "PROMOTED: Pg67a serving :30006 = Pg66 + F1a routing-glue fusion ONLY (ladder production-untouched; gates: selftest OK x4, A12 PASS, kv_tokens=${kt:-?}, det 8/8). rollback = bash $GR/promote/rollback-pg67a-to-pg66.sh; autostart repointed; promote log $LOG"
echo "$(ts) PROMOTE OK: Pg67a live at $W"
