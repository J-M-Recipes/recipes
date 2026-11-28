set -u
GR=~/dsv41/garner; F=~/dsv41/sidecar-resid
MODEL_DIR=/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170
ts(){ date "+%F %T %Z"; }
# build the clean Pg67b tree from the proven PF arm (Pg67a + pf1_mm.py + patched hook)
cd $F && rm -rf work-Pg67b && cp -rL work-PF-w1-PF work-Pg67b && chmod -R u+w work-Pg67b
n=$(diff -rq $F/work-Pg67a/tree $F/work-Pg67b/tree 2>/dev/null | grep -v __pycache__ | wc -l)
[ "$n" = 2 ] || { echo "pre-refuse tree diffs $n != 2"; exit 4; }
echo "- $(ts) PROMOTE Pg67b starting (Pg67a + PF1 direct mm_mxfp8 call plan; James: 'it is acceptable' on the G2 byte-equal-pool reading)" >> ~/dsv41/m3/LANE-OWNER.txt
env WORK=$F/work-Pg67a MODEL_DIR=$MODEL_DIR DS41_HOME=$GR/home bash $GR/launcher/launch-pg61.sh down > /tmp/pg67b-pf-promote.log 2>&1 || true
sleep 20
for i in $(seq 1 60); do docker ps -a --format '{{.Names}}' | grep -q '^ds41\(-sidecar\)\?$' || break; sleep 2; done
sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; sleep 45
if ! env WORK=$F/work-Pg67b MODEL_DIR=$MODEL_DIR DS41_HOME=$GR/home bash $GR/launcher/launch-pg61.sh up >> /tmp/pg67b-pf-promote.log 2>&1; then
  echo "up failed -> restoring Pg67a"; env WORK=$F/work-Pg67a MODEL_DIR=$MODEL_DIR DS41_HOME=$GR/home bash $GR/launcher/launch-pg61.sh up >> /tmp/pg67b-pf-promote.log 2>&1; exit 5
fi
docker logs ds41 > /tmp/pg67b-pf-boot.log 2>&1
ok=1
grep -aq "verify OK: gumbel 6eac3e134833 rsu edc8f8f4c425 libw6mk b54168223847560b PEER3_SPINS=1000000000000" /tmp/pg67b-pf-promote.log || { echo "GATE boot hash"; ok=0; }
for L in 0 20 21 39; do grep -aq "MEGA_PEER FUSE1 layer $L: selftest OK" /tmp/pg67b-pf-boot.log || { echo "GATE FUSE1 selftest $L"; ok=0; }; done
grep -aq "PF1 installed" /tmp/pg67b-pf-boot.log || { echo "GATE PF1 not installed"; ok=0; }
grep -aq "PF1.*DISABLED" /tmp/pg67b-pf-boot.log && { echo "GATE PF1 DISABLED present"; ok=0; }
cl=$(grep -ac "Capturing CUDA graph" /tmp/pg67b-pf-boot.log); [ "$cl" -ge 45 ] || { echo "GATE capture lines $cl"; ok=0; }
kt=$(grep -aoE 'GPU KV cache size: [0-9,]+ tokens' /tmp/pg67b-pf-boot.log | tail -1 | tr -dc '0-9')
[ "${kt:-0}" -ge 11550000 ] || { echo "GATE kv_tokens ${kt:-?} < 11.55M"; ok=0; }
python3 - <<'PY' || { echo "GATE A12"; ok=0; }
import json, re
txt = open('/tmp/pg67b-pf-boot.log', errors='ignore').read(); last={}
for m in re.finditer(r'A12_INSTALL_MEMORY (\{.*)', txt):
    try: d = json.loads(m.group(1))
    except Exception: continue
    last = d
    if d.get('status') == 'PASS': break
t = last.get('terms') or {}
free = float(t.get('post_capture_free_bytes') or 0); kv = float(t.get('kv_bytes') or 0)
print('A12:', last.get('status'), 'kv_bytes', kv, 'free', free)
ok = (last.get('status') == 'PASS' and free >= 12.0 * (1 << 30) and kv >= 2.84e10)
exit(0 if ok else 7)
PY
e=$(docker inspect -f '{{json .Config.Env}}' ds41)
echo "$e" | grep -o '"DS41_OPC[^"]*"\|"DS41_W12[^"]*"\|"DS41_INSTR[^"]*"\|"DS41_ATTN_PROF[^"]*"\|"PEER2_[^"]*"' | grep -q . && { echo "GATE instrument env"; ok=0; }
echo "$e" | grep -q '"VLLM_PREFILL_YIELD=short"' && echo "$e" | grep -q 'w8oproj' && echo "$e" | grep -q 'lowb_aux' && echo "$e" | grep -q '"VLLM_PREFILL_CADENCE=3"' || { echo "GATE prod env incomplete"; ok=0; }
echo "$e" | grep -q '"MEGA_ROWMAP=/w/rowmap-mix-v1-h255.json"' || { echo "GATE rowmap"; ok=0; }
[ "$ok" = 1 ] || { echo "GATES FAILED -> restoring Pg67a"; echo "- $(ts) PROMOTE Pg67b GATES FAILED; restoring Pg67a" >> ~/dsv41/m3/LANE-OWNER.txt; env WORK=$F/work-Pg67b MODEL_DIR=$MODEL_DIR DS41_HOME=$GR/home bash $GR/launcher/launch-pg61.sh down >> /tmp/pg67b-pf-promote.log 2>&1; sleep 20; env WORK=$F/work-Pg67a MODEL_DIR=$MODEL_DIR DS41_HOME=$GR/home bash $GR/launcher/launch-pg61.sh up >> /tmp/pg67b-pf-promote.log 2>&1; exit 6; }
cd $F && PATH=~/dsv41/resid/venv/bin:$PATH python3 pf-overlay/r6_floor_probe.py --docs $F/w10/corpus --out /tmp/pg67b-pf-probe.jsonl --det-out /tmp/pg67b-pf-det.json --cells "264,392,648" --reps 3 --det 8 --arm Pg67b --base http://127.0.0.1:30006 --model dsv41-flash-uva >> /tmp/pg67b-pf-promote.log 2>&1
python3 - <<'PY' || { echo "GATE det/probe"; exit 7; }
import json, statistics as st
rows = [json.loads(l) for l in open('/tmp/pg67b-pf-probe.jsonl') if json.loads(l).get('tag') == 'cell']
med = {t: st.median(r['srv_prefill_ms'] for r in rows if r['target'] == t) for t in (264, 392, 648)}
d = json.load(open('/tmp/pg67b-pf-det.json')); n = len(d.get('rows', []))
print('Pg67b cell medians srv_prefill ms:', med, 'det rows', n)
exit(0 if (all(med[t] < 115.0 for t in med) and n == 8) else 7)
PY
cp $GR/promote/autostart-pg61.sh $GR/promote/autostart-pg61.sh.bak-pg67a-to-pg67b-pf-$(date +%Y%m%d-%H%M)
perl -pi -e 's|~/dsv41/sidecar-resid/work-Pg67a|~/dsv41/sidecar-resid/work-Pg67b|g' $GR/promote/autostart-pg61.sh
echo "- $(ts) PROMOTED: Pg67b serving :30006 = Pg67a + PF1 direct mm_mxfp8 call plan (measured -18.33 ms/eager step pooled on the card; gates: hash OK, FUSE1 selftests x4, PF1 installed no DISABLED, capture 45, A12 PASS, kv_tokens ${kt:-?} >= 11.55M, det 8/8, cells 264-648 < 115 ms). rollback = promote/rollback-pg67b-to-pg67a.sh; autostart repointed" >> ~/dsv41/m3/LANE-OWNER.txt
echo "PROMOTED"
