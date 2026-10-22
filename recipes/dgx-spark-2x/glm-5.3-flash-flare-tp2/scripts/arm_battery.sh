#!/bin/bash
# arm_battery.sh TAG [MAXC]  -- runs on spark5 against 127.0.0.1:8003. Same instrument every arm.
set -u
WORK=${WORK:-$HOME/flare-prep-glm}   # campaign dir on the rank-0 node (logs, results, tools/)
TAG=$1; MAXC=${2:-4}
T=$WORK/tools; B=http://127.0.0.1:8003/v1
D=$WORK/results/$TAG; mkdir -p "$D"; cd "$D"
log(){ echo "$(date '+%F %T') $*" | tee -a "$D/battery.log"; }
dump(){ curl -s -m 10 http://127.0.0.1:8003/metrics > "$D/metrics-$1.txt"; }
M=$(curl -s -m 10 $B/models | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])')
log "BATTERY START tag=$TAG model=$M maxc=$MAXC"; curl -s -m 10 $B/models > models.json; dump start
free -g | awk 'NR==2{print "head MemAvailable GiB", $7}' >> battery.log
python3 $T/glm_gates.py $B > gates.json 2>&1; log "$(tail -1 gates.json)"; dump after-gates
PROSE="Write a detailed, multi-paragraph explanation of how a CPU branch predictor works, including history tables and misprediction costs."
CODE="Write a complete Python module implementing an LRU cache with TTL expiry, type hints, docstrings, and a small pytest test suite. Output only code."
JSON="Return a JSON array of 40 objects, each with fields id (int), city (string), country (string), population (int), lat (float), lon (float). Output only JSON."
python3 $T/probe.py $B "$M" 1 128 "$PROSE" >/dev/null 2>&1; python3 $T/probe.py $B "$M" $MAXC 128 "$CODE" >/dev/null 2>&1
for cls in PROSE CODE JSON; do P=${!cls}
  for c in 1 1 2 4 $( [ $MAXC -ge 8 ] && echo 8 ); do
    f="probe-$cls-c$c.$(date +%s%N).jsonl"; python3 $T/probe.py $B "$M" $c 512 "$P" > "$f" 2>&1; log "$cls C$c: $(tail -1 $f)"
  done
done
dump after-probe
python3 $T/quant_faceoff.py --base $B --label "$TAG" --out "$D/faceoff.json" > faceoff.log 2>&1; log "faceoff rc=$? $(grep RESULT faceoff.log | cut -c1-400)"
dump after-faceoff
python3 $T/decode_fix.py $B "$D/decode_fix.json" > decode_fix.log 2>&1; log "decode_fix rc=$? $(grep RESULT decode_fix.log)"
dump after-decodefix
free -g | awk 'NR==2{print "head MemAvailable GiB", $7}' >> battery.log
log "BATTERY END"
