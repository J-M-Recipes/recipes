#!/usr/bin/env bash
# CARD H3 (2026-09-24): does serving an image request perturb later text-only requests?
# tf5 container: TF (clean) -> 1 image request -> TF -> flush_cache -> TF -> 10 more image requests -> TF. Greedy 20x200 at the ends.
set -uo pipefail
cd $HOME/glmf
D=$HOME/glmf/cardH-2026-09-24/h3; mkdir -p "$D"
exec > "$D/runner.log" 2>&1
TF=$HOME/window-20260913/scripts/tf_noninferiority.py
V=$HOME/glmf/tf5161
REF=cardD-2026-09-21/newdf/tf-newdf.json
export MODEL=glm-5.3-flash API_KEY="" BASE_URL=http://127.0.0.1:30001/v1
log(){ printf '%s %s\n' "$(date -u '+%FT%TZ')" "$*"; }
wait_ready(){ for i in $(seq 1 240); do curl -sf --max-time 2 http://127.0.0.1:30001/v1/models >/dev/null 2>&1 && { log "READY $1 after $((i*5))s"; return 0; }
  docker ps --format '{{.Names}}' | grep -q "^$1$" || { log "EXITED $1"; return 5; }; sleep 5; done; return 5; }
tf(){ LANE=$1 BASE_URL=http://127.0.0.1:30001 python3 "$TF" $HOME/glmf/w3/greedy-w2-a.json "$D/tf-$1.json" > "$D/tf-log-$1.txt" 2>&1
  python3 "$TF" --compare "$REF" "$D/tf-$1.json" > "$D/tf-cmp-$1.txt" 2>&1
  log "TF $1: $(python3 -c "import json;d=json.load(open('$D/tf-cmp-$1.txt'));print(d.get('bar'), 'NONINF', d.get('NONINFERIOR'))" 2>/dev/null || tail -3 $D/tf-cmp-$1.txt | tr '\n' ' ')"; }
C=${1:-glmf-tf5}
log "H3 START container=$C"
for n in $(docker ps --format '{{.Names}}' | grep -E '^glmf-'); do docker stop "$n" >/dev/null; done; sleep 3
docker start "$C" >/dev/null; wait_ready "$C" || exit 1
python3 -c 'import flash_bench as fb; fb.knee(concs=(1,), reps=0)' > "$D/warm.txt" 2>&1
tf a0-clean
python3 greedy_equiv.py "$D/greedy-a0.json" > /dev/null 2>&1
log "greedy a0 vs newdf: $(python3 greedy_equiv.py --compare cardD-2026-09-21/newdf/greedy-newdf.json $D/greedy-a0.json | grep GREEDY)"
python3 $V/vision_gate.py > "$D/vision-1.txt" 2>&1; log "image req 1: $(tail -1 $D/vision-1.txt)"
tf a1-after-1-image
curl -s -X POST http://127.0.0.1:30001/flush_cache > "$D/flush.txt" 2>&1; log "flush_cache: $(head -c 200 $D/flush.txt)"
tf a2-after-flush
for i in $(seq 2 11); do python3 $V/vision_gate.py > "$D/vision-$i.txt" 2>&1; done; log "10 more image reqs: $(grep -h VISION_GATE $D/vision-*.txt | sort | uniq -c | tr '\n' ' ')"
tf a3-after-11-images
python3 greedy_equiv.py "$D/greedy-a3.json" > /dev/null 2>&1
log "greedy a3 vs newdf: $(python3 greedy_equiv.py --compare cardD-2026-09-21/newdf/greedy-newdf.json $D/greedy-a3.json | grep GREEDY) · a3 vs a0: $(python3 greedy_equiv.py --compare $D/greedy-a0.json $D/greedy-a3.json | grep GREEDY)"
docker stop "$C" >/dev/null
log "H3 END ($C stopped)"
