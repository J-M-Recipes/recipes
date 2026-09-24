#!/usr/bin/env bash
# CARD H4 (2026-09-24): Card H's fresh tf5 boot scored text differently from Card D (greedy 2/20, TF mean |d| ~0.13 from token 0),
# but a docker-start of the same container matches Card D exactly (20/20, TF within bar). Question: does the FIRST request after a
# fresh boot being an image request change the model function for the whole process lifetime?
#   x: fresh container, tf5 image, first request = image, then text instruments
#   y: fresh container, tf5 image, first request = text (warm + TF + greedy), then image, then TF + greedy again
set -uo pipefail
cd $HOME/glmf
D=$HOME/glmf/cardH-2026-09-24/h4; mkdir -p "$D"
exec > "$D/runner.log" 2>&1
TF=$HOME/window-20260913/scripts/tf_noninferiority.py
V=$HOME/glmf/tf5161
IMG=glmf-sglang:0.5.20-tf5.16.1
export MODEL=glm-5.3-flash API_KEY="" BASE_URL=http://127.0.0.1:30001/v1
log(){ printf '%s %s\n' "$(date -u '+%FT%TZ')" "$*"; }
wait_ready(){ for i in $(seq 1 240); do curl -sf --max-time 2 http://127.0.0.1:30001/v1/models >/dev/null 2>&1 && { log "READY $1 after $((i*5))s"; return 0; }
  docker ps --format '{{.Names}}' | grep -q "^$1$" || { log "EXITED $1"; return 5; }; sleep 5; done; return 5; }
tf(){ LANE=$1 BASE_URL=http://127.0.0.1:30001 python3 "$TF" $HOME/glmf/w3/greedy-w2-a.json "$D/tf-$1.json" > "$D/tf-log-$1.txt" 2>&1
  python3 "$TF" --compare cardD-2026-09-21/newdf/tf-newdf.json "$D/tf-$1.json" > "$D/tf-cmp-$1.txt" 2>&1
  log "TF $1 vs CardD: $(python3 -c "import json;d=json.load(open('$D/tf-cmp-$1.txt'));print(d.get('bar'))" 2>/dev/null)"; }
gr(){ python3 greedy_equiv.py "$D/greedy-$1.json" > /dev/null 2>&1; log "greedy $1 vs CardD: $(python3 greedy_equiv.py --compare cardD-2026-09-21/newdf/greedy-newdf.json $D/greedy-$1.json | grep GREEDY)"; }
boot(){ local T=$1
  for n in $(docker ps --format '{{.Names}}' | grep -E '^glmf-'); do docker stop "$n" >/dev/null; done; sleep 3
  sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
  TAG=$T MODEL=/models/GLM-5.3-Flash-NVFP4-nvidia-09b04e5e DRAFT=/models/GLM-5.3-Flash-DFlash2 IMAGE=$IMG SPEC=dflash KV=fp8_e4m3 MEM=0.85 CTX=1048576 MAXBS=16 \
    EXTRA="--max-mamba-cache-size 48 --speculative-dflash-block-size 7" DOCKER_ENV='-e TORCHINDUCTOR_COMPILE_THREADS=1' bash $HOME/glmf/launch-glmf.sh > "$D/launch-$T.txt" 2>&1
  wait_ready glmf-$T; }
log "H4 START"
if boot h4x; then
  python3 $V/vision_gate.py > "$D/x-vision-first.txt" 2>&1; log "x: image first: $(tail -1 $D/x-vision-first.txt)"
  python3 -c 'import flash_bench as fb; fb.knee(concs=(1,2,4,8), reps=0)' > "$D/x-warm.txt" 2>&1
  tf x; gr x
fi
if boot h4y; then
  python3 -c 'import flash_bench as fb; fb.knee(concs=(1,2,4,8), reps=0)' > "$D/y-warm.txt" 2>&1
  tf y0-text-first; gr y0
  python3 $V/vision_gate.py > "$D/y-vision.txt" 2>&1; log "y: image after text: $(tail -1 $D/y-vision.txt)"
  tf y1-after-image; gr y1
fi
for n in $(docker ps --format '{{.Names}}' | grep -E '^glmf-'); do docker stop "$n" >/dev/null; done
log "H4 END (glmf-* stopped)"
