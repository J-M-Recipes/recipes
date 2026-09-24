#!/usr/bin/env bash
# CARD H5 (2026-09-24): replay Card H's exact tf5 request order on fresh boots.
#   r1: vision, warm C1-8, vision, c1_methods, accept_probe, tool_harness x2, greedy  (Card H order)
#   r2: same order without the two vision requests (= Card D order)
# Card H's first tf5 boot gave greedy 2/20 vs Card D; three other tf5 boots gave 20/20. Is it the sequence or the boot?
set -uo pipefail
cd $HOME/glmf
D=$HOME/glmf/cardH-2026-09-24/h5; mkdir -p "$D"
exec > "$D/runner.log" 2>&1
TF=$HOME/window-20260913/scripts/tf_noninferiority.py
V=$HOME/glmf/tf5161
IMG=glmf-sglang:0.5.20-tf5.16.1
export MODEL=glm-5.3-flash API_KEY="" BASE_URL=http://127.0.0.1:30001/v1
log(){ printf '%s %s\n' "$(date -u '+%FT%TZ')" "$*"; }
wait_ready(){ for i in $(seq 1 240); do curl -sf --max-time 2 http://127.0.0.1:30001/v1/models >/dev/null 2>&1 && { log "READY $1 after $((i*5))s"; return 0; }
  docker ps --format '{{.Names}}' | grep -q "^$1$" || { log "EXITED $1"; return 5; }; sleep 5; done; return 5; }
boot(){ local T=$1
  for n in $(docker ps --format '{{.Names}}' | grep -E '^glmf-'); do docker stop "$n" >/dev/null; done; sleep 3
  sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
  TAG=$T MODEL=/models/GLM-5.3-Flash-NVFP4-nvidia-09b04e5e DRAFT=/models/GLM-5.3-Flash-DFlash2 IMAGE=$IMG SPEC=dflash KV=fp8_e4m3 MEM=0.85 CTX=1048576 MAXBS=16 \
    EXTRA="--max-mamba-cache-size 48 --speculative-dflash-block-size 7" DOCKER_ENV='-e TORCHINDUCTOR_COMPILE_THREADS=1' bash $HOME/glmf/launch-glmf.sh > "$D/launch-$T.txt" 2>&1
  wait_ready glmf-$T; }
seq_run(){ local R=$1 VIS=$2
  [[ $VIS == 1 ]] && python3 $V/vision_gate.py > "$D/$R-v1.txt" 2>&1
  python3 -c 'import flash_bench as fb; fb.knee(concs=(1,2,4,8), reps=0)' > "$D/$R-warm.txt" 2>&1
  [[ $VIS == 1 ]] && python3 $V/vision_gate.py > "$D/$R-v2.txt" 2>&1
  TAG=$R EFFORT=low python3 c1_methods.py > "$D/$R-c1.txt" 2>&1
  TAG=$R EFFORT=low python3 accept_probe.py > "$D/$R-accept.txt" 2>&1
  EFFORT=low python3 tool_harness.py > "$D/$R-tool1.txt" 2>&1; EFFORT=low python3 tool_harness.py > "$D/$R-tool2.txt" 2>&1
  python3 greedy_equiv.py "$D/greedy-$R.json" > /dev/null 2>&1
  log "$R (vision=$VIS) greedy vs CardD: $(python3 greedy_equiv.py --compare cardD-2026-09-21/newdf/greedy-newdf.json $D/greedy-$R.json | grep GREEDY) · vs CardH-tf5: $(python3 greedy_equiv.py --compare cardH-2026-09-24/tf5/greedy-tf5.json $D/greedy-$R.json | grep GREEDY)"
  LANE=$R BASE_URL=http://127.0.0.1:30001 python3 "$TF" $HOME/glmf/w3/greedy-w2-a.json "$D/tf-$R.json" > /dev/null 2>&1
  python3 "$TF" --compare cardD-2026-09-21/newdf/tf-newdf.json "$D/tf-$R.json" > "$D/tf-cmp-$R.txt" 2>&1
  log "$R TF vs CardD: $(python3 -c "import json;d=json.load(open('$D/tf-cmp-$R.txt'));print(d.get('bar'))" 2>/dev/null)"; }
log "H5 START"
boot h5r1 && seq_run r1 1
boot h5r2 && seq_run r2 0
for n in $(docker ps --format '{{.Names}}' | grep -E '^glmf-'); do docker stop "$n" >/dev/null; done
log "H5 END (glmf-* stopped)"
