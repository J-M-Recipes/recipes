#!/usr/bin/env bash
# CARD H6 (2026-09-24): gate the PINNED rebuild of glmf-sglang:0.5.20-tf5.16.1 (Dockerfile sha256 8cee99bd…, tokenizers pinned).
# Fresh boot, daily config. Vision gate, warm, C1, accept (CONT set), tools x2, greedy + TF vs Card D newdf, C8, vision again.
set -uo pipefail
cd $HOME/glmf
D=$HOME/glmf/cardH-2026-09-24/h6; mkdir -p "$D"
exec > "$D/runner.log" 2>&1
TF=$HOME/window-20260913/scripts/tf_noninferiority.py
V=$HOME/glmf/tf5161
IMG=glmf-sglang:0.5.20-tf5.16.1
REF=cardD-2026-09-21/newdf
export MODEL=glm-5.3-flash API_KEY="" BASE_URL=http://127.0.0.1:30001/v1 CONT=glmf-h6
log(){ printf '%s %s\n' "$(date -u '+%FT%TZ')" "$*"; }
log "H6 START"
for n in $(docker ps --format '{{.Names}}' | grep -E '^glmf-'); do docker stop "$n" >/dev/null; done; sleep 3
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
docker image inspect $IMG --format '{{.Id}}' > "$D/image-id.txt"
nvidia-smi -i 1 --query-gpu=power.limit,power.draw,clocks.sm,temperature.gpu --format=csv,noheader > "$D/power-before.txt"
TAG=h6 MODEL=/models/GLM-5.3-Flash-NVFP4-nvidia-09b04e5e DRAFT=/models/GLM-5.3-Flash-DFlash2 IMAGE=$IMG SPEC=dflash KV=fp8_e4m3 MEM=0.85 CTX=1048576 MAXBS=16 \
  EXTRA="--max-mamba-cache-size 48 --speculative-dflash-block-size 7" DOCKER_ENV='-e TORCHINDUCTOR_COMPILE_THREADS=1' bash $HOME/glmf/launch-glmf.sh > "$D/launch.txt" 2>&1
T0=$(date +%s)
for i in $(seq 1 240); do curl -sf --max-time 2 http://127.0.0.1:30001/v1/models >/dev/null 2>&1 && break
  docker ps --format '{{.Names}}' | grep -q '^glmf-h6$' || { log EXITED; exit 1; }; sleep 5; done
log "READY after $(( $(date +%s) - T0 ))s"
docker exec glmf-h6 python3 -c 'import sglang, transformers, tokenizers; print("sglang", sglang.__version__, "transformers", transformers.__version__, "tokenizers", tokenizers.__version__)' > "$D/versions.txt" 2>&1
python3 $V/vision_gate.py > "$D/vision-boot.txt" 2>&1; log "vision (boot): $(tail -1 $D/vision-boot.txt)"
python3 -c 'import flash_bench as fb; fb.knee(concs=(1,2,4,8,16), reps=0)' > "$D/warm.txt" 2>&1
TAG=h6 EFFORT=low python3 c1_methods.py > "$D/c1-low.txt" 2>&1
TAG=h6 EFFORT=low python3 accept_probe.py > "$D/accept-low.txt" 2>&1
EFFORT=low python3 tool_harness.py > "$D/tool-1.txt" 2>&1; EFFORT=low python3 tool_harness.py > "$D/tool-2.txt" 2>&1
python3 greedy_equiv.py "$D/greedy-h6.json" > /dev/null 2>&1
python3 greedy_equiv.py --compare $REF/greedy-newdf.json "$D/greedy-h6.json" > "$D/greedy-vs-newdf.txt" 2>&1
LANE=h6 BASE_URL=http://127.0.0.1:30001 python3 "$TF" $HOME/glmf/w3/greedy-w2-a.json "$D/tf-h6.json" > "$D/tf-log.txt" 2>&1
python3 "$TF" --compare $REF/tf-newdf.json "$D/tf-h6.json" > "$D/tf-vs-newdf.txt" 2>&1
python3 - > "$D/tf-delta.txt" <<'EOF'
import json
a=json.load(open("cardD-2026-09-21/newdf/tf-newdf.json"))["prompts"]; b=json.load(open("cardH-2026-09-24/h6/tf-h6.json"))["prompts"]
d=[abs(x-y) for k in a for x,y in zip(a[k]["logprobs"],b[k]["logprobs"])]
print("TF vs CardD newdf: tokens %d max|d| %.6f mean|d| %.6f" % (len(d), max(d), sum(d)/len(d)))
EOF
python3 -c 'import flash_bench as fb; fb.knee(concs=(8,), reps=0)' > "$D/knee-c8-warm.txt" 2>&1
python3 -c 'import flash_bench as fb; fb.knee(concs=(8,), reps=3)' > "$D/knee-c8.txt" 2>&1
python3 $V/vision_gate.py > "$D/vision-end.txt" 2>&1
nvidia-smi -i 1 --query-gpu=power.limit,power.draw,clocks.sm,temperature.gpu --format=csv,noheader > "$D/power-after.txt"
log "greedy: $(grep GREEDY $D/greedy-vs-newdf.txt) · $(cat $D/tf-delta.txt) · tools: $(grep -h SUMMARY $D/tool-*.txt | tr '\n' ' ') · vision end: $(tail -1 $D/vision-end.txt)"
log "C1: $(grep -h median $D/c1-low.txt | sed -E 's/.*\] //; s/ \(.*//' | tr '\n' ';')"
log "accept: $(tail -1 $D/accept-low.txt) · C8: $(tail -1 $D/knee-c8.txt)"
docker stop glmf-h6 >/dev/null
log "H6 END (glmf-h6 stopped)"
