#!/usr/bin/env bash
# CARD H (2026-09-24) — GLM-5.3-Flash: transformers 5.16.1 derived image (image input fix, reported by Paul Torruella).
# Lane :30001, containers glmf-h-*. Station was idle at start (no containers running); nothing is restored at end (stop+keep).
# Boot 1: stock lmsysorg/sglang:v0.5.20-cu130, DFlash2 daily config -> vision gate only (expect FAIL = reproduces the report).
# Boot 2: glmf-sglang:0.5.20-tf5.16.1, same flags -> vision gate + the Card D newdf instrument set, compared byte-for-byte to
#         Card D's newdf receipts (same base image, weights, draft, flags; only transformers differs).
# Written by Milo (Hermes milo profile) for James Meadlock. Never print API_KEY.
set -uo pipefail
cd $HOME/glmf
export PYTHONUNBUFFERED=1
D=$HOME/glmf/cardH-2026-09-24; mkdir -p "$D"
REF=$HOME/glmf/cardD-2026-09-21/newdf
STOCK=lmsysorg/sglang:v0.5.20-cu130
TF5=glmf-sglang:0.5.20-tf5.16.1
S=$HOME/pin-hot-experts/scripts
TF=$HOME/window-20260913/scripts/tf_noninferiority.py
V=$HOME/glmf/tf5161
export BASE_URL=http://127.0.0.1:30001/v1 MODEL=glm-5.3-flash API_KEY=""
EXTRA="--max-mamba-cache-size 48 --speculative-dflash-block-size 7"
exec > "$D/runner.log" 2>&1
log(){ printf '%s %s\n' "$(date -u '+%FT%TZ')" "$*"; }
stop_glmf(){ local n; for n in $(docker ps --format '{{.Names}}' | grep -E '^glmf-'); do log "stop-and-keep $n"; docker stop "$n" >/dev/null; done; sleep 3; }
drop_caches(){ sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; }
wait_ready(){ local n=$1; for i in $(seq 1 240); do
    curl -sf --max-time 2 http://127.0.0.1:30001/v1/models >/dev/null 2>&1 && { log "READY $n after $((i*5))s"; return 0; }
    docker ps --format '{{.Names}}' | grep -q "^$n$" || { log "EXITED $n"; docker logs --tail 40 "$n" 2>&1 | tr '\r' '\n' | tail -20; return 5; }
    sleep 5; done; log "READY_TIMEOUT $n"; return 5; }
boot(){ local TAG=$1 IMG=$2 OUT=$D/$1; mkdir -p "$OUT"
  stop_glmf; drop_caches; log "BOOT $TAG image=$IMG"
  TAG="$TAG" MODEL=/models/GLM-5.3-Flash-NVFP4-nvidia-09b04e5e DRAFT=/models/GLM-5.3-Flash-DFlash2 IMAGE="$IMG" \
    SPEC=dflash KV=fp8_e4m3 MEM=0.85 CTX=1048576 MAXBS=16 EXTRA="$EXTRA" DOCKER_ENV='-e TORCHINDUCTOR_COMPILE_THREADS=1' \
    bash $HOME/glmf/launch-glmf.sh > "$OUT/launch.txt" 2>&1 || { cat "$OUT/launch.txt"; return 1; }
  wait_ready "glmf-$TAG" || return 1
  docker image inspect "$IMG" --format '{{.Id}}' > "$OUT/image-id.txt"
  docker exec "glmf-$TAG" python3 -c 'import sglang, transformers; print("sglang", sglang.__version__, "transformers", transformers.__version__)' > "$OUT/versions.txt" 2>&1
  docker exec "glmf-$TAG" python3 -c 'from transformers import AutoProcessor; p=AutoProcessor.from_pretrained("/model", trust_remote_code=True); print(type(p).__name__, type(getattr(p,"image_processor",None)).__name__)' >> "$OUT/versions.txt" 2>&1
  docker logs "glmf-$TAG" 2>&1 | tr '\r' '\n' | grep -E "sglang version|KV Cache is allocated|Mamba Cache|max_running_requests|multimodal|processor" | grep -v server_args | cut -c1-220 | head -20 > "$OUT/boot-excerpt.txt"
  nvidia-smi -i 1 --query-gpu=power.limit,power.draw,clocks.sm,temperature.gpu --format=csv,noheader >> "$OUT/boot-excerpt.txt"
  return 0; }

log "===== CARD H START ====="
docker ps --format '{{.Names}}' > "$D/containers-at-start.txt"

# Boot 1: stock image, vision gate only
if boot stock "$STOCK"; then
  python3 -c 'import flash_bench as fb; fb.knee(concs=(1,), reps=0)' > "$D/stock/warm.txt" 2>&1
  python3 $V/vision_gate.py > "$D/stock/vision.txt" 2>&1; log "stock vision: $(tail -1 $D/stock/vision.txt)"
else log "BOOT_FAIL stock"; fi

# Boot 2: derived image, full set
if boot tf5 "$TF5"; then
  O=$D/tf5
  python3 $V/vision_gate.py > "$O/vision-cold.txt" 2>&1; log "tf5 vision (cold): $(tail -1 $O/vision-cold.txt)"
  python3 -c 'import flash_bench as fb; fb.knee(concs=(1,2,4,8), reps=0)' > "$O/warm.txt" 2>&1
  python3 $V/vision_gate.py > "$O/vision.txt" 2>&1; log "tf5 vision: $(tail -1 $O/vision.txt)"
  TAG="tf5-low" EFFORT=low python3 c1_methods.py > "$O/c1-low.txt" 2>&1
  TAG="tf5-low" EFFORT=low python3 accept_probe.py > "$O/accept-low.txt" 2>&1
  EFFORT=low python3 tool_harness.py > "$O/tool-1.txt" 2>&1; EFFORT=low python3 tool_harness.py > "$O/tool-2.txt" 2>&1
  python3 greedy_equiv.py "$O/greedy-tf5.json" > "$O/greedy-gen.txt" 2>&1
  python3 greedy_equiv.py --compare "$REF/greedy-newdf.json" "$O/greedy-tf5.json" > "$O/greedy-vs-newdf.txt" 2>&1
  python3 greedy_equiv.py --compare $HOME/glmf/w3/greedy-w2-a.json "$O/greedy-tf5.json" > "$O/greedy-vs-w2a.txt" 2>&1
  python3 $S/long_greedy.py "$O/long-tf5.json" > "$O/long-gen.txt" 2>&1
  python3 $S/long_greedy.py --compare "$REF/long-newdf.json" "$O/long-tf5.json" > "$O/long-vs-newdf.txt" 2>&1
  LANE=tf5 BASE_URL=http://127.0.0.1:30001 python3 "$TF" $HOME/glmf/w3/greedy-w2-a.json "$O/tf-tf5.json" > "$O/tf-log.txt" 2>&1
  python3 "$TF" --compare "$REF/tf-newdf.json" "$O/tf-tf5.json" > "$O/tf-vs-newdf.txt" 2>&1
  python3 -c 'import flash_bench as fb; fb.knee(concs=(8,), reps=0)' > "$O/knee-c8-warm.txt" 2>&1
  python3 -c 'import flash_bench as fb; fb.knee(concs=(8,), reps=3)' > "$O/knee-c8.txt" 2>&1
  python3 $V/vision_gate.py > "$O/vision-after.txt" 2>&1; log "tf5 vision (end): $(tail -1 $O/vision-after.txt)"
else log "BOOT_FAIL tf5"; fi

{
 echo "== vision gate"; for f in stock/vision.txt tf5/vision-cold.txt tf5/vision.txt tf5/vision-after.txt; do printf '%s: ' $f; tail -1 "$D/$f" 2>/dev/null || echo missing; head -1 "$D/$f" 2>/dev/null | cut -c1-300; done
 echo "== versions"; for t in stock tf5; do printf '%s: ' $t; tr '\n' ' ' < "$D/$t/versions.txt"; echo; done
 echo "== greedy 20x200 tf5 vs Card D newdf (same image base, stock transformers)"; grep -h GREEDY_EQUIV "$D/tf5/greedy-vs-newdf.txt"
 echo "== greedy 20x200 tf5 vs w2-a"; grep -h GREEDY_EQUIV "$D/tf5/greedy-vs-w2a.txt"
 echo "== long greedy 8x2500 tf5 vs Card D newdf"; cat "$D/tf5/long-vs-newdf.txt"
 echo "== TF logprob tf5 vs Card D newdf"; tail -25 "$D/tf5/tf-vs-newdf.txt"
 echo "== tools"; grep -hE '[0-9]+/10' "$D/tf5/tool-1.txt" "$D/tf5/tool-2.txt" | tail -2
 echo "== C1 low"; grep -h 'tok/s' "$D/tf5/c1-low.txt" | head -5; echo "Card D newdf:"; grep -h 'tok/s' "$REF/c1-low.txt" | head -5
 echo "== accept"; tail -2 "$D/tf5/accept-low.txt"
 echo "== C8"; tail -3 "$D/tf5/knee-c8.txt"; echo "Card D newdf:"; tail -3 "$REF/knee-c8.txt"
} > "$D/VERDICT-cardH.txt" 2>&1
stop_glmf
log "===== CARD H END (glmf-* stopped and kept; nothing restored — station was idle at start) ====="
