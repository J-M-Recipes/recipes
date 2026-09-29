#!/usr/bin/env bash
# Promote (2026-09-29, James: y): GLM-5.3-Flash daily lane on :30001 = 0922-582389ce + revert #39688 + ReplaySSM.
# Boot with the daily knobs, verify the patch is in-container by hash, run vision gate + greedy/TF/tools + warm C1..C32.
set -uo pipefail
cd $HOME/glmf
D=$HOME/glmf/promote-2026-09-29; mkdir -p "$D"; exec >> "$D/runner.log" 2>&1
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
CKPT=$MODELS/GLM-5.3-Flash-NVFP4-nvidia-09b04e5e; DRAFT=$MODELS/GLM-5.3-Flash-DFlash2
IMG=glmf-sglang:0922-582389ce-revert39688-tf5.16.1
NAME=glmf-DAILY-rssm-20260929
TF=$HOME/window-20260913/scripts/tf_noninferiority.py; CTRL_TF=$HOME/glmf/w5-ctrl/tf-w5-ctrl.json
SGL=/sgl-workspace/sglang/python/sglang
export BASE_URL=http://127.0.0.1:30001/v1 MODEL_NAME=glm-5.3-flash API_KEY=none
log "===== PROMOTE START ====="
other=$(docker ps --format '{{.Names}}' | grep -vE "^glmf-" || true); [[ -n $other ]] && { log "REFUSE foreign container: $other"; exit 4; }
if docker ps --format '{{.Names}}' | grep -q "^$NAME$"; then log "daily container already up (attempt-1 launch); skipping relaunch"; else
for c in $(docker ps --format '{{.Names}}' | grep '^glmf-' || true); do log "stop+keep $c"; docker stop "$c" >/dev/null; done; sleep 3
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
unset MODEL
TAG="DAILY-rssm-20260929" MODEL=$CKPT DRAFT=$DRAFT IMAGE="$IMG" SPEC=dflash KV=fp8_e4m3 MEM=0.85 CTX=1048576 MAXBS=16 \
  EXTRA="--max-mamba-cache-size 48 --speculative-dflash-block-size 7 --enable-linear-replayssm-spec" \
  DOCKER_ENV="-e TORCHINDUCTOR_COMPILE_THREADS=1" bash launch-glmf.sh > "$D/launch.txt" 2>&1
fi
sleep 5
for i in $(seq 1 300); do curl -sf --max-time 2 http://127.0.0.1:30001/v1/models >/dev/null 2>&1 && { log "ready $NAME i=$i"; break; }
  docker ps --format '{{.Names}}' | grep -q "^$NAME$" || { log "CONTAINER_EXITED"; exit 5; }; sleep 5; done
# patch-in-container gate: glm5_next.py + kda.py must be the reverted blobs, prefill_track_metadata.py must be absent
docker exec "$NAME" sh -c "python3 -c 'import sglang;print(\"sglang\",sglang.__version__)'; sha256sum $SGL/srt/models/glm5_next.py $SGL/kernels/ops/attention/fla/kda.py; ls $SGL/srt/layers/attention/mamba/prefill_track_metadata.py 2>&1" > "$D/version.txt" 2>&1
grep -q bae77af69d0b280582daeeed7631b54687b91f4073d61906f4a454b31baa8fe1 "$D/version.txt" && grep -q 5d0e0922583c158634cd8093c987c4a73aa0d28fcd52fd9d244de1956b55ceeb "$D/version.txt" && grep -q "No such file" "$D/version.txt" && log "PATCH_GATE PASS" || { log "PATCH_GATE FAIL"; exit 6; }
docker inspect "$NAME" --format '{{.Config.Image}} {{.Image}}' > "$D/image.txt"; docker logs "$NAME" 2>&1 | tr '\r' '\n' | grep -E "KV Cache.*#tokens|intermediate_ssm_state_cache|replayssm|ReplaySSM" | head -6 > "$D/boot-excerpt.txt"
export MODEL=glm-5.3-flash
python3 tf5161/vision_gate.py > "$D/vision.txt" 2>&1; grep -q "VISION_GATE PASS" "$D/vision.txt" && log "VISION_GATE PASS" || log "VISION_GATE FAIL: $(tail -1 $D/vision.txt)"
python3 greedy_equiv.py "$D/greedy-daily.json" > "$D/greedy-gen.txt" 2>&1
python3 greedy_equiv.py --compare $HOME/glmf/cardK2-2026-09-29/C1/greedy-C1.json "$D/greedy-daily.json" > "$D/greedy-vs-C1.txt" 2>&1
LANE=daily BASE_URL=http://127.0.0.1:30001 python3 "$TF" $HOME/glmf/w3/greedy-w2-a.json "$D/tf-daily.json" > "$D/tf-gen.txt" 2>&1
python3 "$TF" --compare "$CTRL_TF" "$D/tf-daily.json" > "$D/tf-compare.txt" 2>&1
EFFORT=low python3 tool_harness.py > "$D/tool-1.txt" 2>&1
python3 -c 'import flash_bench as fb; fb.knee(reps=0)' > "$D/warm-knee.txt" 2>&1
TAG=daily-low EFFORT=low python3 c1_methods.py > "$D/c1-low.txt" 2>&1
unset MODEL
log "DAILY: $(grep -ho 'GREEDY_EQUIV identical=[0-9/]*' $D/greedy-vs-C1.txt) vs C1 | TF $(grep -oE '"mean": [0-9.e-]+|"p99": [0-9.e-]+' $D/tf-compare.txt | tr '\n' ' ') | tools $(grep -ho 'tool_ok=[0-9/]*' $D/tool-1.txt) | vision $(grep -o 'VISION_GATE [A-Z]*' $D/vision.txt) | C1 $(grep -h recipe-method $D/c1-low.txt | grep -oE 'median [0-9.]+') code $(grep -h 'code 400' $D/c1-low.txt | grep -oE 'median [0-9.]+') | $(head -1 $D/boot-excerpt.txt | grep -oE '#tokens: [0-9]+')"
{ echo; echo "## $(date '+%F %H:%M CDT') — PROMOTED: GLM-5.3-Flash daily on :30001 = $IMG + ReplaySSM ($NAME)"; echo "James: y 2026-09-29 ~15:50 · Round 8 (K3): #39200 not the residual; v0.5.20 AR vs DFlash ref also 1/20 (instrument floor); candidate self-repeat 20/20 x3; max-effort flat"; grep -h "DAILY:\|GATE" "$D/runner.log" | cut -c1-500; echo "left RUNNING (daily). glmf-K*/L* stopped-and-kept."; } >> $HOME/dsv41/results/ledger.md
log "===== PROMOTE END (daily left running) ====="
