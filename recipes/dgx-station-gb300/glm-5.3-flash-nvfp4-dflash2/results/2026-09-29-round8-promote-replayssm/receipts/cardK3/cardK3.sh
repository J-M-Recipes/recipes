#!/usr/bin/env bash
# Card K3 (2026-09-29): close the Round 7 open items on the ReplaySSM candidate.
#   D    : 0922-582389ce + revert #39688 + revert #39200 (VERIFIED overlay: communicator_mhc.py = pre-2fa6b94e blob) + ReplaySSM
#          -> does the greedy 1/20 residual vs the W2 reference close?
#   F-ar : v0.5.20 release image, SPEC=none (AR), daily knobs minus DFlash -> greedy vs W2 (DFlash ref): instrument floor for DFlash-vs-AR
#   E    : C1 candidate image (0922 + revert #39688 + ReplaySSM) deliberate self-repeat: greedy + TF, and EFFORT=max c1_methods
#   E-ctl: v0.5.20 daily (DFlash) EFFORT=max c1_methods same window (max-effort trade, paired)
# Gates: greedy 20 vs W2, TF vs v0.5.20 ctrl, tools x1, c1 low; E/E-ctl add c1 max. Stop-and-keep. Nothing restored.
set -uo pipefail
cd $HOME/glmf
D=$HOME/glmf/cardK3-2026-09-29; mkdir -p "$D"; exec >> "$D/runner.log" 2>&1
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
CKPT=$MODELS/GLM-5.3-Flash-NVFP4-nvidia-09b04e5e; DRAFT=$MODELS/GLM-5.3-Flash-DFlash2
KNOBS_DFLASH="--max-mamba-cache-size 48 --speculative-dflash-block-size 7"; KNOBS_AR="--max-mamba-cache-size 48"; ENV_BASE="-e TORCHINDUCTOR_COMPILE_THREADS=1"
TF=$HOME/window-20260913/scripts/tf_noninferiority.py; CTRL_TF=$HOME/glmf/w5-ctrl/tf-w5-ctrl.json
export BASE_URL=http://127.0.0.1:30001/v1 MODEL_NAME=glm-5.3-flash API_KEY=none
IMG_0922=glmf-sglang:nightly-dev-cu13-20260922-582389ce-tf5.16.1
IMG_C1=glmf-sglang:0922-582389ce-revert39688-tf5.16.1
IMG_REL=glmf-sglang:0.5.20-tf5.16.1
OV=$D/overlays; SGL=/sgl-workspace/sglang/python/sglang
stop_glmf(){ for c in $(docker ps --format '{{.Names}}' | grep '^glmf-' || true); do log "stop+keep $c"; docker stop "$c" >/dev/null; done; sleep 3; }
wait_ready(){ local i; sleep 5; docker ps -a --format "{{.Names}}" | grep -q "^$1$" || { log "NO_CONTAINER $1"; return 5; }
  for i in $(seq 1 300); do curl -sf --max-time 2 http://127.0.0.1:30001/v1/models >/dev/null 2>&1 && { log "ready $1 i=$i"; return 0; }
    docker ps --format '{{.Names}}' | grep -q "^$1$" || { log "CONTAINER_EXITED $1"; docker logs "$1" 2>&1 | tr '\r' '\n' | grep -nE "Error|Traceback|raise" | tail -8 | cut -c1-250; return 5; }; sleep 5; done
  log "READY_TIMEOUT $1"; return 5; }
build_overlay(){ local base=$1 ov=$2 out=$3 ctx=$D/build-$3; rm -rf "$ctx"; mkdir -p "$ctx"; cp -r "$ov/python" "$ctx/python"
  { echo "FROM $base"; echo "COPY python/sglang/ $SGL/"
    if [[ -f $ov/DELETE.txt ]]; then while read -r f; do [[ -n $f ]] && echo "RUN rm -f /sgl-workspace/sglang/$f"; done < "$ov/DELETE.txt"; fi
    echo "RUN python3 -c \"import sglang, sglang.srt.models.glm5_next as g; print('overlay import ok', sglang.__version__)\""; } > "$ctx/Dockerfile"
  docker build -t "$out" -f "$ctx/Dockerfile" "$ctx" > "$D/build-$3.log" 2>&1 || { log "BUILD FAIL $out: $(grep -E 'Error|error' $D/build-$3.log | tail -2 | tr '\n' ' ' | cut -c1-300)"; return 1; }
  log "built $out"; }
gates(){ local TAG=$1 OUT=$D/$1 MAXEFF=${2:-0}; export MODEL=glm-5.3-flash
  python3 -c 'import flash_bench as fb; fb.knee(concs=(1,), reps=0)' > "$OUT/warm.txt" 2>&1
  python3 greedy_equiv.py "$OUT/greedy-$TAG.json" > "$OUT/greedy-gen.txt" 2>&1 || log "GREEDY_GEN_FAIL $TAG"
  test -f "$OUT/greedy-$TAG.json" && python3 greedy_equiv.py --compare $HOME/glmf/w3/greedy-w2-a.json "$OUT/greedy-$TAG.json" > "$OUT/greedy-vs-w2.txt" 2>&1
  LANE="$TAG" BASE_URL=http://127.0.0.1:30001 python3 "$TF" $HOME/glmf/w3/greedy-w2-a.json "$OUT/tf-$TAG.json" > "$OUT/tf-gen.txt" 2>&1 || log "TF_GEN_FAIL $TAG"
  test -f "$OUT/tf-$TAG.json" && python3 "$TF" --compare "$CTRL_TF" "$OUT/tf-$TAG.json" > "$OUT/tf-compare.txt" 2>&1
  EFFORT=low python3 tool_harness.py > "$OUT/tool-1.txt" 2>&1
  TAG="$TAG-low" EFFORT=low python3 c1_methods.py > "$OUT/c1-low.txt" 2>&1
  [[ $MAXEFF == 1 ]] && TAG="$TAG-max" EFFORT=max python3 c1_methods.py > "$OUT/c1-max.txt" 2>&1
  unset MODEL
  log "ARM $TAG: $(grep -ho 'GREEDY_EQUIV identical=[0-9/]*' $OUT/greedy-vs-w2.txt 2>/dev/null) | TF $(grep -oE '"mean": [0-9.e-]+|"p99": [0-9.e-]+' $OUT/tf-compare.txt 2>/dev/null | tr '\n' ' ') | tools $(grep -ho 'tool_ok=[0-9/]*' $OUT/tool-1.txt) | C1low $(grep -h recipe-method $OUT/c1-low.txt | grep -oE 'median [0-9.]+') code $(grep -h 'code 400' $OUT/c1-low.txt | grep -oE 'median [0-9.]+')$([[ $MAXEFF == 1 ]] && echo " | C1max $(grep -h recipe-method $OUT/c1-max.txt | grep -oE 'median [0-9.]+') code $(grep -h 'code 400' $OUT/c1-max.txt | grep -oE 'median [0-9.]+')") | KV $(docker logs glmf-K3-$TAG 2>&1 | tr '\r' '\n' | grep -oE 'KV Cache.*#tokens: [0-9]+' | tail -1)"; }
# arm <tag> <image> <spec> <maxeff> [extra flags...]
arm(){ local tag=$1 img=$2 spec=$3 maxeff=$4; shift 4; local knobs; [[ $spec == none ]] && knobs="$KNOBS_AR $*" || knobs="$KNOBS_DFLASH $*"
  mkdir -p "$D/$tag"; stop_glmf; unset MODEL
  TAG="K3-$tag" MODEL=$CKPT DRAFT=$DRAFT IMAGE="$img" SPEC=$spec KV=fp8_e4m3 MEM=0.85 CTX=1048576 MAXBS=16 EXTRA="$knobs" DOCKER_ENV="$ENV_BASE" bash launch-glmf.sh > "$D/$tag/launch.txt" 2>&1
  if wait_ready "glmf-K3-$tag"; then
    docker exec "glmf-K3-$tag" sh -c "python3 -c 'import sglang;print(\"sglang\",sglang.__version__)'; sha256sum $SGL/srt/models/glm5_next.py $SGL/kernels/ops/attention/fla/kda.py $SGL/srt/layers/communicator_mhc.py; ls $SGL/srt/layers/attention/mamba/prefill_track_metadata.py 2>&1" > "$D/$tag/version.txt" 2>&1
    gates "$tag" "$maxeff"
  else log "BOOT FAIL $tag"; fi; }
log "===== CARD K3 (#39200 revert for real · AR floor · candidate self-repeat + max-effort) START ====="
other=$(docker ps --format '{{.Names}}' | grep -vE "^glmf-" || true); [[ -n $other ]] && { log "REFUSE foreign container: $other"; exit 4; }
[[ -d $OV/D-revert-both ]] || { log "NO OVERLAY"; exit 3; }
for i in "$IMG_0922" "$IMG_C1" "$IMG_REL"; do docker image inspect "$i" >/dev/null 2>&1 || { log "NO IMAGE $i"; exit 3; }; done
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
build_overlay "$IMG_0922" "$OV/D-revert-both" glmf-sglang:0922-582389ce-revert39688-39200-tf5.16.1 || exit 2
# pre-flight: the overlay must actually carry the #39200 revert this time
docker run --rm --entrypoint sh glmf-sglang:0922-582389ce-revert39688-39200-tf5.16.1 -c "sha256sum $SGL/srt/layers/communicator_mhc.py" | tee "$D/D-mhc-sha.txt" | grep -q f8fd0ac97809ad6e || { log "OVERLAY D DID NOT APPLY (communicator_mhc.py sha != pre-39200). ABORT."; exit 2; }
log "overlay D verified: communicator_mhc.py = pre-#39200 blob"
arm D     glmf-sglang:0922-582389ce-revert39688-39200-tf5.16.1 dflash 0 --enable-linear-replayssm-spec
arm F-ar  "$IMG_REL" none   0
arm E     "$IMG_C1"  dflash 1 --enable-linear-replayssm-spec
arm E-ctl "$IMG_REL" dflash 1
stop_glmf
{ echo; echo "## $(date '+%F %H:%M CDT') — Card K3: #39200 revert (verified overlay) · v0.5.20 AR floor · candidate self-repeat + max-effort"
  echo "worker: Milo (hermes milo) · James: go 2026-09-29 ~08:50"
  grep -h "ARM \|BOOT FAIL\|BUILD FAIL\|built \|verified\|ABORT" "$D/runner.log" | cut -c1-600
  echo "receipts $D · glmf-K3-* stopped-and-kept · nothing restored"; } >> $HOME/dsv41/results/ledger.md
log "===== CARD K3 END ====="
