#!/usr/bin/env bash
# Card K2 (2026-09-29 early): the prize arms. Base = 09-22 nightly 582389ce (first image carrying #40517 ReplaySSM),
# which takes clean reverts of #39688 and #39200. Card K proved: revert #39688 on 0921 -> TF 0.000 (drift found), greedy still 1/20.
#   C1 : 0922 + revert #39688            + --enable-linear-replayssm-spec  (TF gate + ReplaySSM KV/speed)
#   C2 : 0922 + revert #39688 + #39200   + --enable-linear-replayssm-spec  (does removing the mHC fuse restore greedy 20/20?)
#   C1n: 0922 + revert #39688            (no ReplaySSM)  -> isolates ReplaySSM's own greedy/TF effect vs C1
# Gates per arm as Card K. Stop-and-keep. Nothing restored.
set -uo pipefail
cd $HOME/glmf
D=$HOME/glmf/cardK2-2026-09-29; mkdir -p "$D"; exec >> "$D/runner.log" 2>&1
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
CKPT=$MODELS/GLM-5.3-Flash-NVFP4-nvidia-09b04e5e; DRAFT=$MODELS/GLM-5.3-Flash-DFlash2
BASE_KNOBS="--max-mamba-cache-size 48 --speculative-dflash-block-size 7"; ENV_BASE="-e TORCHINDUCTOR_COMPILE_THREADS=1"
TF=$HOME/window-20260913/scripts/tf_noninferiority.py; CTRL_TF=$HOME/glmf/w5-ctrl/tf-w5-ctrl.json
export BASE_URL=http://127.0.0.1:30001/v1 MODEL_NAME=glm-5.3-flash API_KEY=none
IMG_0922=glmf-sglang:nightly-dev-cu13-20260922-582389ce-tf5.16.1
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
gates(){ local TAG=$1 OUT=$D/$1; export MODEL=glm-5.3-flash
  python3 -c 'import flash_bench as fb; fb.knee(concs=(1,), reps=0)' > "$OUT/warm.txt" 2>&1
  python3 greedy_equiv.py "$OUT/greedy-$TAG.json" > "$OUT/greedy-gen.txt" 2>&1 || log "GREEDY_GEN_FAIL $TAG"
  test -f "$OUT/greedy-$TAG.json" && python3 greedy_equiv.py --compare $HOME/glmf/w3/greedy-w2-a.json "$OUT/greedy-$TAG.json" > "$OUT/greedy-vs-w2.txt" 2>&1
  LANE="$TAG" BASE_URL=http://127.0.0.1:30001 python3 "$TF" $HOME/glmf/w3/greedy-w2-a.json "$OUT/tf-$TAG.json" > "$OUT/tf-gen.txt" 2>&1 || log "TF_GEN_FAIL $TAG"
  test -f "$OUT/tf-$TAG.json" && python3 "$TF" --compare "$CTRL_TF" "$OUT/tf-$TAG.json" > "$OUT/tf-compare.txt" 2>&1
  EFFORT=low python3 tool_harness.py > "$OUT/tool-1.txt" 2>&1
  TAG="$TAG-low" EFFORT=low python3 c1_methods.py > "$OUT/c1-low.txt" 2>&1
  unset MODEL
  log "ARM $TAG: $(grep -ho 'GREEDY_EQUIV identical=[0-9/]*' $OUT/greedy-vs-w2.txt 2>/dev/null) | TF $(grep -oE '"mean": [0-9.e-]+|"p99": [0-9.e-]+' $OUT/tf-compare.txt 2>/dev/null | tr '\n' ' ') | tools $(grep -ho 'tool_ok=[0-9/]*' $OUT/tool-1.txt) | C1 $(grep -h recipe-method $OUT/c1-low.txt | grep -oE 'median [0-9.]+') code $(grep -h 'code 400' $OUT/c1-low.txt | grep -oE 'median [0-9.]+') | KV $(docker logs glmf-K2-$TAG 2>&1 | tr '\r' '\n' | grep -oE 'KV Cache.*#tokens: [0-9]+' | tail -1) | SSM $(docker logs glmf-K2-$TAG 2>&1 | tr '\r' '\n' | grep -oE 'intermediate_ssm_state_cache size: [0-9.]+GB' | tail -1)"; }
arm(){ local tag=$1 img=$2; shift 2; local extra="$BASE_KNOBS $*"; mkdir -p "$D/$tag"; stop_glmf; unset MODEL
  TAG="K2-$tag" MODEL=$CKPT DRAFT=$DRAFT IMAGE="$img" SPEC=dflash KV=fp8_e4m3 MEM=0.85 CTX=1048576 MAXBS=16 EXTRA="$extra" DOCKER_ENV="$ENV_BASE" bash launch-glmf.sh > "$D/$tag/launch.txt" 2>&1
  if wait_ready "glmf-K2-$tag"; then
    docker exec "glmf-K2-$tag" sh -c "python3 -c 'import sglang;print(\"sglang\",sglang.__version__)'; sha256sum $SGL/srt/models/glm5_next.py $SGL/kernels/ops/attention/fla/kda.py $SGL/srt/layers/communicator_mhc.py; ls $SGL/srt/layers/attention/mamba/prefill_track_metadata.py 2>&1" > "$D/$tag/version.txt" 2>&1
    gates "$tag"
  else log "BOOT FAIL $tag"; fi; }
log "===== CARD K2 (0922 + reverts + ReplaySSM) START ====="
other=$(docker ps --format '{{.Names}}' | grep -vE "^glmf-" || true); [[ -n $other ]] && { log "REFUSE foreign container: $other"; exit 4; }
[[ -d $OV/C1-revert-39688 && -d $OV/C2-revert-both ]] || { log "NO OVERLAYS"; exit 3; }
docker image inspect "$IMG_0922" >/dev/null 2>&1 || { log "NO BASE IMAGE $IMG_0922"; exit 3; }
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
build_overlay "$IMG_0922" "$OV/C1-revert-39688" glmf-sglang:0922-582389ce-revert39688-tf5.16.1 || exit 2
build_overlay "$IMG_0922" "$OV/C2-revert-both"  glmf-sglang:0922-582389ce-revert39688-39200-tf5.16.1 || exit 2
arm C1  glmf-sglang:0922-582389ce-revert39688-tf5.16.1       --enable-linear-replayssm-spec
arm C2  glmf-sglang:0922-582389ce-revert39688-39200-tf5.16.1 --enable-linear-replayssm-spec
arm C1n glmf-sglang:0922-582389ce-revert39688-tf5.16.1
stop_glmf
{ echo; echo "## $(date '+%F %H:%M CDT') — Card K2: 0922-582389ce + revert #39688 (+#39200) + ReplaySSM (GLM Flash)"
  echo "worker: Milo (hermes milo) · James: go 2026-09-29 ~02:35 · Card K result: revert #39688 on 0921 -> TF 0.000 / greedy 1/20; gate-only 0.138"
  grep -h "ARM \|BOOT FAIL\|BUILD FAIL\|built " "$D/runner.log" | cut -c1-500
  echo "receipts $D · glmf-K2-* stopped-and-kept · nothing restored"; } >> $HOME/dsv41/results/ledger.md
log "===== CARD K2 END ====="
