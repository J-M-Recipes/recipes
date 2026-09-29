#!/usr/bin/env bash
# Card K (2026-09-28, overnight, unattended): does reverting sglang #39688 (c8eb54c4, "Fuse GLM-5.3-Flash KDA
# projections and prefill metadata") on the drifted 09-21 nightly 0f6761b5 restore v0.5.20 numerics?
# Arms (one axis each, all on the 0921-0f6761b5 image + transformers 5.16.1):
#   ctl : 0f6761b5 unmodified            -> expect greedy 1/20, TF 0.139 (re-confirms the drifted baseline in-window)
#   A   : 0f6761b5 + full revert of #39688 (10 runtime files, prefill_track_metadata.py removed)
#   B   : 0f6761b5 + gate-only (_can_fuse_proj returns False for quantized models; pre-#39688 gate; leaves kernel/metadata edits)
#   C   : (only if A is clean) 0927-425a1f8f + full revert of #39688 + --enable-linear-replayssm-spec = the actual prize
# Gates per arm: warm C1, greedy 20 vs w2-a, TF vs w5-ctrl (v0.5.20), tools x1, c1_methods low (speed row).
# Stop-and-keep. Foreign containers refused. No restore of anything at end (no-restore-on-done rule).
set -uo pipefail
cd $HOME/glmf
D=$HOME/glmf/cardK-2026-09-28; mkdir -p "$D"; exec >> "$D/runner.log" 2>&1
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
CKPT=$MODELS/GLM-5.3-Flash-NVFP4-nvidia-09b04e5e; DRAFT=$MODELS/GLM-5.3-Flash-DFlash2
BASE_KNOBS="--max-mamba-cache-size 48 --speculative-dflash-block-size 7"; ENV_BASE="-e TORCHINDUCTOR_COMPILE_THREADS=1"
TF=$HOME/window-20260913/scripts/tf_noninferiority.py; CTRL_TF=$HOME/glmf/w5-ctrl/tf-w5-ctrl.json
export BASE_URL=http://127.0.0.1:30001/v1 MODEL_NAME=glm-5.3-flash API_KEY=none
IMG_0921=glmf-sglang:nightly-dev-cu13-20260921-0f6761b5-tf5.16.1
IMG_0927=glmf-sglang:nightly-20260927-425a1f8f-tf5.16.1
OV=$D/overlays   # A-full-revert/ B-gate-only/ (python/sglang/... trees) staged by scp before launch
SGL=/sgl-workspace/sglang/python/sglang

stop_glmf(){ for c in $(docker ps --format '{{.Names}}' | grep '^glmf-' || true); do log "stop+keep $c"; docker stop "$c" >/dev/null; done; sleep 3; }
wait_ready(){ local i; sleep 5; docker ps -a --format "{{.Names}}" | grep -q "^$1$" || { log "NO_CONTAINER $1"; return 5; }
  for i in $(seq 1 300); do curl -sf --max-time 2 http://127.0.0.1:30001/v1/models >/dev/null 2>&1 && { log "ready $1 i=$i"; return 0; }
    docker ps --format '{{.Names}}' | grep -q "^$1$" || { log "CONTAINER_EXITED $1"; docker logs "$1" 2>&1 | tr '\r' '\n' | grep -nE "Error|Traceback|raise" | tail -8 | cut -c1-250; return 5; }; sleep 5; done
  log "READY_TIMEOUT $1"; return 5; }

# build_overlay <base image> <overlay dir> <out tag>: COPY overlay tree over the editable sglang checkout; delete listed files.
build_overlay(){ local base=$1 ov=$2 out=$3 ctx=$D/build-$3; rm -rf "$ctx"; mkdir -p "$ctx"
  cp -r "$ov/python" "$ctx/python"
  { echo "FROM $base"; echo "COPY python/sglang/ $SGL/"
    if [[ -f $ov/DELETE.txt ]]; then while read -r f; do [[ -n $f ]] && echo "RUN rm -f /sgl-workspace/sglang/$f"; done < "$ov/DELETE.txt"; fi
    echo "RUN python3 -c \"import sglang, sglang.srt.models.glm5_next as g; print('overlay import ok', sglang.__version__)\""
  } > "$ctx/Dockerfile"
  docker build -t "$out" -f "$ctx/Dockerfile" "$ctx" > "$D/build-$3.log" 2>&1 || { log "BUILD FAIL $out: $(tail -3 $D/build-$3.log | tr '\n' ' ')"; return 1; }
  log "built $out"; }

gates(){ local TAG=$1 OUT=$D/$1; export MODEL=glm-5.3-flash
  python3 -c 'import flash_bench as fb; fb.knee(concs=(1,), reps=0)' > "$OUT/warm.txt" 2>&1
  python3 greedy_equiv.py "$OUT/greedy-$TAG.json" > "$OUT/greedy-gen.txt" 2>&1 || log "GREEDY_GEN_FAIL $TAG: $(tail -1 $OUT/greedy-gen.txt)"
  test -f "$OUT/greedy-$TAG.json" && python3 greedy_equiv.py --compare $HOME/glmf/w3/greedy-w2-a.json "$OUT/greedy-$TAG.json" > "$OUT/greedy-vs-w2.txt" 2>&1
  LANE="$TAG" BASE_URL=http://127.0.0.1:30001 python3 "$TF" $HOME/glmf/w3/greedy-w2-a.json "$OUT/tf-$TAG.json" > "$OUT/tf-gen.txt" 2>&1 || log "TF_GEN_FAIL $TAG"
  test -f "$OUT/tf-$TAG.json" && python3 "$TF" --compare "$CTRL_TF" "$OUT/tf-$TAG.json" > "$OUT/tf-compare.txt" 2>&1
  EFFORT=low python3 tool_harness.py > "$OUT/tool-1.txt" 2>&1
  TAG="$TAG-low" EFFORT=low python3 c1_methods.py > "$OUT/c1-low.txt" 2>&1
  unset MODEL
  local g tf
  g=$(grep -ho 'GREEDY_EQUIV identical=[0-9/]*' "$OUT/greedy-vs-w2.txt" 2>/dev/null); tf=$(grep -oE '"mean": [0-9.e-]+|"p99": [0-9.e-]+' "$OUT/tf-compare.txt" 2>/dev/null | tr '\n' ' ')
  log "ARM $TAG: $g | TF $tf | tools $(grep -ho 'tool_ok=[0-9/]*' $OUT/tool-1.txt) | C1 $(grep -h recipe-method $OUT/c1-low.txt | grep -oE 'median [0-9.]+') code $(grep -h 'code 400' $OUT/c1-low.txt | grep -oE 'median [0-9.]+') | KV $(docker logs glmf-K-$TAG 2>&1 | tr '\r' '\n' | grep -oE 'KV Cache.*#tokens: [0-9]+' | tail -1)"
  echo "$tf" > "$OUT/VERDICT_TF.txt"; }

# arm <tag> <image> [extra flags...]
arm(){ local tag=$1 img=$2; shift 2; local extra="$BASE_KNOBS $*"; mkdir -p "$D/$tag"; stop_glmf; unset MODEL
  TAG="K-$tag" MODEL=$CKPT DRAFT=$DRAFT IMAGE="$img" SPEC=dflash KV=fp8_e4m3 MEM=0.85 CTX=1048576 MAXBS=16 EXTRA="$extra" DOCKER_ENV="$ENV_BASE" bash launch-glmf.sh > "$D/$tag/launch.txt" 2>&1
  if wait_ready "glmf-K-$tag"; then
    docker exec "glmf-K-$tag" sh -c "python3 -c 'import sglang;print(\"sglang\",sglang.__version__)'; sha256sum $SGL/srt/models/glm5_next.py $SGL/kernels/ops/attention/fla/kda.py; ls $SGL/srt/layers/attention/mamba/prefill_track_metadata.py 2>&1" > "$D/$tag/version.txt" 2>&1
    gates "$tag"
  else log "BOOT FAIL $tag"; fi; }

tf_mean(){ grep -oE '"mean": [0-9.e-]+' "$D/$1/tf-compare.txt" 2>/dev/null | head -1 | grep -oE '[0-9.e-]+$'; }
is_clean(){ local m; m=$(tf_mean "$1"); [[ -n $m ]] && python3 -c "import sys; sys.exit(0 if float('$m') < 1e-4 else 1)"; }

log "===== CARD K (#39688 revert test) START ====="
other=$(docker ps --format '{{.Names}}' | grep -vE "^glmf-" || true); [[ -n $other ]] && { log "REFUSE foreign container running: $other"; exit 4; }
[[ -d $OV/A-full-revert && -d $OV/B-gate-only ]] || { log "NO OVERLAYS at $OV"; exit 3; }
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

build_overlay "$IMG_0921" "$OV/A-full-revert" glmf-sglang:0921-0f6761b5-revert39688-tf5.16.1 || exit 2
build_overlay "$IMG_0921" "$OV/B-gate-only"   glmf-sglang:0921-0f6761b5-gateonly39688-tf5.16.1 || exit 2

arm ctl "$IMG_0921"
arm A   glmf-sglang:0921-0f6761b5-revert39688-tf5.16.1
arm B   glmf-sglang:0921-0f6761b5-gateonly39688-tf5.16.1

if is_clean A; then
  log "A CLEAN -> building C (0927 + revert + ReplaySSM)"
  if build_overlay "$IMG_0927" "$OV/A-full-revert" glmf-sglang:0927-425a1f8f-revert39688-tf5.16.1; then
    arm C glmf-sglang:0927-425a1f8f-revert39688-tf5.16.1 --enable-linear-replayssm-spec
  fi
elif is_clean B; then
  log "B CLEAN (A not) -> building C-gate (0927 + gate-only + ReplaySSM)"
  if build_overlay "$IMG_0927" "$OV/B-gate-only" glmf-sglang:0927-425a1f8f-gateonly39688-tf5.16.1; then
    arm C glmf-sglang:0927-425a1f8f-gateonly39688-tf5.16.1 --enable-linear-replayssm-spec
  fi
else
  log "NEITHER A NOR B CLEAN -> #39688 is not (alone) the drift; next suspect #39695 (9f21fbc3). No C arm."
fi
stop_glmf
{ echo; echo "## $(date '+%F %H:%M CDT') — Card K: revert sglang #39688 on 0921-0f6761b5 (GLM Flash TF drift)"
  echo "worker: Milo (hermes milo) · James: go 2026-09-28 ~20:25, unattended overnight"
  grep -h "ARM \|BOOT FAIL\|BUILD FAIL\|CLEAN\|NEITHER\|built " "$D/runner.log" | cut -c1-400
  echo "reference: v0.5.20 greedy 20/20 TF 0.000 · 0921 drift greedy 1/20 TF 0.139 · receipts $D · glmf-K-* stopped-and-kept · nothing restored"; } >> $HOME/dsv41/results/ledger.md
log "===== CARD K END ====="
