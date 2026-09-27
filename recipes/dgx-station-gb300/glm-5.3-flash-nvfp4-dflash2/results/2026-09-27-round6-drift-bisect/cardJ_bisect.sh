#!/usr/bin/env bash
# Card J (2026-09-27): bisect the GLM Flash nightly TF drift between v0.5.20 (94602c9c, 09-18) and 425a1f8f (09-27).
# Rungs chosen to partition the GLM/KDA/spec PRs merged in the gap:
#   r1 0921-0f6761b5 : +#39200 mHC attn->MLP fuse (09-20)                      [no #40517]
#   r2 0922-582389ce : +#40517 ReplaySSM-GLM, +#40607 forget-gate nvCUTEDSL fix (09-21)
#   r3 0923-06008c17 : +#39524 fused-KDA-verify conv, #33778 GDN QKV verify, #40685 KDA beta sigmoid PTX (09-22)
#   r4 0925-8ca82118 : +#39816 Cute-DSL AR fusion, #41194 NextN plan (09-24/25)
#   (0927-425a1f8f already measured: drift 0.139 / greedy 1/20)
# Per rung: derive transformers 5.16.1 (same Dockerfile as release), boot daily config, warm, greedy 20 vs w2-a, TF vs w5-ctrl, tools x1. No speed rows.
# Stop-and-keep. :30006 v21 stopped for the window, restored at end.
set -uo pipefail
cd /home/milo/glmf
D=/home/milo/glmf/cardJ-2026-09-27; mkdir -p "$D"; exec >> "$D/runner.log" 2>&1
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
DSV=dsv41-vllm-v21-fused-split-BOUND-REF
CKPT=/models/GLM-5.3-Flash-NVFP4-nvidia-09b04e5e; DRAFT=/models/GLM-5.3-Flash-DFlash2
BASE_KNOBS="--max-mamba-cache-size 48 --speculative-dflash-block-size 7"; ENV_BASE="-e TORCHINDUCTOR_COMPILE_THREADS=1"
TF=/home/milo/window-20260913/scripts/tf_noninferiority.py; CTRL_TF=/home/milo/glmf/w5-ctrl/tf-w5-ctrl.json
export BASE_URL=http://127.0.0.1:30001/v1 MODEL_NAME=glm-5.3-flash API_KEY=none
RUNGS="r1:nightly-dev-cu13-20260921-0f6761b5 r2:nightly-dev-cu13-20260922-582389ce r3:nightly-dev-cu13-20260923-06008c17 r4:nightly-dev-cu13-20260925-8ca82118"
stop_glmf(){ for c in $(docker ps --format '{{.Names}}' | grep '^glmf-' || true); do log "stop+keep $c"; docker stop "$c" >/dev/null; done; sleep 3; }
wait_ready(){ local i; sleep 5; docker ps -a --format "{{.Names}}" | grep -q "^$1$" || { log "NO_CONTAINER $1: $(head -2 $D/${1#glmf-J-}/launch.txt | tr "\n" " ")"; return 5; }; for i in $(seq 1 240); do curl -sf --max-time 2 http://127.0.0.1:30001/v1/models >/dev/null 2>&1 && { log "ready $1 i=$i"; return 0; }; docker ps --format '{{.Names}}' | grep -q "^$1$" || { log "CONTAINER_EXITED $1"; docker logs "$1" 2>&1 | tr '\r' '\n' | grep -nE "Error|Traceback|raise" | tail -8 | cut -c1-250; return 5; }; sleep 5; done; log "READY_TIMEOUT $1"; return 5; }
derive(){ local base=lmsysorg/sglang:$1 out=glmf-sglang:$1-tf5.16.1
  docker image inspect "$out" >/dev/null 2>&1 && { echo "$out"; return 0; }
  docker pull "$base" > "$D/pull-$1.log" 2>&1 || { log "PULL FAIL $1"; return 1; }
  printf 'FROM %s\nRUN pip install --no-cache-dir "transformers==5.16.1" "tokenizers==0.23.2" && python3 -c "import transformers, tokenizers; assert transformers.__version__ == \x275.16.1\x27; assert tokenizers.__version__ == \x270.23.2\x27"\n' "$base" > "$D/Dockerfile.$1"
  docker build -t "$out" -f "$D/Dockerfile.$1" "$D" > "$D/build-$1.log" 2>&1 || { log "BUILD FAIL $1"; return 1; }
  echo "$out"; }
gates(){ local TAG=$1 OUT=$D/$1; export MODEL=glm-5.3-flash
  python3 -c 'import flash_bench as fb; fb.knee(concs=(1,), reps=0)' > "$OUT/warm.txt" 2>&1
  python3 greedy_equiv.py "$OUT/greedy-$TAG.json" > "$OUT/greedy-gen.txt" 2>&1 || log "GREEDY_GEN_FAIL $TAG: $(tail -1 $OUT/greedy-gen.txt)"
  test -f "$OUT/greedy-$TAG.json" && python3 greedy_equiv.py --compare /home/milo/glmf/w3/greedy-w2-a.json "$OUT/greedy-$TAG.json" > "$OUT/greedy-vs-w2.txt" 2>&1
  LANE="$TAG" BASE_URL=http://127.0.0.1:30001 python3 "$TF" /home/milo/glmf/w3/greedy-w2-a.json "$OUT/tf-$TAG.json" > "$OUT/tf-gen.txt" 2>&1 || log "TF_GEN_FAIL $TAG"
  test -f "$OUT/tf-$TAG.json" && python3 "$TF" --compare "$CTRL_TF" "$OUT/tf-$TAG.json" > "$OUT/tf-compare.txt" 2>&1
  EFFORT=low python3 tool_harness.py > "$OUT/tool-1.txt" 2>&1
  TAG="$TAG-low" EFFORT=low python3 c1_methods.py > "$OUT/c1-low.txt" 2>&1
  log "RUNG $TAG: $(grep -ho 'GREEDY_EQUIV identical=[0-9/]*' $OUT/greedy-vs-w2.txt) | TF $(grep -oE '"mean": [0-9.e-]+|"p99": [0-9.e-]+' $OUT/tf-compare.txt | tr '\n' ' ') | tools $(grep -ho 'tool_ok=[0-9/]*' $OUT/tool-1.txt) | C1 $(grep -h recipe-method $OUT/c1-low.txt | grep -oE 'median [0-9.]+') code $(grep -h 'code 400' $OUT/c1-low.txt | grep -oE 'median [0-9.]+')"
}
log "===== CARD J (drift bisect) START (rerun 2: launcher MODEL clash fixed) ====="
other=$(docker ps --format '{{.Names}}' | grep -vE "^glmf-|^$DSV$" || true); [[ -n $other ]] && { log "REFUSE foreign container: $other"; exit 4; }
docker ps --format '{{.Names}}' | grep -q "^$DSV$" && { log "stop+keep $DSV"; docker stop "$DSV" >/dev/null; }
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
# derive all first (network-bound, ~3 min each) so boots are back-to-back
declare -A IMG
for rt in $RUNGS; do r=${rt%%:*}; t=${rt#*:}; img=$(derive "$t") && IMG[$r]=$img && log "image $r = $img" || IMG[$r]=""; done
for rt in $RUNGS; do r=${rt%%:*}; t=${rt#*:}; img=${IMG[$r]}; unset MODEL
  [[ -z $img ]] && { log "SKIP $r (no image)"; continue; }
  mkdir -p "$D/$r"; stop_glmf
  TAG="J-$r" MODEL=$CKPT DRAFT=$DRAFT IMAGE="$img" SPEC=dflash KV=fp8_e4m3 MEM=0.85 CTX=1048576 MAXBS=16 EXTRA="$BASE_KNOBS" DOCKER_ENV="$ENV_BASE" bash launch-glmf.sh > "$D/$r/launch.txt" 2>&1
  if wait_ready "glmf-J-$r"; then
    docker exec "glmf-J-$r" python3 -c "import sglang;print('sglang',sglang.__version__)" > "$D/$r/version.txt" 2>&1
    gates "$r"
  else log "BOOT FAIL $r"; fi
done
stop_glmf
{ echo; echo "## $(date '+%F %H:%M CDT') — Card J: GLM Flash nightly drift bisect (v0.5.20 94602c9c → 425a1f8f)"; echo "worker: Milo (hermes milo) · James: go 2026-09-27 ~14:50"; grep -h "RUNG\|BOOT FAIL\|SKIP\|image r" "$D/runner.log" | cut -c1-400; echo "reference: v0.5.20 controls greedy 20/20 TF mean 0.000 · 0927-425a1f8f greedy 1/20 TF mean 0.139 p99 1.93 · receipts $D · glmf-J-* stopped-and-kept · v21 restored"; } >> /home/milo/dsv41/results/ledger.md
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; docker start "$DSV" >/dev/null; t0=$SECONDS
until curl -sf -m 3 http://127.0.0.1:30006/v1/models >/dev/null 2>&1; do sleep 20; [[ $((SECONDS-t0)) -gt 1800 ]] && break; done
log "v21 rebound after $((SECONDS-t0))s"; log "===== CARD J END ====="
