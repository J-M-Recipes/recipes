#!/usr/bin/env bash
# Card M (2026-10-01): KDA snapshot-eviction probe on the GLM-5.3-Flash daily (HelixML 2026-09-26 mechanism).
#   ctl  : daily config as promoted 2026-09-29 (mamba_max_states_per_path = -1)
#   cap2 : daily + --mamba-max-states-per-path 2
#   cap4 : daily + --mamba-max-states-per-path 4
# Per arm: warm C1, snapshot probe x2 reps (prime/warm/long/after-long/branch/loop), knee C8+C16 (same instrument as
# Round 8), gates (greedy 20 vs daily, TF vs v0.5.20 ctrl, tools x1). Stop-and-keep. Daily restarted at the end
# (docker start of the kept daily container = same-config re-bind, not a promotion).
set -uo pipefail
cd $HOME/glmf
D=$HOME/glmf/cardM-2026-10-01; mkdir -p "$D"; exec >> "$D/runner.log" 2>&1
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
CKPT=/models/GLM-5.3-Flash-NVFP4-nvidia-09b04e5e; DRAFT=/models/GLM-5.3-Flash-DFlash2
IMG=glmf-sglang:0922-582389ce-revert39688-tf5.16.1
DAILY=glmf-DAILY-rssm-20260929
KNOBS="--max-mamba-cache-size 48 --speculative-dflash-block-size 7 --enable-linear-replayssm-spec"
ENV_BASE="-e TORCHINDUCTOR_COMPILE_THREADS=1"
TF=$HOME/window-20260913/scripts/tf_noninferiority.py; CTRL_TF=$HOME/glmf/w5-ctrl/tf-w5-ctrl.json
REF_GREEDY=$HOME/glmf/cardK3-2026-09-29/E/greedy-E.json
export BASE_URL=http://127.0.0.1:30001/v1 MODEL_NAME=glm-5.3-flash API_KEY=none
stop_glmf(){ for c in $(docker ps --format '{{.Names}}' | grep '^glmf-' || true); do log "stop+keep $c"; docker stop "$c" >/dev/null; done; sleep 3; }
wait_ready(){ local i; sleep 5; docker ps -a --format "{{.Names}}" | grep -q "^$1$" || { log "NO_CONTAINER $1"; return 5; }
  for i in $(seq 1 300); do curl -sf --max-time 2 http://127.0.0.1:30001/health >/dev/null 2>&1 && { log "ready $1 i=$i"; return 0; }
    docker ps --format '{{.Names}}' | grep -q "^$1$" || { log "CONTAINER_EXITED $1"; docker logs "$1" 2>&1 | tr '\r' '\n' | grep -nE "Error|Traceback|raise" | tail -8 | cut -c1-250; return 5; }; sleep 5; done
  log "READY_TIMEOUT $1"; return 5; }
arm(){ local tag=$1; shift; local OUT=$D/$tag; mkdir -p "$OUT"; stop_glmf; unset MODEL
  TAG="M-$tag" MODEL=$CKPT DRAFT=$DRAFT IMAGE="$IMG" SPEC=dflash KV=fp8_e4m3 MEM=0.85 CTX=1048576 MAXBS=16 EXTRA="$KNOBS $*" DOCKER_ENV="$ENV_BASE" bash launch-glmf.sh > "$OUT/launch.txt" 2>&1
  wait_ready "glmf-M-$tag" || { log "BOOT FAIL $tag"; return 1; }
  docker logs "glmf-M-$tag" 2>&1 | tr '\r' '\n' | grep -E "mamba_max_states_per_path|Mamba Cache is allocated|KV Cache is allocated|max_running_requests is capped" | cut -c1-300 > "$OUT/boot.txt"
  grep -q "'mamba_max_states_per_path': ${CAP:--1}," "$OUT/boot.txt" || grep -o "'mamba_max_states_per_path': [-0-9]*" <(docker logs "glmf-M-$tag" 2>&1) | tail -1 > "$OUT/cap-check.txt"
  export MODEL=glm-5.3-flash
  python3 -c 'import flash_bench as fb; fb.knee(concs=(1,), reps=0)' > "$OUT/warm.txt" 2>&1
  for rep in 1 2; do TAG="$tag" REP=$rep OUT="$OUT/probe.jsonl" python3 kda_snapshot_probe.py > "$OUT/probe-rep$rep.txt" 2>&1 || log "PROBE_FAIL $tag rep$rep"; done
  python3 -c 'import flash_bench as fb; fb.knee(concs=(8,16), reps=3)' > "$OUT/knee.txt" 2>&1
  python3 greedy_equiv.py "$OUT/greedy-$tag.json" > "$OUT/greedy-gen.txt" 2>&1 || log "GREEDY_GEN_FAIL $tag"
  test -f "$OUT/greedy-$tag.json" && python3 greedy_equiv.py --compare "$REF_GREEDY" "$OUT/greedy-$tag.json" > "$OUT/greedy-vs-daily.txt" 2>&1
  LANE="$tag" BASE_URL=http://127.0.0.1:30001 python3 "$TF" $HOME/glmf/w3/greedy-w2-a.json "$OUT/tf-$tag.json" > "$OUT/tf-gen.txt" 2>&1 || log "TF_GEN_FAIL $tag"
  test -f "$OUT/tf-$tag.json" && python3 "$TF" --compare "$CTRL_TF" "$OUT/tf-$tag.json" > "$OUT/tf-compare.txt" 2>&1
  EFFORT=low python3 tool_harness.py > "$OUT/tool-1.txt" 2>&1
  unset MODEL
  log "ARM $tag: $(grep -ho 'GREEDY_EQUIV identical=[0-9/]*' $OUT/greedy-vs-daily.txt 2>/dev/null) | TF $(grep -oE '"mean": [0-9.e-]+|"p99": [0-9.e-]+' $OUT/tf-compare.txt 2>/dev/null | tr '\n' ' ') | tools $(grep -ho 'tool_ok=[0-9/]*' $OUT/tool-1.txt) | knee $(grep -hoE 'conc=(8|16): +[0-9.]+ agg' $OUT/knee.txt | tr '\n' ';') | probe $(grep -h SUMMARY $OUT/probe-rep*.txt | sed 's/.*SUMMARY //' | tr '\n' ' ' | cut -c1-500)"; }
log "===== CARD M (KDA snapshot cap: ctl / cap2 / cap4) START ====="
other=$(docker ps --format '{{.Names}}' | grep -vE "^glmf-" || true); [[ -n $other ]] && { log "REFUSE foreign container: $other"; exit 4; }
docker image inspect "$IMG" >/dev/null 2>&1 || { log "NO IMAGE $IMG"; exit 3; }
for f in kda_snapshot_probe.py greedy_equiv.py tool_harness.py flash_bench.py "$TF" "$CTRL_TF" "$REF_GREEDY"; do test -f "$f" || { log "MISSING $f"; exit 3; }; done
CAP=-1 arm ctl
CAP=2  arm cap2 --mamba-max-states-per-path 2
CAP=4  arm cap4 --mamba-max-states-per-path 4
stop_glmf
log "restart daily: docker start $DAILY"; docker start "$DAILY" >/dev/null && wait_ready "$DAILY" && log "daily back on :30001" || log "DAILY RESTART FAILED — page James"
{ echo; echo "## $(date '+%F %H:%M CDT') — Card M: KDA snapshot cap probe (HelixML mechanism) on the daily image · ctl / cap2 / cap4"
  echo "worker: sonnet55 under Milo (hermes milo) · James: go 2026-10-01 ~09:45"
  grep -h "ARM \|BOOT FAIL\|PROBE_FAIL\|daily back\|DAILY RESTART" "$D/runner.log" | cut -c1-900
  echo "receipts $D · glmf-M-* stopped-and-kept · daily restarted (same config, not a promotion)"; } >> $HOME/dsv41/results/ledger.md
log "===== CARD M END ====="
