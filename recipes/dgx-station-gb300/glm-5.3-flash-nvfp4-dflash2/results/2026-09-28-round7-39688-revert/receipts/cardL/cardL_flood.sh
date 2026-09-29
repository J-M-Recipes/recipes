#!/usr/bin/env bash
# Card L (2026-09-28 overnight): MiMo-style flood fixture on GLM-5.3-Flash daily lane (v0.5.20 release image, DFlash2 on and off).
# Waits for Card K to end, then boots the daily config, runs flood_fixture.py (same script sha as the MiMo recipe), DFlash on then off.
# Publishes flood axis + within/cross-turn repetition. Stop-and-keep. Nothing restored at the end.
set -uo pipefail
cd $HOME/glmf
D=$HOME/glmf/cardL-2026-09-28; mkdir -p "$D"; exec >> "$D/runner.log" 2>&1
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
CKPT=$MODELS/GLM-5.3-Flash-NVFP4-nvidia-09b04e5e; DRAFT=$MODELS/GLM-5.3-Flash-DFlash2
IMG=glmf-sglang:0.5.20-tf5.16.1
BASE_KNOBS_DFLASH="--max-mamba-cache-size 48 --speculative-dflash-block-size 7"
BASE_KNOBS_AR="--max-mamba-cache-size 48"
FIX=$HOME/mimo26/flood_fixture.py
stop_glmf(){ for c in $(docker ps --format '{{.Names}}' | grep '^glmf-' || true); do log "stop+keep $c"; docker stop "$c" >/dev/null; done; sleep 3; }
wait_ready(){ local i; sleep 5; for i in $(seq 1 300); do curl -sf --max-time 2 http://127.0.0.1:30001/v1/models >/dev/null 2>&1 && { log "ready $1 i=$i"; return 0; }
    docker ps --format '{{.Names}}' | grep -q "^$1$" || { log "CONTAINER_EXITED $1"; return 5; }; sleep 5; done; log "READY_TIMEOUT $1"; return 5; }

log "===== CARD L (GLM Flash flood fixture) waiting for Card K ====="
t0=$SECONDS
while pgrep -f cardK_revert39688.sh >/dev/null; do sleep 60; [[ $((SECONDS-t0)) -gt 21600 ]] && { log "K still running after 6h, giving up"; exit 6; }; done
log "Card K ended; starting L"
other=$(docker ps --format '{{.Names}}' | grep -vE "^glmf-" || true); [[ -n $other ]] && { log "REFUSE foreign container: $other"; exit 4; }
sha256sum "$FIX" > "$D/fixture-sha.txt"

run_arm(){ local tag=$1 spec=$2 knobs=$3; mkdir -p "$D/$tag"; stop_glmf; unset MODEL
  TAG="L-$tag" MODEL=$CKPT DRAFT=$DRAFT IMAGE="$IMG" SPEC=$spec KV=fp8_e4m3 MEM=0.85 CTX=1048576 MAXBS=16 EXTRA="$knobs" DOCKER_ENV="-e TORCHINDUCTOR_COMPILE_THREADS=1" bash launch-glmf.sh > "$D/$tag/launch.txt" 2>&1
  wait_ready "glmf-L-$tag" || { log "BOOT FAIL $tag"; return 1; }
  export MODEL=glm-5.3-flash BASE_URL=http://127.0.0.1:30001/v1 API_KEY=none MODEL_NAME=glm-5.3-flash
  python3 -c 'import flash_bench as fb; fb.knee(concs=(1,), reps=0)' > "$D/$tag/warm.txt" 2>&1
  BASE_URL=http://127.0.0.1:30001/v1 MODEL=glm-5.3-flash API_KEY=none CONC=4 MAX_TURNS=10 MAXTOK=4096 TEMP=0 THINKING=1 HISTORY=both MAX_CALLS=120 \
    python3 "$FIX" run "glmf-$tag" "$D/$tag" > "$D/$tag/fixture.log" 2>&1 || log "FIXTURE_FAIL $tag: $(tail -2 $D/$tag/fixture.log | tr '\n' ' ')"
  unset MODEL
  log "ARM $tag: $(python3 -c "
import json,glob,sys
fs=glob.glob('$D/$tag/flood-*-summary.json')
if not fs: print('NO SUMMARY'); sys.exit()
s=json.load(open(fs[0])); print({k:s[k] for k in s if k not in ('tag','base','model')})" 2>&1 | cut -c1-600)"; }

run_arm dflash dflash "$BASE_KNOBS_DFLASH"
run_arm ar     none   "$BASE_KNOBS_AR"
stop_glmf
{ echo; echo "## $(date '+%F %H:%M CDT') — Card L: MiMo flood fixture on GLM-5.3-Flash (v0.5.20 daily; DFlash on/off)"
  echo "worker: Milo (hermes milo) · unattended overnight · fixture sha $(cut -c1-12 $D/fixture-sha.txt)"
  grep -h "ARM \|BOOT FAIL\|FIXTURE_FAIL" "$D/runner.log" | cut -c1-700
  echo "reference: MiMo-V2.6-Pro RL 25/36 history-primed flooded, MOPD 0/36 · receipts $D · glmf-L-* stopped-and-kept · nothing restored"; } >> $HOME/dsv41/results/ledger.md
log "===== CARD L END ====="
