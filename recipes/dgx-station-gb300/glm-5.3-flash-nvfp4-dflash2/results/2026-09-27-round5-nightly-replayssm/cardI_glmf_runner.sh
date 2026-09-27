#!/usr/bin/env bash
# Card I (2026-09-27) — GLM-5.3-Flash on :30001. Milo for James ("go" 2026-09-27 ~13:20 CDT).
#  I0  profile: current release image (glmf-sglang:0.5.20-tf5.16.1), daily config, torch profiler on 3 C1 windows.
#  I1  CTL:  same image, daily config (same-window control; also = I0 image without profiler)
#  I2  NIGHT: lmsysorg/sglang:nightly-dev-cu13-20260927-425a1f8f + transformers 5.16.1 derive (carries #39200 mHC fuse, #40517 ReplaySSM-GLM, #39816 cutedsl AR, #41194 NextN plan)
#  I3  NIGHT+RSSM: I2 + --enable-linear-replayssm-spec  (W5 reopened by #40517). If DFlash rejects it, retry SPEC=none for the batch lane.
#  I4  CTL2: same as I1 (bracket)
# Gates per arm (from w478_run.sh instrument): c1_methods low/max, accept_probe, tool_harness x2, C8 knee, greedy 20 vs w2-a, TF vs w5-ctrl (per-image gate only!).
# Box: :30006 v21 (DSV4.1) MUST be stopped for GLM Flash to fit; stop-and-keep, restore at end. Never :30003. No rm, no pull beyond the one nightly.
set -uo pipefail
cd ~/glmf
D=~/glmf/cardI-2026-09-27; mkdir -p "$D"; exec >> "$D/runner.log" 2>&1
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
LEDGER=~/dsv41/results/ledger.md
DSV=dsv41-vllm-v21-fused-split-BOUND-REF
REL=glmf-sglang:0.5.20-tf5.16.1
NIGHT_BASE=lmsysorg/sglang:nightly-dev-cu13-20260927-425a1f8f
NIGHT=glmf-sglang:nightly-20260927-425a1f8f-tf5.16.1
MODEL=/models/GLM-5.3-Flash-NVFP4-nvidia-09b04e5e; DRAFT=/models/GLM-5.3-Flash-DFlash2
BASE_KNOBS="--max-mamba-cache-size 48 --speculative-dflash-block-size 7"
ENV_BASE="-e TORCHINDUCTOR_COMPILE_THREADS=1"
TF=~/window-20260913/scripts/tf_noninferiority.py; CTRL_TF=~/glmf/w5-ctrl/tf-w5-ctrl.json
export BASE_URL=http://127.0.0.1:30001/v1 MODEL_NAME=glm-5.3-flash
for f in launch-glmf.sh flash_bench.py c1_methods.py accept_probe.py tool_harness.py greedy_equiv.py w3/greedy-w2-a.json "$TF" "$CTRL_TF"; do test -f "$f" || { log "missing $f"; exit 3; }; done

stop_glmf(){ for c in $(docker ps --format '{{.Names}}' | grep '^glmf-' || true); do log "stop+keep $c"; docker stop "$c" >/dev/null; done; sleep 3; }
wait_ready(){ local i; for i in $(seq 1 240); do
    curl -sf --max-time 2 http://127.0.0.1:30001/v1/models >/dev/null 2>&1 && { log "ready $1 i=$i"; return 0; }
    docker ps --format '{{.Names}}' | grep -q "^$1$" || { log "CONTAINER_EXITED $1"; docker logs "$1" 2>&1 | tr '\r' '\n' | grep -nE "Error|error:|Traceback|raise|Exception" | tail -15 | cut -c1-300; return 5; }
    sleep 5; done; log "READY_TIMEOUT $1"; return 5; }
boot(){ # tag image spec extra
  local TAG=$1 IMG=$2 SPEC=$3 EXTRA=$4 OUT=$D/$1; mkdir -p "$OUT"
  log "BOOT $TAG image=$IMG spec=$SPEC extra=[$EXTRA]"
  stop_glmf
  TAG="I-$TAG" MODEL=$MODEL DRAFT=$DRAFT IMAGE="$IMG" SPEC="$SPEC" KV=fp8_e4m3 MEM=0.85 CTX=1048576 MAXBS=16 EXTRA="$EXTRA" DOCKER_ENV="$ENV_BASE" bash launch-glmf.sh > "$OUT/launch.txt" 2>&1
  local CONT=glmf-I-$TAG
  wait_ready "$CONT" || { docker logs "$CONT" > "$OUT/boot-fail.log" 2>&1; return 1; }
  docker logs "$CONT" 2>&1 | tr '\r' '\n' | grep -E "Load weight|DFLASH|draft|KDA|KV Cache is allocated|Mamba|max_running_requests|replay|Replay|mHC|mhc|fuse|Warning|WARNING|version" | grep -v server_args | head -80 > "$OUT/boot-excerpt.txt" || true
  docker exec "$CONT" python3 -c "import sglang,transformers,torch;print('sglang',sglang.__version__,'transformers',transformers.__version__,'torch',torch.__version__)" >> "$OUT/boot-excerpt.txt" 2>&1 || true
  return 0
}
instrument(){ local TAG=$1 OUT=$D/$1
  python3 -c 'import flash_bench as fb; fb.knee(concs=(1,2,4,8), reps=0)' > "$OUT/warm.txt" 2>&1
  TAG="$TAG-low" EFFORT=low python3 c1_methods.py > "$OUT/c1-low.txt" 2>&1
  TAG="$TAG-low" EFFORT=low python3 accept_probe.py > "$OUT/accept-low.txt" 2>&1
  TAG="$TAG-max" EFFORT=max python3 c1_methods.py > "$OUT/c1-max.txt" 2>&1
  EFFORT=low python3 tool_harness.py > "$OUT/tool-1.txt" 2>&1; EFFORT=low python3 tool_harness.py > "$OUT/tool-2.txt" 2>&1
  python3 -c 'import flash_bench as fb; fb.knee(concs=(8,), reps=0)' > "$OUT/knee-c8-warm.txt" 2>&1
  python3 -c 'import flash_bench as fb; fb.knee(concs=(8,), reps=3)' > "$OUT/knee-c8.txt" 2>&1
  if grep -qE "spread ([1-9][0-9]|[0-9]{3,})%" "$OUT/knee-c8.txt"; then python3 -c 'import flash_bench as fb; fb.knee(concs=(8,), reps=0)' >/dev/null 2>&1; python3 -c 'import flash_bench as fb; fb.knee(concs=(8,), reps=3)' > "$OUT/knee-c8-rerun.txt" 2>&1; fi
  python3 greedy_equiv.py "$OUT/greedy-$TAG.json" > "$OUT/greedy-gen.txt" 2>&1 || log "GREEDY_GEN_FAIL $TAG"
  test -f "$OUT/greedy-$TAG.json" && python3 greedy_equiv.py --compare ~/glmf/w3/greedy-w2-a.json "$OUT/greedy-$TAG.json" > "$OUT/greedy-vs-w2.txt" 2>&1
  LANE="$TAG" BASE_URL=http://127.0.0.1:30001 python3 "$TF" ~/glmf/w3/greedy-w2-a.json "$OUT/tf-$TAG.json" > "$OUT/tf-gen.txt" 2>&1
  test -f "$OUT/tf-$TAG.json" && python3 "$TF" --compare "$CTRL_TF" "$OUT/tf-$TAG.json" > "$OUT/tf-compare.txt" 2>&1
  log "ARM $TAG: $(grep -h 'recipe-method' "$OUT/c1-low.txt" | cut -c1-90) | $(grep -h 'code 400' "$OUT/c1-low.txt" | cut -c1-70) | C8 $(grep -h 'conc= 8' "$OUT/knee-c8.txt" | tail -1 | cut -c1-80) | tools $(grep -ho 'tool_ok=[0-9/]*' "$OUT/tool-1.txt" "$OUT/tool-2.txt" | tr '\n' ' ') | $(grep -ho 'GREEDY_EQUIV identical=[0-9/]*' "$OUT/greedy-vs-w2.txt") | TF $(tail -1 "$OUT/tf-compare.txt" | cut -c1-120)"
}

log "===== CARD I START ====="
other=$(docker ps --format '{{.Names}}' | grep -vE "^glmf-|^$DSV$" || true); [[ -n $other ]] && { log "REFUSE foreign container: $other"; exit 4; }
if docker ps --format '{{.Names}}' | grep -q "^$DSV$"; then log "stop+keep $DSV (GLM Flash needs the GPU)"; docker stop "$DSV" >/dev/null; fi
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

# ---- build nightly derive (CPU-only, may take a few min for the pull)
if ! docker image inspect "$NIGHT" >/dev/null 2>&1; then
  log "pull+derive $NIGHT_BASE -> $NIGHT"
  docker pull "$NIGHT_BASE" > "$D/pull.log" 2>&1 || log "PULL FAILED (see pull.log)"
  printf 'FROM %s\nRUN pip install --no-cache-dir "transformers==5.16.1" "tokenizers==0.23.2" && python3 -c "import transformers, tokenizers; assert transformers.__version__ == \x275.16.1\x27; assert tokenizers.__version__ == \x270.23.2\x27"\n' "$NIGHT_BASE" > "$D/Dockerfile.nightly-tf5161"
  docker build -t "$NIGHT" -f "$D/Dockerfile.nightly-tf5161" "$D" > "$D/build.log" 2>&1 || log "BUILD FAILED (see build.log)"
fi
docker image inspect "$NIGHT" >/dev/null 2>&1 && log "nightly image ready: $(docker image inspect $NIGHT --format '{{.Id}}' | cut -c8-19)" || log "nightly image MISSING — I2/I3 will be skipped"

# ---- I0 profile on release image
if boot prof "$REL" dflash "$BASE_KNOBS"; then
  OUT=$D/prof; PROF=$D/prof/traces; mkdir -p "$PROF"
  python3 -c 'import flash_bench as fb; fb.knee(concs=(1,), reps=0)' > "$OUT/warm.txt" 2>&1
  python3 - "$PROF" <<'PY'
import json, sys, urllib.request, time, os
B="http://127.0.0.1:30001"; PROF=sys.argv[1]
def post(path, body=None):
    r=urllib.request.Request(B+path, data=json.dumps(body or {}).encode(), headers={"Content-Type":"application/json"}); return urllib.request.urlopen(r, timeout=600).read()
def gen(prompt, n, effort="low"):
    body={"model":"glm-5.3-flash","messages":[{"role":"user","content":prompt}],"max_tokens":n,"temperature":0,"chat_template_kwargs":{"reasoning_effort":effort}}
    return post("/v1/chat/completions", body)
for name,prompt in (("prose","Write a 350-word essay on why engineers should keep failure ledgers. No headings, no lists."),("code","Write a Python module with three functions: parse an nginx log line, aggregate status codes, and print a table. Include docstrings."),("shell","Give me the exact bash commands to: find files over 1GB under /var, list docker containers with their images, and show the ten largest directories in /home. Commands only.")):
    gen(prompt, 64)  # warm
    try:
        post("/start_profile", {"output_dir":"/tmp/prof-"+name, "num_steps": 60, "activities":["CPU","GPU"], "with_stack": False, "record_shapes": False})
    except Exception as e:
        print("start_profile err", e); continue
    gen(prompt, 200)
    try: post("/stop_profile")
    except Exception as e: print("stop_profile err", e)
    time.sleep(10); print("captured", name, flush=True)
PY
  for n in prose code shell; do docker cp "glmf-I-prof:/tmp/prof-$n" "$PROF/trace-$n" 2>/dev/null || log "no trace dir for $n"; done
  ls -R "$PROF" | head -20 >> "$OUT/meta.txt"
  log "I0 profile captured: $(find $PROF -name '*.json*' | wc -l) trace files"
else log "I0 BOOT FAIL"; fi

# ---- I1 CTL (release)
boot ctl1 "$REL" dflash "$BASE_KNOBS" && instrument ctl1 || log "I1 BOOT FAIL"
# ---- I2 nightly
if docker image inspect "$NIGHT" >/dev/null 2>&1; then
  boot night "$NIGHT" dflash "$BASE_KNOBS" && instrument night || log "I2 BOOT FAIL"
  # ---- I3 nightly + replayssm
  if boot night-rssm "$NIGHT" dflash "$BASE_KNOBS --enable-linear-replayssm-spec"; then
    grep -E "KV Cache is allocated|Mamba|replay" "$D/night-rssm/boot-excerpt.txt" >> "$D/night-rssm/meta.txt"; instrument night-rssm
  else
    log "I3 dflash+replayssm rejected at boot — trying SPEC=none (batch lane)"
    docker logs glmf-I-night-rssm 2>&1 | tr '\r' '\n' | grep -iE "replayssm|requires|Error" | tail -5 | cut -c1-300
    if boot night-rssm-ar "$NIGHT" none "--max-mamba-cache-size 640 --enable-linear-replayssm-spec"; then
      grep -E "KV Cache is allocated|Mamba|replay" "$D/night-rssm-ar/boot-excerpt.txt" >> "$D/night-rssm-ar/meta.txt"
      python3 -c 'import flash_bench as fb; fb.knee(concs=(1,8,48), reps=0)' > "$D/night-rssm-ar/warm.txt" 2>&1
      python3 -c 'import flash_bench as fb; fb.knee(concs=(1,8,48), reps=2)' > "$D/night-rssm-ar/knee.txt" 2>&1
      log "I3-AR: $(grep -h 'conc=' $D/night-rssm-ar/knee.txt | tr '\n' ';' | cut -c1-300)"
    else log "I3-AR BOOT FAIL too"; fi
  fi
else log "I2/I3 skipped: nightly image missing"; fi
# ---- I4 CTL2 (release)
boot ctl2 "$REL" dflash "$BASE_KNOBS" && instrument ctl2 || log "I4 BOOT FAIL"

stop_glmf
{ echo; echo "## $(date '+%F %H:%M CDT') — Card I GLM Flash: profile + nightly 425a1f8f bracket + ReplaySSM (W5 reopened)"; echo "worker: Milo (hermes milo) · James: go 2026-09-27"; grep -h "^.*ARM \|I0 profile\|I3\|BOOT FAIL\|skipped" "$D/runner.log" | cut -c1-400; echo "receipts $D · glmf-I-* stopped-and-kept · v21 DSV4.1 restored on :30006"; } >> "$LEDGER"
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
docker start "$DSV" >/dev/null; t0=$SECONDS
until curl -sf -m 3 http://127.0.0.1:30006/v1/models >/dev/null 2>&1; do sleep 20; [[ $((SECONDS-t0)) -gt 1800 ]] && { log "v21 RESTORE TIMEOUT"; break; }; done
log "v21 rebound after $((SECONDS-t0))s"
log "===== CARD I END ====="
