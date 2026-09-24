#!/usr/bin/env bash
# CARD W3a.2 (2026-09-23) — GLM-5.3-Flash NVFP4 + DFlash2: FIXED VERIFY WIDTH (drafter block 7, target verifies K).
# James 2026-09-23: "try port" (W3) · "take your time we have all week" · "go" (after W3a found the glm47 grammar gap).
# W3a.2 vs W3a: the patch now also narrows GRAMMAR steps (glm47 attaches full_assistant_ebnf to every chat request, so
# W3a fell back to full width on 100% of traffic) — grammar bitmask built over the same K-wide chain. Unit: T5.
# One axis: SGLANG_DFLASH_VERIFY_WIDTH=K via two bind-mounted patched files, sha256-pinned in-container.
# Everything else = recipe pin (v0.5.20-cu130, NVFP4 09b04e5e, draft 7d74cdd8, block 7, 48 mamba slots, mem 0.85,
# 1M ctx, fp8 KV). Ladder K=5,4,3 then an ADJACENT stock control (glmf-e0 restarted unchanged).
# HARD gates per boot (fail => no measurement, boot reported INVALID): patch sha in container; runtime-built line;
# VW gate: after warm + greedy probes, verify-width steps>0 and fallback share <5%. Soft/reported gates: tools 10/10,
# greedy self-repeat identical, greedy vs stock-e0 + ref REPORTED as-is, TF vs e0 (prefill path: must be max 0.0;
# harness NONINFERIOR needs >=3000 tokens and this set has 2857, so it is reported, not gated).
# Win bar (decided before data): beats adjacent stock control on prose AND recipe essay C1 with no code regression
# (> -1%), all gates green. Lane :30001, containers glmf-w3a2-*. Stop-and-keep. No other lane restored. STOP honoured.
# Written by Milo (Hermes milo profile) for James Meadlock.
set -uo pipefail
export PYTHONUNBUFFERED=1
R=$HOME/glmf/w3a2-2026-09-23; cd "$R"
G=$HOME/glmf; TF=$HOME/window-20260913/scripts/tf_noninferiority.py
E0=$HOME/glmf/cardE-2026-09-23/e0; REF=$HOME/glmf/w3/greedy-w2-a.json
IMG=lmsysorg/sglang:v0.5.20-cu130
MODELDIR=/models/GLM-5.3-Flash-NVFP4-nvidia-09b04e5e; DRAFTDIR=/models/GLM-5.3-Flash-DFlash2
P=/sgl-workspace/sglang/python/sglang/srt
PW=$R/patch/speculative/dflash_worker_v2.py; PK=$R/patch/layers/attention/dsa/kpool_plan.py
export BASE_URL=http://127.0.0.1:30001/v1 MODEL=glm-5.3-flash API_KEY=""
STOPF=$R/STOP; LOG=$R/cardW3a2.log
exec >> "$LOG" 2>&1
exec 9>"$R/.lock"; flock -n 9 || { echo "LOCKED"; exit 9; }
log(){ printf '%s %s\n' "$(date '+%F %T %Z')" "$*"; }
stop_glmf(){ local n; for n in $(docker ps --format '{{.Names}}' | grep -E '^glmf-'); do log "stop-and-keep $n"; docker stop "$n" >/dev/null; done; sleep 3; }
finish(){ stop_glmf; log "finish: all glmf-* stopped-and-kept; no other lane restored (by instruction)"; log "===== CARD W3a.2 END ====="; echo W3A2_DETACHED_END; }
trap finish EXIT
stop_check(){ [[ -f $STOPF ]] && { log "STOP seen"; exit 0; }; return 0; }

log "===== CARD W3a.2 BEGIN ====="
busy=$(docker ps --format '{{.Names}}' || true)
[[ -n $busy ]] && { log "REFUSE: containers running: $busy"; trap - EXIT; exit 4; }
grep -q "W3A_UNIT ALL_PASS" unit.txt && grep -q "W3A2_GRAMMAR ALL_PASS" unit.txt || { log "REFUSE: unit tests not ALL_PASS"; exit 3; }
sha256sum "$PW" "$PK" > "$R/patch-SHA256SUMS"; while read l; do log "PATCH $l"; done < "$R/patch-SHA256SUMS"
EW=$(sha256sum "$PW" | cut -d' ' -f1); EK=$(sha256sum "$PK" | cut -d' ' -f1)

warm(){ ( cd $G && python3 -c 'import flash_bench as fb; fb.knee(concs=(1,2,4,8), reps=0)' ) > "$1/warm.txt" 2>&1; }

boot(){ # TAG ENVEXTRA  (fresh container glmf-TAG)
  local TAG=$1 ENVX=$2 OUT=$R/$1; mkdir -p "$OUT"; stop_check; stop_glmf
  sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
  log "BOOT $TAG env=[$ENVX]"
  local t0=$(date +%s)
  TAG="$TAG" MODEL="$MODELDIR" DRAFT="$DRAFTDIR" IMAGE="$IMG" SPEC=dflash KV=fp8_e4m3 MEM=0.85 CTX=1048576 MAXBS=16 \
    EXTRA="--max-mamba-cache-size 48 --speculative-dflash-block-size 7" LOGDIR="$R" \
    DOCKER_ENV="-e TORCHINDUCTOR_COMPILE_THREADS=1 $ENVX" bash $G/launch-glmf.sh > "$OUT/launch.txt" 2>&1 \
    || { log "LAUNCH_FAIL $TAG"; cat "$OUT/launch.txt"; return 1; }
  wait_ready "$TAG" "$t0" "$OUT"; }

wait_ready(){ local TAG=$1 t0=$2 OUT=$3
  for i in $(seq 1 360); do
    curl -sf --max-time 2 http://127.0.0.1:30001/v1/models >/dev/null 2>&1 && break
    docker ps --format '{{.Names}}' | grep -q "^glmf-$TAG$" || { log "EXITED $TAG"; docker logs --tail 80 "glmf-$TAG" 2>&1 | tr '\r' '\n' | grep -iE "error|Traceback|assert|raise|mismatch" | tail -12; return 5; }
    stop_check; sleep 5; done
  curl -sf --max-time 2 http://127.0.0.1:30001/v1/models >/dev/null 2>&1 || { log "READY_TIMEOUT $TAG"; return 5; }
  log "READY $TAG after $(( $(date +%s)-t0 ))s"
  docker logs "glmf-$TAG" 2>&1 | tr '\r' '\n' | grep -E "DFLASH|verify-width|Capture target verify|Capture draft verify|max_total_num_tokens|KPool" | grep -v server_args | cut -c1-300 > "$OUT/boot-excerpt.txt"
  docker inspect "glmf-$TAG" --format '{{.Image}} {{json .Config.Env}} {{json .HostConfig.Binds}}' | sed 's/API_KEY=[^",]*/API_KEY=[REDACTED]/g' > "$OUT/inspect.txt"
  return 0; }

greedy_probe(){ python3 - <<'PY' 2>&1 | tail -1
import json,urllib.request
p={"model":"glm-5.3-flash","temperature":0,"max_tokens":512,"chat_template_kwargs":{"reasoning_effort":"low"},
   "messages":[{"role":"user","content":"Write a detailed 400-word explanation of how lighthouse lenses work."}]}
r=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:30001/v1/chat/completions",data=json.dumps(p).encode(),headers={"Content-Type":"application/json"}),timeout=600))
print("probe_tokens", r["usage"]["completion_tokens"])
PY
}

vw_gate(){ # TAG OUT -> 0 if steps>0 and fallback share <5% on the latest stats line with total>1
  local T=$1 O=$2 line s f
  for i in 1 2 3 4 5 6 7 8; do
    line=$(docker logs "glmf-$T" 2>&1 | tr '\r' '\n' | grep "verify-width stats" | tail -1)
    s=$(sed -n 's/.*steps=\([0-9]*\) fallback_full_width=\([0-9]*\).*/\1/p' <<< "$line"); f=$(sed -n 's/.*steps=\([0-9]*\) fallback_full_width=\([0-9]*\).*/\2/p' <<< "$line")
    [[ -n $s && $(( s + f )) -gt 1 ]] && break
    log "VWGATE $T waiting for stats (probe $i: $(greedy_probe))"
  done
  echo "$line" > "$O/vw-gate.txt"
  [[ -n ${s:-} && $s -gt 0 && $(( f * 20 )) -lt $(( s + f )) ]] && { log "VWGATE $T ok: $(cut -c1-260 <<< "$line")"; return 0; }
  log "VWGATE $T FAIL: ${line:-no stats line}"; return 1; }

measure(){ # TAG OUT  (instruments identical to Card E e0)
  local T=$1 O=$2
  ( cd $G && TAG=$T EFFORT=low python3 c1_methods.py ) > "$O/c1-low.txt" 2>&1; log "C1 $T $(tr '\n' '|' < $O/c1-low.txt | cut -c1-700)"
  ( cd $G && TAG=$T EFFORT=low CONT=glmf-$T python3 accept_probe.py ) > "$O/accept-low.txt" 2>&1; log "ACCEPT $T $(tail -1 $O/accept-low.txt)"
  ( cd $G && EFFORT=low python3 tool_harness.py ) > "$O/tool-1.txt" 2>&1; log "TOOLS $T $(grep SUMMARY $O/tool-1.txt)"
  ( cd $G && python3 greedy_equiv.py "$O/greedy-a.json" > /dev/null 2>&1; python3 greedy_equiv.py "$O/greedy-b.json" > /dev/null 2>&1
    python3 greedy_equiv.py --compare "$O/greedy-a.json" "$O/greedy-b.json" > "$O/greedy-self.txt" 2>&1
    python3 greedy_equiv.py --compare "$E0/greedy-e0.json" "$O/greedy-a.json" > "$O/greedy-vs-e0.txt" 2>&1
    python3 greedy_equiv.py --compare "$REF" "$O/greedy-a.json" > "$O/greedy-vs-ref.txt" 2>&1 )
  log "GREEDY $T self $(grep -h GREEDY_EQUIV $O/greedy-self.txt) · vs-e0 $(grep -h GREEDY_EQUIV $O/greedy-vs-e0.txt) · vs-ref $(grep -h GREEDY_EQUIV $O/greedy-vs-ref.txt)"
  LANE=$T BASE_URL=http://127.0.0.1:30001 python3 "$TF" "$REF" "$O/tf-$T.json" > "$O/tf-log.txt" 2>&1
  python3 "$TF" --compare "$E0/tf-e0.json" "$O/tf-$T.json" > "$O/tf-vs-e0.json" 2>&1
  log "TF $T vs e0: $(python3 -c "import json;d=json.load(open('$O/tf-vs-e0.json'));print('tokens',d['tokens'],'max',d['max'],'IDENTICAL' if d['max']==0.0 else 'DIFFERS','harnessNONINF',d['NONINFERIOR'])" 2>&1 | tail -1)"
  docker logs "glmf-$T" 2>&1 | tr '\r' '\n' | grep -E "verify-width (stats|fallback)" | tail -6 > "$O/vw-stats.txt"
  log "VWSTATS $T $(tail -1 $O/vw-stats.txt | cut -c1-260)"
}

for K in 5 4 3; do
  T=w3a2-k$K; O=$R/$T
  ENVX="-e SGLANG_DFLASH_VERIFY_WIDTH=$K -v $PW:$P/speculative/dflash_worker_v2.py:ro -v $PK:$P/layers/attention/dsa/kpool_plan.py:ro"
  if boot "$T" "$ENVX"; then
    gw=$(docker exec glmf-$T sha256sum $P/speculative/dflash_worker_v2.py | cut -d' ' -f1)
    gk=$(docker exec glmf-$T sha256sum $P/layers/attention/dsa/kpool_plan.py | cut -d' ' -f1)
    [[ $gw == "$EW" && $gk == "$EK" ]] && log "PATCH_GATE $T ok (in-container sha match)" || { log "PATCH_GATE $T FAIL w=$gw k=$gk -> INVALID"; continue; }
    grep -q "verify-width runtime built" "$O/boot-excerpt.txt" && log "RUNTIME $T $(grep -m1 'verify-width runtime built' $O/boot-excerpt.txt | cut -c1-220)" || { log "RUNTIME_GATE $T FAIL -> INVALID"; continue; }
    warm "$O"
    vw_gate "$T" "$O" || { log "RESULT $T INVALID (verify width not exercised)"; continue; }
    measure "$T" "$O"
  else log "BOOT_FAIL $T"; fi
  stop_check
done

# adjacent stock control: the Card E e0 container itself, unchanged (docker start)
O=$R/ctrl-e0; mkdir -p "$O"; stop_check; stop_glmf; sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
t0=$(date +%s); docker start glmf-e0 >/dev/null; log "START ctrl glmf-e0 (stock, unchanged)"
if wait_ready e0 "$t0" "$O"; then
  warm "$O"
  ( cd $G && TAG=ctrl-e0 EFFORT=low python3 c1_methods.py ) > "$O/c1-low.txt" 2>&1; log "C1 ctrl-e0 $(tr '\n' '|' < $O/c1-low.txt | cut -c1-700)"
  ( cd $G && TAG=ctrl-e0 EFFORT=low CONT=glmf-e0 python3 accept_probe.py ) > "$O/accept-low.txt" 2>&1; log "ACCEPT ctrl-e0 $(tail -1 $O/accept-low.txt)"
  ( cd $G && python3 greedy_equiv.py "$O/greedy-a.json" > /dev/null 2>&1; python3 greedy_equiv.py --compare "$E0/greedy-e0.json" "$O/greedy-a.json" > "$O/greedy-vs-e0.txt" 2>&1 )
  log "GREEDY ctrl-e0 vs cardE-e0 $(grep -h GREEDY_EQUIV $O/greedy-vs-e0.txt)"
else log "CTRL_BOOT_FAIL"; fi

# verdict table (medians) — computed, not eyeballed
python3 - "$R" > "$R/verdict.txt" 2>&1 <<'PY'
import re,sys,os
R=sys.argv[1]
def med(path):
    out={}
    if not os.path.exists(path): return out
    for l in open(path):
        m=re.search(r"(\S+) \d+ effort=low: runs \[.*?\] median ([0-9.]+)",l)
        if m: out[m.group(1)]=float(m.group(2))
    return out
ctrl=med(f"{R}/ctrl-e0/c1-low.txt"); print("ctrl-e0", ctrl)
for K in (5,4,3):
    d=med(f"{R}/w3a2-k{K}/c1-low.txt")
    if not d or not ctrl: print(f"k{K} NO_DATA"); continue
    rel={k:round((d[k]/ctrl[k]-1)*100,2) for k in d if k in ctrl}
    win = rel.get("prose",-1)>0 and rel.get("history-essay",-1)>0 and rel.get("code",-9)>-1.0
    print(f"k{K}", d, "vs ctrl %", rel, "WIN_BAR" if win else "no-win")
PY
while read l; do log "VERDICT $l"; done < "$R/verdict.txt"
