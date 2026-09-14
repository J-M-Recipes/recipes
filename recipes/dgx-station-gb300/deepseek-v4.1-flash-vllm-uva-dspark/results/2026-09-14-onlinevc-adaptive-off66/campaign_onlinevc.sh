#!/usr/bin/env bash
# ONLINEVC — online-updated DSpark verify-cost curve on the offloaded DSV4.1 lane (vllm#38256 follow-up).
# Claim under test: the adaptive-verification controller's verify_ms[num_tokens] table is profiled on dummy
# batches at boot and never sees expert streaming; replacing it with an EMA of real verify-step forward
# times lets the controller drop drafts it currently keeps → prose ≥ k=1 (106) without losing shell.
# Same-window pair: (1) control = restart the stopped ADAPT-off66-k5-EXP container (adaptive, boot curve),
# knee + fixture; (2) candidate = identical launch + 2 patched files bind-mounted + VLLM_DSPARK_ONLINE_VERIFY_CURVE=1,
# knee + fixture. One axis: the verify curve source. Never touches :30003.
set -euo pipefail
cd ~/dsv41
CTRL=dsv41-vllm-ADAPT-off66-k5-EXP
MODEL=/models/DeepSeek-V4.1-Flash-df42c109f1defefcbfcedbe7d905718a12266e40
PATCH=~/dsv41/patches/onlinevc
SP=/usr/local/lib/python3.12/dist-packages/vllm
LEDGER=~/dsv41/results/ledger.md
CAMPLOG=~/dsv41/results/campaign-onlinevc-$(date +%F).log
STOPF=~/dsv41/STOP-CAMPAIGN
TAG="${TAG:-ONLINEVC-off66-k5}"
PAIR="${PAIR:-1}"
OFFGB="${OFFGB:-66}"; UTIL="${UTIL:-0.97}"
SKIP_CONTROL="${SKIP_CONTROL:-0}"
mkdir -p ~/dsv41/results
exec >> "$CAMPLOG" 2>&1

log() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*"; }
die() { log "FATAL: $*"; finish || true; exit 1; }
trap 'log "ERR line=$LINENO status=$?"' ERR
check_stop() { if [[ -f "$STOPF" ]]; then log "STOP-CAMPAIGN"; finish; exit 0; fi; return 0; }

check_safety() {
  other=$(docker ps --format '{{.Names}}' | grep -vE '^dsv41-vllm-' || true)
  if [[ -n "$other" ]]; then die "other container(s) running: $other"; fi
  if dmesg -T 2>/dev/null | tail -n 40 | grep -qiE 'Out of memory|oom-kill'; then die "oom in dmesg"; fi
  for f in model_runner.py adaptive_verification.py; do [[ -s "$PATCH/$f" ]] || die "missing $PATCH/$f"; done
  grep -q 'class OnlineVerifyCurve' "$PATCH/adaptive_verification.py" || die "adaptive_verification.py not patched"
  grep -q 'online_forward_start' "$PATCH/model_runner.py" || die "model_runner.py not patched"
  docker ps -a --format '{{.Names}}' | grep -qx "$CTRL" || die "control container $CTRL missing"
  return 0
}
drop_caches() {
  sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
  log "dropped page cache; $(free -g | awk 'NR==2{print "avail="$7"GiB used="$3"GiB"}')"
}
wait_bound() {  # $1=container name
  local name=$1
  local t0=$SECONDS
  log "waiting bind for $name"
  while true; do
    check_stop
    if ! docker ps --format '{{.Names}}' | grep -qx "$name"; then die "container $name not running"; fi
    if curl -sf -m 3 http://127.0.0.1:30006/v1/models 2>/dev/null | grep -q dsv41-flash-uva; then
      log "BOUND $name after $((SECONDS-t0))s"; return 0
    fi
    local elapsed=$((SECONDS-t0))
    local gpu; gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 1 | tr -cd '0-9' || true); gpu=${gpu:-0}
    if [[ $elapsed -gt 10800 ]]; then die "$name boot exceeded 180 min without bind"; fi
    if (( elapsed % 60 < 16 )); then log "still booting $name t=${elapsed}s gpu=${gpu}%"; fi
    sleep 15
  done
}
smoke() {
  python3 - <<'PY'
import json, urllib.request, sys
B="http://127.0.0.1:30006/v1/chat/completions"
def chat(msgs, mx=64, think=False):
    body={"model":"dsv41-flash-uva","messages":msgs,"max_tokens":mx,"temperature":0,"chat_template_kwargs":{"thinking":think}}
    if think: body["reasoning_effort"]="low"
    r=json.load(urllib.request.urlopen(urllib.request.Request(B, data=json.dumps(body).encode(), headers={"Content-Type":"application/json"}), timeout=300))
    m=r["choices"][0]["message"]; return (m.get("content") or ""), m
fails=[]
c,_=chat([{"role":"user","content":"What is 17*19? Return only the integer."}])
print(f"ARITH {c!r}");  "323" in c.replace(",","") or fails.append("arith")
c,_=chat([{"role":"user","content":"Count from 1 to 30 separated by spaces. Only the numbers."}], mx=80)
nums=[int(x) for x in c.split() if x.isdigit()]; nums[:30]==list(range(1,31)) or fails.append("count")
c,m=chat([{"role":"user","content":"What is 15% of 240? Reply with the number."}], mx=400, think=True)
(c or m.get("reasoning_content") or m.get("reasoning")) or fails.append("think")
if fails: print("SMOKE_FAIL", fails); sys.exit(2)
print("SMOKE_PASS")
PY
}
c1_of() { python3 -c "import json;print(round(json.load(open('~/dsv41/knee-$1.json'))['rows'][0][1],1))" 2>/dev/null || echo NA; }
c8_of() { python3 -c "import json;print(round(json.load(open('~/dsv41/knee-$1.json'))['rows'][3][1],1))" 2>/dev/null || echo NA; }
c16_of() { python3 -c "import json;print(round(json.load(open('~/dsv41/knee-$1.json'))['rows'][-1][1],1))" 2>/dev/null || echo NA; }
fx_of() {
  python3 - "$1" <<'PY' 2>/dev/null || echo NA
import json,sys
d=json.load(open(f"~/dsv41/agentfix-{sys.argv[1]}.json")); r=d["res"]
print(" ".join(f"{c}={r[c]['tok_s']}({r[c]['accept']:.2f})" for c in ("prose","shell_ops","code","tool_json","structured") if c in r)+f" wacc={d['weighted_accept']:.3f}")
PY
}
curve_facts() {  # $1=tag  — the boot-vs-online curve lines are the mechanism receipt
  local logf=~/dsv41/dsv41-vllm-$1.log
  {
    echo "=== $1 curve facts $(date) ==="
    grep -aoE "DSpark online verify curve ENABLED[^\"]*" "$logf" | head -1 || echo "online: NOT ENABLED"
    grep -aoE "DSpark boot verify curve[^\"]*" "$logf" | head -1 | cut -c1-600
    grep -aoE "DSpark boot draft curve[^\"]*" "$logf" | head -1 | cut -c1-300
    grep -aoE "DSpark online verify curve after[^\"]*" "$logf" | tail -3 | cut -c1-900
    grep -aoE "Graph capturing finished in [0-9]+ secs, took [0-9.]+ GiB|Available KV cache memory: [0-9.-]+ GiB" "$logf" || true
  } | tee "~/dsv41/results/facts-$1.txt"
}
finish() {
  log "FINISH: stop candidate, keep container"
  docker stop "dsv41-vllm-$TAG" 2>/dev/null || true
  docker stop "$CTRL" 2>/dev/null || true
  log "lane left dark; REF is dsv41-vllm-v12-1M-k5-off60-util97-BOUND-REF (start manually when wanted)"
}

log "ONLINEVC start: online verify curve vs boot curve, adaptive k=5, off${OFFGB}/util${UTIL}, V2 runner"
check_safety

# ---- (1) same-window control: ADAPT-off66 (adaptive, boot curve) ----
ctrl=CONTROL-adapt-for-ONLINEVC-p$PAIR
if [[ "$SKIP_CONTROL" != "1" ]]; then
  drop_caches; log "starting $CTRL for control"; docker start "$CTRL"
  nohup docker logs -f "$CTRL" > ~/dsv41/dsv41-vllm-$ctrl.log 2>&1 &
  wait_bound "$CTRL"
  log "SMOKE control"; smoke || die "control smoke failed"
  log "KNEE control"; bash knee.sh "$ctrl"
  log "FIXTURE control"; bash agent_fixture.sh "$ctrl" || true
  log "CONTROL C1=$(c1_of $ctrl) C8=$(c8_of $ctrl) C16=$(c16_of $ctrl) | $(fx_of $ctrl)"
  docker stop "$CTRL"; log "stopped $CTRL"
  check_stop
fi

# ---- (2) candidate ----
drop_caches
avail=$(awk '/MemAvailable/ {printf "%d", $2/1024/1024}' /proc/meminfo)
[[ "$avail" -lt 20 ]] && die "available ${avail} < 20 after drop"
launched=$(date '+%H:%M')
TAG="$TAG" OFFGB="$OFFGB" UTIL="$UTIL" SEQS=16 CTX=1048576 SPEC="" \
  EXTRA="--long-prefill-token-threshold 6144 --speculative-config {\"method\":\"dspark\",\"num_speculative_tokens\":5,\"enable_adaptive_verification\":true}" \
  DOCKER_ENV="-e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_DSPARK_ONLINE_VERIFY_CURVE=1 -v $PATCH/model_runner.py:$SP/v1/worker/gpu/model_runner.py:ro -v $PATCH/adaptive_verification.py:$SP/v1/worker/gpu/spec_decode/adaptive_verification.py:ro" \
  MODEL="$MODEL" bash launch-dsv41-vllm.sh
logf=~/dsv41/dsv41-vllm-$TAG.log
for i in $(seq 1 30); do
  sleep 10
  if ! docker ps --format '{{.Names}}' | grep -qx "dsv41-vllm-$TAG"; then
    log "FAST-FAIL: container exited"; docker logs --tail 60 "dsv41-vllm-$TAG" 2>&1 | grep -aiE "error|traceback|nameerror|importerror" | tail -n 12; die "boot rejected"
  fi
  if grep -aE "ValueError:|NameError:|ImportError:|AttributeError:|Traceback \(most recent|RuntimeError:" "$logf" 2>/dev/null | grep -qv Warning; then
    log "FAST-FAIL: error in log"; grep -aE "ValueError:|NameError:|ImportError:|AttributeError:|Traceback|RuntimeError:" -A 3 "$logf" | tail -n 16
    sleep 5; curl -sf -m 2 http://127.0.0.1:30006/v1/models >/dev/null 2>&1 || die "boot rejected (log error)"
  fi
done
docker exec "dsv41-vllm-$TAG" grep -c OnlineVerifyCurve "$SP/v1/worker/gpu/spec_decode/adaptive_verification.py" | grep -q '^[1-9]' || die "patch not visible in container"
wait_bound "dsv41-vllm-$TAG"
grep -aq "DSpark online verify curve ENABLED" "$logf" || die "online curve did not enable"
log "SMOKE"; smoke || die "smoke failed"
# warm the online curve before measuring: one fixture pass (~3-5 min of real verify steps), discarded
log "WARM fixture (discarded) $TAG"; bash agent_fixture.sh "$TAG-warm" || true
curve_facts "$TAG"
log "KNEE $TAG"; bash knee.sh "$TAG"
log "FIXTURE $TAG"; bash agent_fixture.sh "$TAG" || true
curve_facts "$TAG"
line="| $TAG | $launched | $(date '+%H:%M') | online verify curve | ctrl C1=$(c1_of $ctrl) C8=$(c8_of $ctrl) C16=$(c16_of $ctrl) $(fx_of $ctrl) | cand C1=$(c1_of $TAG) C8=$(c8_of $TAG) C16=$(c16_of $TAG) | $(fx_of $TAG) |"
log "LEDGER $line"; echo "$(date +%F) ONLINEVC-p$PAIR $line" >> "$LEDGER"
finish
log "ONLINEVC done"
