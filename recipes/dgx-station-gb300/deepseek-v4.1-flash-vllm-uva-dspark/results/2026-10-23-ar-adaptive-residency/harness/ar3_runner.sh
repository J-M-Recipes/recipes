#!/usr/bin/env bash
# ar3_runner.sh — v26ar.1 (relaxed atomics + restore_map): overhead A/B (12 pairs),
# mixed-domain traffic C1/C4 frozen vs adaptive, shift reproducibility, and
# round-trip parity (static -> adapted -> restored static).
set -uo pipefail
AR=/home/milo/dsv41/ar
RUN=$AR/run3
S=$RUN/STATUS
OLD=dsv41-vllm-1017-v26ar-AR
NAME=dsv41-vllm-1017-v26ar1-AR3
IMAGE=vllm/vllm-openai:deepseekv41-flash-0909
GB300=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55
HOOK=/home/milo/pin-hot-experts/v26ar1/hook
PROMPTS=/home/milo/pin-hot-experts/e2c/parity-prompts.json
MIX_S=${MIX_S:-180}
PHASE_S=${PHASE_S:-180}
mkdir -p "$RUN"
log() { echo "$(date '+%F %T %Z') $*" | tee -a "$S"; }
die() { log "STOP: $*"; exit 1; }
ctrl() { printf '%s' "$1" > "$RUN/ctrl.tmp" && mv "$RUN/ctrl.tmp" "$RUN/ctrl.json" && touch "$RUN/ctrl.json"; sleep 3; }
parity() { python3 "$AR/e2c_parity.py" "$1" "$PROMPTS" "$RUN" > "$RUN/parity-$1.log" 2>&1 || die "parity $1 failed"; }
cmp() { python3 "$AR/e2c_compare_parity.py" "$RUN/parity-$1.json" "$RUN/parity-$2.json" "$1 vs $2" > "$RUN/cmp-$1-vs-$2.json" 2>&1; \
        log "CMP $1 vs $2: $(python3 -c "import json;d=json.load(open('$RUN/cmp-$1-vs-$2.json'));print(d['exact'],'/',d['n'],[m['id'] for m in d['first_misses']])")"; }

log "AR3 runner start (mix_s=$MIX_S phase_s=$PHASE_S)"
docker run --rm --device "nvidia.com/gpu=$GB300" -e PYTHONUNBUFFERED=1 -v "$AR":/arhost -v "$HOOK":/w:ro \
  -e PIN_HOOK=/w/pin_hot_experts_hook.py -e PIN_ROWMAP=/w/rowmap-static-v1.json -e PIN_COUNTER=atomic \
  -e AR0_OUT=/arhost/run3/ar0-unit.json --entrypoint python3 "$IMAGE" /arhost/ar0_unit.py > "$RUN/ar0.log" 2>&1
rc=$?
log "AR0' rc=$rc $(grep -h VERDICT "$RUN/ar0.log" | tail -1) $(grep -h T6_ "$RUN/ar0.log" | cut -c1-60)"
[ $rc -eq 0 ] || die "AR0' failed"
# AR0' ran while the old lane was up (it needs ~1 GiB); now swap lanes.
docker stop -t 60 "$OLD" >/dev/null 2>&1 && log "stopped (kept) $OLD"
docker rm -f "$NAME" >/dev/null 2>&1 || true
rm -f "$RUN/GO" "$RUN/ctrl.json" "$RUN/telemetry.jsonl" "$RUN/swaps.jsonl"
HOOK=$HOOK NAME=$NAME RUN=$RUN bash "$AR/launch_ar.sh" >> "$S" 2>&1 || die "launch failed"
T0=$(date +%s)
while true; do
  curl -sf -m 5 localhost:30006/v1/models 2>/dev/null | grep -q dsv41-flash-uva && break
  docker ps -q -f "name=^${NAME}$" | grep -q . || { docker logs --tail 120 "$NAME" > "$RUN/boot-fail.log" 2>&1; die "container exited during boot"; }
  [ $(( $(date +%s) - T0 )) -gt 2400 ] && die "boot timeout"
  sleep 10
done
log "AR3 ready after $(( $(date +%s) - T0 )) s"
docker logs "$NAME" 2>&1 | grep -E "PIN_HOT|atomic|swap barrier|KV cache size" > "$RUN/boot-pin.log" || true
grep -q "atomic counter ready" "$RUN/boot-pin.log" && grep -q "atomic mode ready" "$RUN/boot-pin.log" && grep -q "swap barrier" "$RUN/boot-pin.log" || die "hook wiring incomplete"
touch "$RUN/GO"; sleep 8

log "S1a/S1b static-map parity x2"
parity S1a; parity S1b; cmp S1a S1b
log "A/B count on/off, 12 pairs (relaxed atomics)"
python3 "$AR/ar1_ab.py" "$RUN" "$RUN/ar1.json" 12 512 > "$RUN/ar1.log" 2>&1 || die "A/B failed"
log "A/B $(grep -h '^\[ab\] median' "$RUN/ar1.log")"

log "mixed FROZEN C1 ${MIX_S}s"
python3 "$AR/ar_mixed.py" frozen-c1 "$RUN" 1 "$MIX_S" > "$RUN/mixed-frozen-c1.log" 2>&1 || die "mixed frozen c1"
log "  $(tail -1 "$RUN/mixed-frozen-c1.log" | cut -c1-300)"
log "mixed FROZEN C4 ${MIX_S}s"
python3 "$AR/ar_mixed.py" frozen-c4 "$RUN" 4 "$MIX_S" > "$RUN/mixed-frozen-c4.log" 2>&1 || die "mixed frozen c4"
log "  $(tail -1 "$RUN/mixed-frozen-c4.log" | cut -c1-300)"

ctrl '{"freeze": false, "baseline": true}'
log "mixed ADAPTIVE C1 ${MIX_S}s"
python3 "$AR/ar_mixed.py" adaptive-c1 "$RUN" 1 "$MIX_S" > "$RUN/mixed-adaptive-c1.log" 2>&1 || die "mixed adaptive c1"
log "  $(tail -1 "$RUN/mixed-adaptive-c1.log" | cut -c1-300)"
log "mixed ADAPTIVE C4 ${MIX_S}s"
python3 "$AR/ar_mixed.py" adaptive-c4 "$RUN" 4 "$MIX_S" > "$RUN/mixed-adaptive-c4.log" 2>&1 || die "mixed adaptive c4"
log "  $(tail -1 "$RUN/mixed-adaptive-c4.log" | cut -c1-300)"
log "shift ADAPTIVE (reproducibility of AR2)"
python3 "$AR/ar_shift.py" ar3-adaptive "$RUN" "$PHASE_S" > "$RUN/shift-adaptive.log" 2>&1 || die "shift adaptive"
log "  $(tail -1 "$RUN/shift-adaptive.log")"

ctrl '{"freeze": true}'
sleep 5
log "P1 post-adaptation parity"
parity P1
cmp S1a P1
ctrl '{"restore_map": true}'
for i in $(seq 1 60); do
  python3 - <<'PY' > /dev/null 2>&1
import json, urllib.request
b = {"model": "dsv41-flash-uva", "messages": [{"role": "user", "content": "Write two sentences about rivers."}],
     "max_tokens": 64, "temperature": 0, "chat_template_kwargs": {"thinking": False}}
urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:30006/v1/chat/completions",
    data=json.dumps(b).encode(), headers={"Content-Type": "application/json"}), timeout=120).read()
PY
  last=$(tail -1 "$RUN/telemetry.jsonl")
  pend=$(echo "$last" | python3 -c "import sys,json;d=json.loads(sys.stdin.read());print(d.get('pending'), d.get('hot_moved'))")
  [ "$pend" = "0 0" ] && break
done
log "restore drained: pending/hot_moved = $pend after $i requests"
[ "$pend" = "0 0" ] || die "restore did not converge"
log "S2 restored-static parity"
parity S2
cmp S1a S2
cmp S1b S2
python3 "$AR/ar3_verdict.py" "$RUN" /home/milo/dsv41/ar/run > "$RUN/verdict.json" 2> "$RUN/verdict.err"
log "VERDICT $(python3 -c "import json;d=json.load(open('$RUN/verdict.json'));print(d['verdict'], json.dumps(d['gates']))")"
log "AR3 done; $NAME left UP and FROZEN (restored static map) on :30006; $OLD stopped-and-kept"
