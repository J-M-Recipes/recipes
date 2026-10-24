#!/usr/bin/env bash
# ar_runner.sh — adaptive-residency campaign AR0 -> AR1 -> AR2 on the GB300.
# Detached: setsid nohup bash ar_runner.sh > run/runner.log 2>&1 &
# Progress: run/STATUS (one line per milestone). Stops on the first hard failure.
set -uo pipefail
AR=/home/milo/dsv41/ar
RUN=$AR/run
S=$RUN/STATUS
NAME=dsv41-vllm-1017-v26ar-AR
IMAGE=vllm/vllm-openai:deepseekv41-flash-0909
GB300=GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55
HOOK=/home/milo/pin-hot-experts/v26ar/hook
PROMPTS=/home/milo/pin-hot-experts/e2c/parity-prompts.json
PHASE_S=${PHASE_S:-180}
mkdir -p "$RUN"
log() { echo "$(date '+%F %T %Z') $*" | tee -a "$S"; }
die() { log "STOP: $*"; exit 1; }
ctrl() {  # ctrl '{"freeze": false}'
  printf '%s' "$1" > "$RUN/ctrl.tmp" && mv "$RUN/ctrl.tmp" "$RUN/ctrl.json" && touch "$RUN/ctrl.json"
  sleep 3
}

log "runner start (phase_s=$PHASE_S)"

# ---------------- AR0: unit tests, no model ----------------
log "AR0 unit tests (GB300, image $IMAGE)"
docker run --rm --device "nvidia.com/gpu=$GB300" -e PYTHONUNBUFFERED=1 \
  -v "$AR":/arhost -v "$HOOK":/w:ro \
  -e PIN_HOOK=/w/pin_hot_experts_hook.py -e PIN_ROWMAP=/w/rowmap-static-v1.json \
  -e PIN_COUNTER=atomic -e AR0_OUT=/arhost/run/ar0-unit.json \
  --entrypoint python3 "$IMAGE" /arhost/ar0_unit.py > "$RUN/ar0.log" 2>&1
rc=$?
log "AR0 rc=$rc $(grep -h 'VERDICT' "$RUN/ar0.log" | tail -1)"
[ $rc -eq 0 ] || die "AR0 failed (see run/ar0.log)"

# ---------------- AR1: boot frozen (count only) ----------------
docker ps --format '{{.Names}} {{.Ports}}' | grep -q . && docker ps --format '{{.Names}}' | grep -qv '^$' && \
  log "containers running before boot: $(docker ps --format '{{.Names}}' | tr '\n' ' ')"
docker rm -f "$NAME" >/dev/null 2>&1 || true
rm -f "$RUN/GO" "$RUN/ctrl.json" "$RUN/telemetry.jsonl" "$RUN/swaps.jsonl"
NAME=$NAME RUN=$RUN bash "$AR/launch_ar.sh" >> "$S" 2>&1 || die "launch failed"
T0=$(date +%s)
log "AR1 boot launched $NAME"
while true; do
  if curl -sf -m 5 localhost:30006/v1/models 2>/dev/null | grep -q dsv41-flash-uva; then break; fi
  if ! docker ps -q -f "name=^${NAME}$" | grep -q .; then
    docker logs --tail 120 "$NAME" > "$RUN/boot-fail.log" 2>&1
    die "container exited during boot (run/boot-fail.log)"
  fi
  [ $(( $(date +%s) - T0 )) -gt 2400 ] && die "boot timeout 40 min"
  sleep 10
done
log "AR1 ready after $(( $(date +%s) - T0 )) s"
docker logs "$NAME" 2>&1 | grep -E "PIN_HOT|pin-hot|atomic|e4|swap barrier|rehome" > "$RUN/boot-pin.log" || true
grep -q "atomic counter ready" "$RUN/boot-pin.log" || die "atomic counter not allocated (run/boot-pin.log)"
grep -q "atomic mode ready" "$RUN/boot-pin.log" || die "atomic mode not initialised"
grep -q "swap barrier" "$RUN/boot-pin.log" || die "execute_model barrier not wrapped"
docker logs "$NAME" 2>&1 | grep -iE "KV cache|Available KV|GPU KV cache size|Maximum concurrency" | tail -4 >> "$RUN/boot-pin.log"
touch "$RUN/GO"
sleep 8
grep -q "baseline (GO)" <(docker logs "$NAME" 2>&1) || log "WARN: GO baseline line not seen yet"

log "AR1a parity capture (frozen map, counting on)"
python3 "$AR/e2c_parity.py" ar1-frozen "$PROMPTS" "$RUN" > "$RUN/parity-ar1.log" 2>&1 || die "parity capture ar1 failed"
log "AR1b same-boot A/B count on/off + TTFT sweep"
python3 "$AR/ar1_ab.py" "$RUN" "$RUN/ar1.json" 6 512 > "$RUN/ar1.log" 2>&1 || die "ar1_ab failed"
log "AR1 $(grep -h '^\[ab\] median' "$RUN/ar1.log")"

# ---------------- AR2: domain shift, frozen vs adaptive ----------------
log "AR2a shift protocol, FROZEN static map"
python3 "$AR/ar_shift.py" ar2-frozen "$RUN" "$PHASE_S" > "$RUN/shift-frozen.log" 2>&1 || die "frozen shift failed"
log "AR2a $(tail -1 "$RUN/shift-frozen.log")"
log "AR2b unfreeze + EWMA baseline"
ctrl '{"freeze": false, "baseline": true}'
python3 "$AR/ar_shift.py" ar2-adaptive "$RUN" "$PHASE_S" > "$RUN/shift-adaptive.log" 2>&1 || die "adaptive shift failed"
log "AR2b $(tail -1 "$RUN/shift-adaptive.log")"
ctrl '{"freeze": true, "dump": "post"}'
sleep 10
log "AR2c post-swap parity (frozen again)"
python3 "$AR/e2c_parity.py" ar2-postswap "$PROMPTS" "$RUN" > "$RUN/parity-ar2.log" 2>&1 || die "parity capture ar2 failed"
python3 "$AR/e2c_compare_parity.py" "$RUN/parity-ar1-frozen.json" "$RUN/parity-ar2-postswap.json" "pre vs post swap" > "$RUN/parity-compare.json" 2>&1
log "AR2c $(tr -d '\n ' < "$RUN/parity-compare.json" | cut -c1-160)"
python3 "$AR/ar_verdict.py" "$RUN" > "$RUN/verdict.json" 2> "$RUN/verdict.err"
log "VERDICT $(python3 -c "import json;d=json.load(open('$RUN/verdict.json'));print(d['verdict'], json.dumps(d['gates']))")"
log "runner done; $NAME left UP and FROZEN on :30006 (no restore-on-done)"
