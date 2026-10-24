#!/usr/bin/env bash
# v25m_runner.sh — AR5: production v25 (sidecar) with ONLY the rowmap changed:
# rowmap-static-v1.json -> rowmap-mixed-ar-v1.json (fitted on AR1+AR2 mixed decode counts).
# Same image, hook, b12x, flags, KV declaration (--kv-cache-memory 9126805504), env.
set -uo pipefail
AR=$HOME/dsv41/ar
RUN=$AR/v25m
S=$RUN/STATUS
M3=$HOME/dsv41/m3
HOOK=$M3/el8/deepseek-v4.1-flash/m3/hook
REF=dsv41-vllm-1017-v25-sidecar-BOUND-REF
NAME=dsv41-vllm-1017-v25m-MIXMAP
MAP=rowmap-mixed-ar-v1.json
PROMPTS=$HOME/pin-hot-experts/e2c/parity-prompts.json
mkdir -p "$RUN"
log() { echo "$(date '+%F %T %Z') $*" | tee -a "$S"; }
die() { log "STOP: $*"; exit 1; }

log "AR5 start"
cp -n "$AR/v25/$MAP" "$HOOK/$MAP" 2>/dev/null || true
cmp -s "$AR/v25/$MAP" "$HOOK/$MAP" || die "rowmap copy into hook dir differs"
docker stop -t 60 "$REF" "$REF-rtx" >/dev/null 2>&1; log "stopped (kept) $REF pair"
sleep 3
NAME=$NAME ROWMAP=$MAP bash "$M3/launch_sidecar.sh" sidecar >> "$S" 2>&1 || die "sidecar launch failed"
log "sidecar serving"
PEER_SPINS=2000000000 REPLAY_OVERLAY=1 EXTRA="--kv-cache-memory 9126805504" MEGA_PEER_CHECK=0 MEGA_COLD_TRT=0 \
  NAME=$NAME ROWMAP=$MAP bash "$M3/launch_sidecar.sh" server >> "$S" 2>&1 || die "server launch failed"
T0=$(date +%s)
while true; do
  curl -sf -m 5 localhost:30006/v1/models 2>/dev/null | grep -q dsv41-flash-uva && break
  docker ps -q -f "name=^${NAME}$" | grep -q . || { docker logs --tail 120 "$NAME" > "$RUN/boot-fail.log" 2>&1; die "server exited during boot"; }
  docker ps -q -f "name=^${NAME}-rtx$" | grep -q . || die "sidecar exited during boot"
  [ $(( $(date +%s) - T0 )) -gt 2700 ] && die "boot timeout"
  sleep 10
done
docker logs "$NAME" > "$RUN/boot.log" 2>&1
log "bound after $(( $(date +%s) - T0 )) s | $(grep -m1 "MEGA_PEER rowmap" "$RUN/boot.log" | cut -c1-120) | $(grep -m1 'GPU KV cache size' "$RUN/boot.log" | sed 's/.*\] //')"
grep -q "MEGA_PEER rowmap /w/$MAP: 40 layers" "$RUN/boot.log" || die "server did not load $MAP"
grep -q "peer mode 2, check 0" "$RUN/boot.log" || die "not peer mode 2"
docker logs "$NAME-rtx" 2>&1 | grep -m1 -E "rowmap|prepared 40 layers" >> "$S"
python3 - <<'PY' >> "$S" 2>&1
import json, urllib.request
b = {"model": "dsv41-flash-uva", "messages": [{"role": "user", "content": "What time is it on the box? Use the tool."}],
     "tools": [{"type": "function", "function": {"name": "terminal", "description": "Run a shell command",
               "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}],
     "tool_choice": "auto", "max_tokens": 256, "chat_template_kwargs": {"thinking": False}}
r = json.loads(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:30006/v1/chat/completions",
    data=json.dumps(b).encode(), headers={"Content-Type": "application/json"}), timeout=300).read())
m = r["choices"][0]["message"]
print("SMOKE tool:", [c["function"]["name"] for c in (m.get("tool_calls") or [])])
PY
for c in 1 4; do
  log "held-out mixed C$c"
  HELDOUT=1 python3 "$AR/ar_mixed.py" v25m-heldout-c$c "$RUN" $c 180 > "$RUN/mixed-heldout-c$c.log" 2>&1 || die "heldout c$c"
  log "  $(tail -1 "$RUN/mixed-heldout-c$c.log" | cut -c1-330)"
done
for c in 1 4; do
  log "in-sample mixed C$c"
  python3 "$AR/ar_mixed.py" v25m-c$c "$RUN" $c 180 > "$RUN/mixed-c$c.log" 2>&1 || die "mixed c$c"
  log "  $(tail -1 "$RUN/mixed-c$c.log" | cut -c1-330)"
done
log "shift protocol + stall sampling"
rm -f "$RUN/stall.jsonl"
setsid nohup python3 "$AR/v25/shm_sample.py" "$RUN/stall.jsonl" 700 > /dev/null 2>&1 < /dev/null &
python3 "$AR/ar_shift.py" v25m "$RUN" 180 > "$RUN/shift.log" 2>&1 || die "shift"
log "  $(tail -1 "$RUN/shift.log")"
log "parity capture"
python3 "$AR/e2c_parity.py" v25m "$PROMPTS" "$RUN" > "$RUN/parity.log" 2>&1 || die "parity"
python3 "$AR/e2c_compare_parity.py" "$AR/v25/parity-v25-static.json" "$RUN/parity-v25m.json" "v25 static vs v25m" > "$RUN/cmp-parity.json" 2>&1
log "PARITY $(python3 -c "import json;d=json.load(open('$RUN/cmp-parity.json'));print(d['exact'],'/',d['n'],[m['id'] for m in d['first_misses']])")"
log "AR5 done; $NAME serving on :30006 (no watchdog); $REF pair stopped-and-kept"
