#!/usr/bin/env bash
# cardGV_runner.sh — Card GV: fidelity gates for v25m (v25 sidecar + rowmap-mixed-ar-v1.json), 2026-10-24, Milo.
# James 2026-10-23 23:4x: "keep v25m · re-run the fidelity gates on v25m: yes · restart the 48 h soak: yes".
# Reuses the helper/cell functions of sc2/cardSC2_runner.sh verbatim (sourced). Reference = SC2's C arm (v25 static map,
# same image/hook/b12x, 2026-10-21) and SC2's A/D (v23) where SC2 used them. Arms:
#   V = live v25m production lane (no reboot): TF capture, GSM8K-1319, BFCL dev+heldout, gauntlet, rt C1, knee C1/8/16,
#       agent replay + per-turn tool emission, needle 500, reasoning_effort + image probes, shm timeouts.
#   K = check-mode boot of v25m (MEGA_PEER_CHECK=100000 MEGA_COLD_TRT=1): SC0 cosine gate on the frozen gate prompts.
#   P = fresh PRODUCTION boot of v25m (same container name, --kv-cache-memory 9126805504), fail-stop watchdog re-armed,
#       48 h soak clock started. Lane dark only K + P (~20-30 min).
# PRE-REGISTERED GATES (cardGV_verdict.py; PASS iff all; missing = FAIL):
#   G1 TF margin-weighted flips GATE PASS for V vs tf-CTL1 (v23 P2 capture; SC2 gate 1 reference) AND V vs SC2 tf-C (v25 static)
#   G2 GSM8K-1319 V >= SC2-C - 1.0 pt            G3 BFCL dev and heldout V >= SC2-C - 1 pt each
#   G4 gauntlet 12/12                            G5 needle-checksum 500/500 exact
#   G6 reasoning_effort HTTP codes == SC2-C      G7 image: http 200, names the content
#   G8 per-turn tool emission: V tool per pass >= min(SC2 A,D) - 1; text spurious <= max(SC2 A,D) + 1; 0 errors
#   G9 cosine gate (K): sc0_gate judge PASS (40/40 layers >= 32 rows, min cos >= 0.999, 0 timeouts)
#   G10 0 sidecar timeouts over V (shm words[2] delta) and K (peer log)
# Speed cells (rt, knee) are REPORTED, not gated. No promotion decision; status stays experimental until the soak ends.
set -uo pipefail
DS=$HOME/dsv41; D=$DS/m3; SC=$D/sc2; GV=$D/gv; R=$GV/results-cardGV; W=$DS/1017
SC2R=$SC/results-cardSC2
BK=$D/bench/kbench_chat_ds.py; P=$D/bench/k3pool/long; U=http://127.0.0.1:30006; M=dsv41-flash-uva
NAME=dsv41-vllm-1017-v25m-MIXMAP; KN=dsv41-vllm-GV-K-EXP
ROWMAP=rowmap-mixed-ar-v1.json; SOAK_S=0
IMAGE=vllm/vllm-openai@sha256:3af6a45b060d681aa6254200f87859d1bb3c1f962afb3d15540b4b31d4a41063
H=$D/el8/deepseek-v4.1-flash/m3/hook; LAUNCH=$D/launch_sidecar.sh; WD=$D/sidecar_watchdog.py; SHM=/dev/shm/vllm_peer_tier2
PROBE=$SC/sc2_probes.py
BENCH=$DS/bench; PROMPTS=$BENCH/perclass_prompts.jsonl
LB=$W/lilbench/app
S=$HOME/pin-hot-experts/scripts; BFCL=$HOME/pin-hot-experts/bfcl_data
CORPUS=$HOME/pin-hot-experts/tf_corpus.jsonl; TFREF=$W/results-cardP2/tf-CTL1.jsonl; TFCAL=$W/results-cardR/tf-R2b.jsonl; FLIPS=$W/flips_margin_p2.py
GDIR=$HOME/pin-hot-experts/round7/gauntlet
GIMG=sha256:00d577a6a63281e15336029d5bcee4e9a2cf182214a4f20ba6111b1c8e79893d
GB300=${GB300_UUID}; RTX=${RTX_UUID}
GATE=$D/sc3/sc0_gate.py; GPROMPTS=$D/sc3/sc-gate-prompts-v1.jsonl
LEDGER=$DS/results/ledger.md
FF=' ERROR [0-9]{2}-[0-9]{2} |Traceback \(most recent call last\)|ValueError:|RuntimeError:|OutOfMemoryError|EngineCore failed'
[[ -e $R/runner.log ]] && { echo "ABORT: $R/runner.log exists" >&2; exit 5; }
mkdir -p "$R"
exec >> "$R/runner.log" 2>&1
# SC2 helpers + cells, verbatim (everything from log(){ up to the preflight banner)
source <(sed -n '/^log(){/,/^# ================= preflight/p' "$SC/cardSC2_runner.sh" | sed '$d')
other_runners(){ ps -eo pid,args | grep -E '^ *[0-9]+ bash .*(runner\.sh|perclass_run\.sh)' | grep -vE "cardGV_runner.sh|cardSC2_runner.sh" | grep -v grep | cut -c1-120 | tr '\n' '|'; }
NEEDLE_OK=0; grep -q '"needle-checksum"' "$LB/llm_decode_bench.py" && NEEDLE_OK=1

log "===== CARD GV START (lane $NAME, map $ROWMAP, ref $SC2R) ====="
o=$(other_runners); [[ -n $o ]] && { log "REFUSE: another runner is live: $o"; exit 4; }
g=$(gpu_foreign); [[ -n ${g// /} ]] && { log "REFUSE foreign GPU container:$g"; exit 4; }
is_up "$NAME" && is_up "$NAME-rtx" || { log "ABORT: $NAME pair not running"; exit 5; }
curl -s -m 3 $U/v1/models | grep -q $M || { log "ABORT: lane not bound"; exit 5; }
for f in "$SC2R/gsm8k1319-C.json" "$SC2R/tf-C.jsonl" "$SC2R/effort-C.json" "$SC2R/turns-SC2-A.json" "$SC2R/turns-SC2-D.json" \
         "$TFREF" "$TFCAL" "$FLIPS" "$CORPUS" "$GATE" "$GPROMPTS" "$H/$ROWMAP" "$PROBE" "$S/bfcl_gate.py" "$S/tf_logprob.py"; do
  [[ -s $f ]] || { log "ABORT: missing $f"; exit 5; }; done
{ echo "needle=$NEEDLE_OK map_sha1=$(sha1sum "$H/$ROWMAP" | cut -c1-12)"; for f in "$0" "$SC/cardSC2_runner.sh" "$GATE" "$GPROMPTS" "$FLIPS" "$H/mega_peer_hook.py" "$D/local/peer_server2_local.py"; do echo "sha1 $(sha1sum "$f" | cut -c1-12) $f"; done; } > "$R/instruments.txt"
log "preflight OK MemAvailable=$(memavail)GiB RTX=$(rtx_used)MiB needle=$NEEDLE_OK"

# ================= V: live v25m =================
docker logs "$NAME" > "$R/boot-V.log" 2>&1
log "FACTS V: $(grep -m1 'MEGA_PEER rowmap' "$R/boot-V.log" | cut -c1-120) | $(grep -m1 'GPU KV cache size' "$R/boot-V.log" | sed 's/.*\] //')"
sinceV=$(date -u +%FT%TZ)
shm_snap V start; wdV=$(wd_start V "$NAME"); psV=$(power_sampler V)
log "--- MEASURE V ($NAME) ---"
tfcap V
gsm8k_full V "$SC2R/gsm8k1319-C.json"
bfcl V dev; bfcl V heldout
gauntlet V
rt_c1 V; knee12 V
replay V
if [[ $NEEDLE_OK == 1 ]]; then needle V; else log "NEEDLE V: instrument absent — PENDING"; fi
probes V; metrics V
flips "V-vs-CTL1" "$TFREF" "$R/tf-V.jsonl" "$R/flips-V-vsCTL1.txt"
flips "V-vs-v25" "$SC2R/tf-C.jsonl" "$R/tf-V.jsonl" "$R/flips-V-vsv25.txt"
health V "$NAME"
side_close V "$NAME" "$wdV" "$psV"
errcount V "$NAME" "$sinceV"; log "--- END MEASURE V ---"

# ================= K: cosine gate boot =================
WP=$(cat "$D/logs/sidecar_watchdog.pid" 2>/dev/null)
[[ -n $WP ]] && kill "$WP" 2>/dev/null && log "fail-stop watchdog $WP stopped for the gate boot"
echo "$(date '+%F %T %Z') watchdog stopped by cardGV for the K gate boot (re-armed at P)" >> "$D/logs/sidecar_watchdog.log"
log "LANE DARK begins (K gate boot)"
stop_all; drop_caches
docker rm -f "$KN" "$KN-rtx" >/dev/null 2>&1 || true
t0=$(date +%s); sinceK=$(date -u +%FT%TZ)
KOK=0
if NAME=$KN ROWMAP=$ROWMAP bash "$LAUNCH" sidecar >> "$R/sidecar-K.txt" 2>&1 && \
   REPLAY_OVERLAY=1 EXTRA="" MEGA_PEER_CHECK=100000 MEGA_COLD_TRT=1 NAME=$KN ROWMAP=$ROWMAP bash "$LAUNCH" server >> "$R/sidecar-K.txt" 2>&1 && \
   wait_bind "$KN" 45 "$sinceK" "" "$KN-rtx"; then
  log "boot->ready K $(( $(date +%s)-t0 ))s"
  docker logs "$KN" > "$R/boot-K.log" 2>&1
  log "FACTS K: check=$(grep -c 'peer mode 2, check 100000' "$R/boot-K.log") rowmap40=$(grep -c "MEGA_PEER rowmap /w/$ROWMAP: 40 layers" "$R/boot-K.log") | $(grep -m1 'GPU KV cache size' "$R/boot-K.log" | sed 's/.*\] //')"
  python3 "$GATE" sendjsonl $U/v1 $M "$GPROMPTS" > "$R/gate-send-K.log" 2>&1 || log "GATE K: some prompts errored"
  sleep 5
  docker logs "$KN" > "$R/serverlog-K.log" 2>&1
  cp "$D/logs/peer_server2.log" "$R/peer_server2-K.log" 2>/dev/null
  python3 "$GATE" judge "$R/serverlog-K.log" "$R/peer_server2-K.log" > "$R/gate-K.json" 2>&1; rc=$?
  log "GATE K: $( [ $rc = 0 ] && echo PASS || echo FAIL ) $(python3 -c "
import json
t=open('$R/gate-K.json').read(); j=json.loads(t[t.index('{'):])
print({k:j.get(k) for k in ('post_startup_checks','uncovered_layers_lt32rows','min_cos','max_rel_diff','peer_timeouts','hook_timeout_lines')})" 2>/dev/null)"
  KOK=1
else
  log "K BOOT FAILED: $(tail -3 "$R/sidecar-K.txt" | tr '\n' '|')"
  docker logs "$KN" > "$R/boot-K-FAIL.log" 2>&1
fi
docker stop "$KN" "$KN-rtx" >/dev/null 2>&1; docker rename "$KN" "$KN-DONE" 2>/dev/null; docker rename "$KN-rtx" "$KN-DONE-rtx" 2>/dev/null

# ================= P: production v25m, watchdog, soak clock =================
if boot_side P "$NAME" 1 "--kv-cache-memory 9126805504" 45; then
  log "LANE DARK ends (P bound)"
  health P "$NAME"
  setsid nohup python3 "$WD" --server "$NAME" --mode enforce --interval 1 --stall 2 \
    --status "$D/logs/sidecar_watchdog.status.json" --log "$D/logs/sidecar_watchdog.log" > /dev/null 2>&1 < /dev/null &
  echo $! > "$D/logs/sidecar_watchdog.pid"; sleep 3
  kill -0 "$(cat "$D/logs/sidecar_watchdog.pid")" && log "fail-stop watchdog re-armed pid $(cat "$D/logs/sidecar_watchdog.pid")" || log "WARN: watchdog did not start"
  SOAK0=$(date '+%F %T %Z'); SOAK1=$(date -d '+48 hours' '+%F %T %Z')
  shm_snap P soakstart
  printf 'start=%s\nend=%s\nlane=%s\nmap=%s\n' "$SOAK0" "$SOAK1" "$NAME" "$ROWMAP" > "$R/SOAK.txt"
  log "SOAK clock started $SOAK0 -> $SOAK1"
else
  log "P FAILED TO BIND — lane dark, needs hands (BOUND-REF v25 static is stopped-and-kept: RESTORE-v25.md)"
fi

python3 "$GV/cardGV_verdict.py" "$R" "$SC2R" > "$R/VERDICT.md" 2> "$R/verdict.err"
log "VERDICT: $(grep -m1 '^\*\*Verdict' "$R/VERDICT.md")"
grep -hE ' (GSM8K1319|TF|FLIPGATE|BFCL|RT|KNEE|REPLAY|TURNS|NEEDLE|EFFORT|IMAGE|GAUNTLET|GATE|FACTS|SHM|TIMEOUTS|POWER|HEALTH|ERRCOUNT|boot->ready|BOUND|FASTFAIL|DIED|TIMEOUT|WARN|ABORT|VERDICT|LANE|SOAK) ' "$R/runner.log" > "$R/SUMMARY.txt"
{ echo; echo "## $(date '+%F %H:%M %Z') — Card GV: fidelity gates for v25m (v25 sidecar + $ROWMAP) — $(grep -m1 '^\*\*Verdict' "$R/VERDICT.md" 2>/dev/null)"
  cat "$R/VERDICT.md"; echo "Receipts: $R. 48 h soak: $(cat "$R/SOAK.txt" 2>/dev/null | tr '\n' ' ')"; } >> "$LEDGER"
log "===== CARD GV END (lane = $NAME) ====="
