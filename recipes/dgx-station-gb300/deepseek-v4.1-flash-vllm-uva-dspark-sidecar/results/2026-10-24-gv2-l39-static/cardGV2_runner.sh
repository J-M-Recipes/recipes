#!/usr/bin/env bash
# cardGV2_runner.sh — Card GV2: v25m2 = v25m mixed map with layer 39 restored to the static split. 2026-10-24, Milo.
# James 2026-10-24 ~06:45: "go ahead; document" (after Card GV: 9/10, cosine bring-up FAIL at L39 only).
# Order: K (check-mode boot, v25m2 map) -> cosine gate.
#   K PASS -> P = production v25m2 boot (new name), V battery (same gates as Card GV) + held-out mixed C1/C4 + stall, soak.
#   K FAIL -> P = production v25m (v1 map) boot under its old name, soak restarted there; no V battery (already gated).
# Pre-registered: v25m2 is the soak lane iff K PASS (sc0_gate judge, bar NOT moved) AND the V battery (cardGV_verdict.py) is 10/10.
#   If K PASS but V < 10/10: lane falls back to v25m v1 (boot) and the soak runs there.
set -uo pipefail
DS=$HOME/dsv41; D=$DS/m3; SC=$D/sc2; GV=$D/gv; R=$GV/results-cardGV2; W=$DS/1017; AR=$DS/ar
SC2R=$SC/results-cardSC2
BK=$D/bench/kbench_chat_ds.py; P=$D/bench/k3pool/long; U=http://127.0.0.1:30006; M=dsv41-flash-uva
NAME1=dsv41-vllm-1017-v25m-MIXMAP; MAP1=rowmap-mixed-ar-v1.json
NAME2=dsv41-vllm-1017-v25m2-L39S; MAP2=rowmap-mixed-ar-v2-l39s.json
KN=dsv41-vllm-GV2-K-EXP; SOAK_S=0
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
source <(sed -n '/^log(){/,/^# ================= preflight/p' "$SC/cardSC2_runner.sh" | sed '$d')
other_runners(){ ps -eo pid,args | grep -E '^ *[0-9]+ bash .*(runner\.sh|perclass_run\.sh)' | grep -vE "cardGV2_runner.sh|cardSC2_runner.sh" | grep -v grep | cut -c1-120 | tr '\n' '|'; }
NEEDLE_OK=0; grep -q '"needle-checksum"' "$LB/llm_decode_bench.py" && NEEDLE_OK=1

arm_wd(){ local name=$1
  setsid nohup python3 "$WD" --server "$name" --mode enforce --interval 1 --stall 2 \
    --status "$D/logs/sidecar_watchdog.status.json" --log "$D/logs/sidecar_watchdog.log" > /dev/null 2>&1 < /dev/null &
  echo $! > "$D/logs/sidecar_watchdog.pid"; sleep 3
  kill -0 "$(cat "$D/logs/sidecar_watchdog.pid")" && log "fail-stop watchdog armed pid $(cat "$D/logs/sidecar_watchdog.pid") on $name" || log "WARN: watchdog did not start"; }
start_soak(){ local name=$1 map=$2
  SOAK0=$(date '+%F %T %Z'); SOAK1=$(date -d '+48 hours' '+%F %T %Z')
  shm_snap P soakstart
  printf 'start=%s\nend=%s\nlane=%s\nmap=%s\n' "$SOAK0" "$SOAK1" "$name" "$map" > "$R/SOAK.txt"
  echo "$name" > "$D/logs/soak_lane.txt"
  log "SOAK clock started $SOAK0 -> $SOAK1 on $name ($map)"; }
fallback_v1(){ local why=$1
  log "FALLBACK to v25m v1 ($why)"
  ROWMAP=$MAP1
  if boot_side F "$NAME1" 1 "--kv-cache-memory 9126805504" 45; then log "LANE DARK ends (v25m v1 bound)"; health F "$NAME1"; arm_wd "$NAME1"; start_soak "$NAME1" "$MAP1"
  else log "F FAILED TO BIND — lane dark, needs hands (RESTORE-v25.md)"; fi; }

log "===== CARD GV2 START (candidate $NAME2 map $MAP2; fallback $NAME1 map $MAP1) ====="
o=$(other_runners); [[ -n $o ]] && { log "REFUSE: another runner is live: $o"; exit 4; }
g=$(gpu_foreign); [[ -n ${g// /} ]] && { log "REFUSE foreign GPU container:$g"; exit 4; }
for f in "$H/$MAP1" "$H/$MAP2" "$GATE" "$GPROMPTS" "$SC2R/gsm8k1319-C.json" "$SC2R/tf-C.jsonl" "$TFREF" "$AR/ar_mixed.py" "$AR/v25/shm_sample.py" "$GV/cardGV_verdict.py"; do
  [[ -s $f ]] || { log "ABORT: missing $f"; exit 5; }; done
{ echo "map2_sha1=$(sha1sum "$H/$MAP2" | cut -c1-12) map1_sha1=$(sha1sum "$H/$MAP1" | cut -c1-12)"; for f in "$0" "$SC/cardSC2_runner.sh" "$GATE" "$GPROMPTS" "$FLIPS" "$H/mega_peer_hook.py" "$D/local/peer_server2_local.py"; do echo "sha1 $(sha1sum "$f" | cut -c1-12) $f"; done; } > "$R/instruments.txt"
log "preflight OK MemAvailable=$(memavail)GiB RTX=$(rtx_used)MiB needle=$NEEDLE_OK"

# ================= K: cosine gate boot on v25m2 =================
WP=$(cat "$D/logs/sidecar_watchdog.pid" 2>/dev/null)
[[ -n $WP ]] && kill "$WP" 2>/dev/null && log "fail-stop watchdog $WP stopped for the gate boot"
echo "$(date '+%F %T %Z') watchdog stopped by cardGV2 for the K gate boot (re-armed at P)" >> "$D/logs/sidecar_watchdog.log"
log "LANE DARK begins (K gate boot, soak on $NAME1 cut)"
stop_all; drop_caches
docker rm -f "$KN" "$KN-rtx" >/dev/null 2>&1 || true
t0=$(date +%s); sinceK=$(date -u +%FT%TZ); KPASS=0
if NAME=$KN ROWMAP=$MAP2 bash "$LAUNCH" sidecar >> "$R/sidecar-K.txt" 2>&1 && \
   REPLAY_OVERLAY=1 EXTRA="" MEGA_PEER_CHECK=100000 MEGA_COLD_TRT=1 NAME=$KN ROWMAP=$MAP2 bash "$LAUNCH" server >> "$R/sidecar-K.txt" 2>&1 && \
   wait_bind "$KN" 45 "$sinceK" "" "$KN-rtx"; then
  log "boot->ready K $(( $(date +%s)-t0 ))s"
  docker logs "$KN" > "$R/boot-K.log" 2>&1
  log "FACTS K: check=$(grep -c 'peer mode 2, check 100000' "$R/boot-K.log") rowmap40=$(grep -c "MEGA_PEER rowmap /w/$MAP2: 40 layers" "$R/boot-K.log") | $(grep -m1 'GPU KV cache size' "$R/boot-K.log" | sed 's/.*\] //')"
  python3 "$GATE" sendjsonl $U/v1 $M "$GPROMPTS" > "$R/gate-send-K.log" 2>&1 || log "GATE K: some prompts errored"
  sleep 5
  docker logs "$KN" > "$R/serverlog-K.log" 2>&1
  cp "$D/logs/peer_server2.log" "$R/peer_server2-K.log" 2>/dev/null
  python3 "$GATE" judge "$R/serverlog-K.log" "$R/peer_server2-K.log" > "$R/gate-K.json" 2>&1; rc=$?
  [[ $rc == 0 ]] && KPASS=1
  log "GATE K: $( [ $rc = 0 ] && echo PASS || echo FAIL ) $(python3 -c "
import json
t=open('$R/gate-K.json').read(); j=json.loads(t[t.index('{'):])
print({k:j.get(k) for k in ('post_startup_checks','uncovered_layers_lt32rows','min_cos','max_rel_diff','peer_timeouts','hook_timeout_lines')})" 2>/dev/null)"
else
  log "K BOOT FAILED: $(tail -3 "$R/sidecar-K.txt" | tr '\n' '|')"
  docker logs "$KN" > "$R/boot-K-FAIL.log" 2>&1
fi
docker stop "$KN" "$KN-rtx" >/dev/null 2>&1; docker rename "$KN" "$KN-DONE" 2>/dev/null; docker rename "$KN-rtx" "$KN-DONE-rtx" 2>/dev/null

if [[ $KPASS != 1 ]]; then
  fallback_v1 "K gate did not pass"
else
  # ================= P: production v25m2 + V battery =================
  ROWMAP=$MAP2
  if boot_side V "$NAME2" 1 "--kv-cache-memory 9126805504" 45; then
    log "LANE DARK ends (v25m2 bound)"
    arm_wd "$NAME2"   # enforce during measurement too: a trip = lane fault, verdict FAIL
    sinceV=$(date -u +%FT%TZ)
    shm_snap V start; wdV=$(wd_start V "$NAME2"); psV=$(power_sampler V)
    log "--- MEASURE V ($NAME2) ---"
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
    for c in 1 4; do
      HELDOUT=1 python3 "$AR/ar_mixed.py" v25m2-heldout-c$c "$R" $c 180 > "$R/mixed-heldout-c$c.log" 2>&1 || log "MIXED heldout C$c nonzero rc"
      log "MIXED heldout C$c: $(tail -1 "$R/mixed-heldout-c$c.log" | cut -c1-330)"
    done
    rm -f "$R/stall.jsonl"
    setsid nohup python3 "$AR/v25/shm_sample.py" "$R/stall.jsonl" 400 > /dev/null 2>&1 < /dev/null &
    python3 "$AR/ar_shift.py" v25m2 "$R" 180 > "$R/shift.log" 2>&1 || log "SHIFT nonzero rc"
    log "SHIFT: $(tail -1 "$R/shift.log" | cut -c1-330)"
    health V "$NAME2"
    side_close V "$NAME2" "$wdV" "$psV"
    errcount V "$NAME2" "$sinceV"; log "--- END MEASURE V ---"
    python3 "$GV/cardGV_verdict.py" "$R" "$SC2R" > "$R/VERDICT.md" 2> "$R/verdict.err"
    # cosine gate in GV2 is G9 from K; the verdict script reads 'GATE K:' and gate-K.json from $R — same files.
    log "VERDICT: $(grep -m1 '^\*\*Verdict' "$R/VERDICT.md")"
    if grep -q '^\*\*Verdict: PASS' "$R/VERDICT.md"; then
      log "v25m2 keeps the lane"; start_soak "$NAME2" "$MAP2"
    else
      WP=$(cat "$D/logs/sidecar_watchdog.pid" 2>/dev/null); [[ -n $WP ]] && kill "$WP" 2>/dev/null
      fallback_v1 "v25m2 V battery not 10/10"
    fi
  else
    fallback_v1 "v25m2 production boot failed to bind"
  fi
fi
grep -hE ' (GSM8K1319|TF|FLIPGATE|BFCL|RT|KNEE|REPLAY|TURNS|NEEDLE|EFFORT|IMAGE|GAUNTLET|GATE|FACTS|SHM|TIMEOUTS|POWER|HEALTH|ERRCOUNT|MIXED|SHIFT|boot->ready|BOUND|FASTFAIL|DIED|TIMEOUT|WARN|ABORT|VERDICT|LANE|SOAK|FALLBACK|fail-stop) ' "$R/runner.log" > "$R/SUMMARY.txt"
{ echo; echo "## $(date '+%F %H:%M %Z') — Card GV2: v25m2 ($MAP2, layer 39 static) — $(grep -m1 'GATE K:' "$R/runner.log" | cut -c25-200) — $(grep -m1 '^\*\*Verdict' "$R/VERDICT.md" 2>/dev/null)"
  [[ -s $R/VERDICT.md ]] && cat "$R/VERDICT.md"; echo "Receipts: $R. Soak: $(cat "$R/SOAK.txt" 2>/dev/null | tr '\n' ' ')"; } >> "$LEDGER"
log "===== CARD GV2 END (lane = $(cat "$D/logs/soak_lane.txt" 2>/dev/null)) ====="
