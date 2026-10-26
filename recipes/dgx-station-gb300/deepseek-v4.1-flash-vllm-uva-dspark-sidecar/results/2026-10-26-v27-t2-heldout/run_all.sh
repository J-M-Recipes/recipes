#!/bin/bash
# v27 live-lane T0 + T2 + T1(C1,C4) + BFCL held-out sibling. Read-only, client-side, concurrency <= 4.
set -u
W=~/.hermes/profiles/milo/work/v27-t2
B=~/hermes/lil-bench
HOST=<gb300>; PORT=30006; M=dsv41-flash-uva
BASE=$W/baseline-gsm8k1319-V-v25m.json
LOG=$W/campaign.log
cd "$W"
log(){ echo "$(date '+%F %T %Z') $*" | tee -a "$LOG"; }
# guest-traffic sampler (vLLM /metrics, every 5 s)
( while true; do echo "$(date '+%F %T') $(curl -s -m 4 http://$HOST:$PORT/metrics | grep -E '^vllm:num_requests_(running|waiting)\{' | sed -E 's/\{[^}]*\}//' | tr '\n' ' ')"; sleep 5; done ) > "$W/engine-load.tsv" 2>&1 &
SAMP=$!
trap 'kill $SAMP 2>/dev/null' EXIT
log "START"
log "T0 begin"; "$B/.venv/bin/python" "$W/t0.py" > "$W/t0.json" 2> "$W/t0.err"; log "T0 rc=$? $(tail -c 400 "$W/t0.json" | tr '\n' ' ')"
run(){ local name=$1; shift
  echo "LLM_BENCH_NO_UPDATE_CHECK=1 lil-bench/run.sh (=python3 src/llm_decode_bench.py --display-mode plain) --host $HOST --port $PORT --model $M $* --output $name.json" >> "$W/command.txt"
  log "run $name begin"
  "$B/run.sh" --host $HOST --port $PORT --model $M "$@" --output "$W/$name.json" > "$W/$name.stdout.txt" 2>&1
  log "run $name end rc=$?"; }
run t2-needle --test-profile needle-checksum --profile-concurrency 4
run t2-gsm8k200 --test-profile gsm8k --profile-runs 200 --profile-concurrency 4 --compare-baseline "$BASE"
run t2-tooleval --tool-eval --tool-eval-parallel 4
run t1-perf --concurrency 1,4 --contexts 0,32768,131072 --duration 30
echo "SUITE=heldout CONC=4 MAXTOK=2048 THINKING=0 BASE_URL=http://$HOST:$PORT/v1 MODEL=$M python3 pin-hot-experts/scripts/bfcl_gate.py v27-heldout bfcl_data ." >> "$W/command.txt"
log "BFCL heldout begin"
SUITE=heldout CONC=4 MAXTOK=2048 THINKING=0 BASE_URL=http://$HOST:$PORT/v1 MODEL=$M python3 ~/hermes/pin-hot-experts/scripts/bfcl_gate.py v27-heldout "$W/bfcl_data" "$W" > "$W/bfcl-v27-heldout.log" 2>&1
log "BFCL heldout end rc=$? $(grep -h '^BFCL' "$W/bfcl-v27-heldout.log" | tail -1)"
log "ALL DONE v27-t2"
