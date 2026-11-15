#!/bin/bash
set -u
W=~/.hermes/profiles/milo/work/pg62-t2; B=~/hermes/lil-bench; HOST=<gb300>; PORT=30006; M=dsv41-flash-uva
G4=~/.hermes/profiles/milo/work/g4-t3/Pg61; LOG=$W/campaign.log; cd "$W"
log(){ echo "$(date '+%F %T %Z') $*" | tee -a "$LOG"; }
( while true; do echo "$(date '+%F %T') $(curl -s -m 4 http://$HOST:$PORT/metrics | grep -E '^vllm:num_requests_(running|waiting)\{' | sed -E 's/\{[^}]*\}//' | tr '\n' ' ')"; sleep 5; done ) > "$W/engine-load.tsv" 2>&1 &
SAMP=$!; trap 'kill $SAMP 2>/dev/null' EXIT
{ echo "bfcl_gate.py sha1 $(shasum ~/hermes/pin-hot-experts/scripts/bfcl_gate.py | cut -c1-12)"; echo "baseline gsm8k $(shasum -a 256 $G4/t3-gsm8k1319.json | cut -c1-16) mmlu $(shasum -a 256 $G4/t3-mmlupro1000.json | cut -c1-16) bfcl $(shasum -a 256 ~/.hermes/profiles/milo/work/g3-pg61/bfcl-g3-pg61-heldout.jsonl | cut -c1-16)"; } > facts.txt
log "START (PREREG sha $(shasum -a 256 PREREG.md | cut -c1-16))"
log "T0 begin"; "$B/.venv/bin/python" "$W/t0.py" > "$W/t0.json" 2> "$W/t0.err"; log "T0 rc=$? $(tail -c 300 "$W/t0.json" | tr '\n' ' ')"
run(){ local name=$1; shift
  echo "LLM_BENCH_NO_UPDATE_CHECK=1 run.sh --host <gb300> --port $PORT --model $M $* --output $name.json" >> "$W/command.txt"
  log "run $name begin"; "$B/run.sh" --host $HOST --port $PORT --model $M "$@" --output "$W/$name.json" > "$W/$name.stdout.txt" 2>&1; log "run $name end rc=$?"; }
run t3-needle --test-profile needle-checksum --profile-concurrency 4
run t3-gsm8k1319 --test-profile gsm8k --profile-runs 1319 --profile-concurrency 4 --compare-baseline "$G4/t3-gsm8k1319.json"
run t3-tooleval --tool-eval --tool-eval-parallel 4
log "BFCL heldout begin"
SUITE=heldout CONC=4 MAXTOK=2048 THINKING=0 BASE_URL=http://$HOST:$PORT/v1 MODEL=$M python3 ~/hermes/pin-hot-experts/scripts/bfcl_gate.py pg62-heldout "$W/bfcl_data" "$W" > "$W/bfcl-pg62-heldout.log" 2>&1
log "BFCL heldout end rc=$? $(grep -h '^BFCL' "$W/bfcl-pg62-heldout.log" | tail -1)"
run t3-mmlupro1000 --test-profile mmlu-pro --profile-runs 1000 --profile-concurrency 4
ssh -o BatchMode=yes milo@$HOST 'docker ps --format "{{.Names}}" | tr "\n" " "; cat ~/dsv41/garner/pg62-work/wdp3.status.json; grep -c TRIP ~/dsv41/garner/pg62-work/wdp3.log' 2>/dev/null | grep -v -e '🦝' -e 'AI magic' > "$W/lane-after.txt"
log "ALL DONE pg62-t2 $(cat $W/lane-after.txt | tr '\n' ' ' | cut -c1-260)"
