#!/usr/bin/env bash
# Lane C per-arm battery (published form of the ladder's arm-battery.sh). usage: BASE_URL=http://<head>:8002/v1 run_battery.sh <outdir> <label> [needle] [mixed] [full] [greedy]
#  default (abbreviated): gates -> C1 run1 -> C1,C4,C6 + prefill 8K,93K -> per-category acceptance
#  needle: + 300K needle x1     mixed: + mixed-traffic probe (93K prefill + short chat TTFT)
#  full:   gates -> qeval -> needles 20K/131K/300Kx3 -> C1 run1 -> C1..C6 + prefill 2K/8K/32K/64K(93K) -> acceptance -> mixed
set -u
D="$1"; ARM="$2"; shift 2
NEEDLE=0; MIXED=0; FULL=0; GREEDY=0
for f in "$@"; do case $f in needle) NEEDLE=1;; mixed) MIXED=1;; full) FULL=1;; greedy) GREEDY=1;; esac; done
T=$(cd "$(dirname "$0")" && pwd)
B=${BASE_URL:-http://127.0.0.1:8002/v1}; B=${B%/}; R=${B%/v1}
M=${MODEL:-deepseek-v4.1-flash-spark-vllm}
mkdir -p "$D"; cd "$D"
P="$D/battery.phases"
ph() { echo "$(date -u +%FT%TZ) PHASE $*" >> "$P"; }
dump() { curl -s "$R/metrics" > "$D/metrics-$1.txt"; grep -E '^vllm:num_preemptions_total|^vllm:spec_decode_num_(drafts|draft_tokens|accepted_tokens)_total' "$D/metrics-$1.txt" | sed "s/^/[$1] /" >> "$P"; }
ph start; dump start
ph gates start; BASE_URL=$B MODEL=$M python3 $T/gates.py "$D" > "$D/gates.log" 2>&1; ph gates end rc=$?; dump after-gates
if [ $FULL -eq 1 ]; then
  ph qeval start; (cd "$D" && python3 $T/qeval.py run "$ARM" --url $B/chat/completions > "$D/qeval.log" 2>&1); ph qeval end rc=$?; dump after-qeval
  ph needles start; python3 $T/long_check.py "$R" 20000 131073 300000 300000 300000 > "$D/needles.log" 2>&1; ph needles end rc=$?; dump after-needles
fi
ph c1-run1 start
python3 $T/v41bench_tf.py --base $B --model $M --label "$ARM-c1" --out "$D" --levels 1 --prefill "" --notes "Lane C ladder $ARM. C1 run 1 of 2." > "$D/c1.log" 2>&1
ph c1-run1 end rc=$?; dump after-c1-run1
if [ $FULL -eq 1 ]; then LV=1,2,3,4,5,6; PF=2000,8000,32000,64000; else LV=1,4,6; PF=8000,64000; fi
ph main start
python3 $T/v41bench_tf.py --base $B --model $M --label "$ARM-main" --out "$D" --levels $LV --prefill $PF --notes "Lane C ladder $ARM. C$LV (C1 run 2 of 2) + prefill $PF (64000 target = 93K prompt)." > "$D/main.log" 2>&1
ph main end rc=$?; dump after-main
if [ $FULL -eq 1 ]; then RL="2000 8000 32000 64000 93335"; else RL="11592 93335"; fi
ph prefill-reps start; python3 $T/prefill_reps.py "$R" "$M" 3 $RL > "$D/prefill_reps.log" 2>&1; ph prefill-reps end rc=$?; dump after-prefill-reps
ph acceptance start; python3 $T/spec_by_cat.py $B $M "$D" 3 > "$D/acceptance.log" 2>&1; ph acceptance end rc=$?; dump after-acceptance
if [ $NEEDLE -eq 1 ] && [ $FULL -eq 0 ]; then
  ph needle300 start; python3 $T/long_check.py "$R" 300000 > "$D/needle300.log" 2>&1; ph needle300 end rc=$?; dump after-needle300
fi
if [ $MIXED -eq 1 ]; then
  ph mixed start; python3 $T/mixed_probe.py "$R" "$M" 93000 > "$D/mixed.log" 2>&1; ph mixed end rc=$?; dump after-mixed
fi
if [ $GREEDY -eq 1 ]; then  # r2-arm.sh ran this after every battery
  ph greedy start; MODEL=$M python3 $T/greedy_probe.py "$R" "$D/greedy-$ARM.json" > "$D/greedy.log" 2>&1; ph greedy end rc=$?
fi
ph done
