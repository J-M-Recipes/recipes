#!/usr/bin/env bash
# Lane B (SGLang, spark1 head :8888) full battery. usage: b-battery.sh <armdir> <arm>
set -u
D="$1"; ARM="$2"
T=~/bench-r2/tools
B=http://127.0.0.1:8888/v1
M=deepseek-v4.1-flash
mkdir -p "$D"; cd "$D"
P="$D/battery.phases"
ph() { echo "$(date -u +%FT%TZ) PHASE $*" >> "$P"; }
dump() { curl -s localhost:8888/metrics > "$D/metrics-$1.txt" 2>/dev/null; }
ph start; dump start
ph gates start; python3 $T/gates.py "$D" > "$D/gates.log" 2>&1; ph gates end rc=$?; dump after-gates
ph qeval start; (cd "$D" && python3 $T/qeval.py run "$ARM" --url $B/chat/completions > "$D/qeval.log" 2>&1); ph qeval end rc=$?; dump after-qeval
ph needles start; python3 $T/long_check.py http://127.0.0.1:8888 20000 131073 300000 300000 300000 > "$D/needles.log" 2>&1; ph needles end rc=$?; dump after-needles
ph c1-run1 start
python3 $T/v41bench_tf.py --base $B --model $M --label "$ARM-c1" --out "$D" --levels 1 --prefill "" --notes "Lane B R2 $ARM (SGLang knapcio v2.4 block, sparks1-4). C1 run 1 of 2." > "$D/c1.log" 2>&1
ph c1-run1 end rc=$?; dump after-c1-run1
ph main start
python3 $T/v41bench_tf.py --base $B --model $M --label "$ARM-main" --out "$D" --levels 1,2,3,4,5,6 --prefill 2000,8000,32000,64000 --notes "Lane B R2 $ARM. C1-C6 (C1 run 2 of 2) + prefill ladder (64000 target = 93K prompt)." > "$D/main.log" 2>&1
ph main end rc=$?; dump after-main
ph prefill-reps start; python3 $T/prefill_reps.py http://127.0.0.1:8888 "$M" 3 2000 8000 32000 64000 93335 > "$D/prefill_reps.log" 2>&1; ph prefill-reps end rc=$?; dump after-prefill-reps
ph mixed start; python3 $T/mixed_probe.py http://127.0.0.1:8888 "$M" 93000 > "$D/mixed.log" 2>&1; ph mixed end rc=$?; dump after-mixed
ph done
