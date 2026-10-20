#!/usr/bin/env bash
# Lane B (SGLang) full round-2 battery. usage: BASE_URL=http://<head>:8888 b-battery.sh <armdir> <arm>
# Published copy (J-M-Recipes dgx-spark-4x/deepseek-v4.1-flash-sglang-tp4-roce): box paths -> env/BASE_URL; logic unchanged.
# Needs DSV41_ENCODING_DIR (checkpoint encoding/ dir) for v41bench_tf.py. Sends real load: never point it at a lane serving users.
set -u
D="$1"; ARM="$2"
T="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_URL="${BASE_URL:-http://127.0.0.1:8888}"; ROOT_URL="${ROOT_URL%/}"; ROOT_URL="${ROOT_URL%/v1}"; B="$ROOT_URL/v1"
M=deepseek-v4.1-flash
mkdir -p "$D"; cd "$D"
P="$D/battery.phases"
ph() { echo "$(date -u +%FT%TZ) PHASE $*" >> "$P"; }
dump() { curl -s "$ROOT_URL/metrics" > "$D/metrics-$1.txt" 2>/dev/null; }
ph start; dump start
ph gates start; python3 $T/gates.py "$D" > "$D/gates.log" 2>&1; ph gates end rc=$?; dump after-gates
ph qeval start; (cd "$D" && python3 $T/qeval.py run "$ARM" --url $B/chat/completions > "$D/qeval.log" 2>&1); ph qeval end rc=$?; dump after-qeval
ph needles start; python3 $T/long_check.py "$ROOT_URL" 20000 131073 300000 300000 300000 > "$D/needles.log" 2>&1; ph needles end rc=$?; dump after-needles
ph c1-run1 start
python3 $T/v41bench_tf.py --base $B --model $M --label "$ARM-c1" --out "$D" --levels 1 --prefill "" --notes "Lane B R2 $ARM (SGLang knapcio v2.4 block, sparks1-4). C1 run 1 of 2." > "$D/c1.log" 2>&1
ph c1-run1 end rc=$?; dump after-c1-run1
ph main start
python3 $T/v41bench_tf.py --base $B --model $M --label "$ARM-main" --out "$D" --levels 1,2,3,4,5,6 --prefill 2000,8000,32000,64000 --notes "Lane B R2 $ARM. C1-C6 (C1 run 2 of 2) + prefill ladder (64000 target = 93K prompt)." > "$D/main.log" 2>&1
ph main end rc=$?; dump after-main
ph prefill-reps start; python3 $T/prefill_reps.py "$ROOT_URL" "$M" 3 2000 8000 32000 64000 93335 > "$D/prefill_reps.log" 2>&1; ph prefill-reps end rc=$?; dump after-prefill-reps
ph mixed start; python3 $T/mixed_probe.py "$ROOT_URL" "$M" 93000 > "$D/mixed.log" 2>&1; ph mixed end rc=$?; dump after-mixed
ph done
