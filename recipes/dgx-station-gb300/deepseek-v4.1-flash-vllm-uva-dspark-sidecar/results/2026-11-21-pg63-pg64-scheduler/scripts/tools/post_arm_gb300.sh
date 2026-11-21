#!/bin/bash
# GB300 post-battery suite for one arm. usage: post_arm_gb300.sh LABEL
set -u
B=~/hermes/research/dsv41-gb300-owen-transfer-20261121/cards
R=~/hermes/jm-recipes/recipes/recipes/dgx-spark-4x/deepseek-v4.1-flash-vllm-tp4-sparknet/scripts
SW=~/hermes/work/spark-recipe-r7-2026-11-16
L=$1; ROOT=http://gb300:30006; export MODEL=dsv41-flash-uva
O=$B/receipts/post/$L; mkdir -p $O; cd $O
echo "== mixed_trace (short chat, thinking on, 32 tok) x4"
for a in 6.6 7.5 8.2 9.0; do python3 ~/hermes/research/dsv41-gb300-owen-transfer-20261121/cards/tools/mixed_trace.py $ROOT deepseek-v4.1-flash 413000 $a trace-s-$a.json 32 on; done > trace-short.log 2>&1; cat trace-short.log
echo "== decode during 413K prefill (thinking off, 400 tok) x3"
for a in 2.0 4.0 6.0; do python3 ~/hermes/research/dsv41-gb300-owen-transfer-20261121/cards/tools/mixed_trace.py $ROOT deepseek-v4.1-flash 413000 $a trace-d-$a.json 400 off; done > trace-decode.log 2>&1; cat trace-decode.log
echo "== decode during 1M prefill (thinking off, 400 tok) x1"
python3 ~/hermes/research/dsv41-gb300-owen-transfer-20261121/cards/tools/mixed_trace.py $ROOT deepseek-v4.1-flash 1000000 4.0 trace-m1m-4.0.json 400 off > trace-1m.log 2>&1; cat trace-1m.log
echo "== C32 stress (v41bench levels 32, 1 rep)"
( cd ~/hermes/work/dsv41-5way-2026-10-26 && DSV41_ENCODING_DIR=tools/encoding python3 tools/v41bench_tf.py --base $ROOT/v1 --model deepseek-v4.1-flash --label $L-c32 --levels 32 --prefill '' ) > c32.log 2>&1; tail -4 c32.log
echo "== gates"; BASE_URL=$ROOT/v1 python3 $R/gates.py $O > gates.log 2>&1; tail -3 gates.log
echo "== greedy x2"; python3 $R/greedy_probe.py $ROOT greedy-1.json >/dev/null 2>&1 && python3 $R/greedy_probe.py $ROOT greedy-2.json >/dev/null 2>&1
python3 - <<'PY'
import json
a=json.load(open('greedy-1.json'))['outputs']; b=json.load(open('greedy-2.json'))['outputs']
print('greedy same-boot identical', sum(x==y for x,y in zip(a,b)), '/', len(a))
PY
echo "== so_probe"; python3 $SW/lanes/so_probe.py $ROOT deepseek-v4.1-flash 32 2 > so.log 2>&1; tail -1 so.log
echo "== dsml live probe"; python3 $SW/results/R4-dsml-port/tests/live_probe.py $ROOT/v1 deepseek-v4.1-flash > dsml-live.log 2>&1; tail -3 dsml-live.log
curl -sf -m 5 $ROOT/v1/models >/dev/null && echo "ALIVE after suite" || echo "SERVER DOWN after suite"
echo "POST_DONE $L $(date)"
