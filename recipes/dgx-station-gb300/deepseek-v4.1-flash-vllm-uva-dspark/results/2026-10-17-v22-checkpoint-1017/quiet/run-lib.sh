#!/usr/bin/env bash
# usage: run-lib.sh NAME [bench args...]  -> NAME.json, NAME.stdout.txt, NAME.window.txt, NAME.engine-load.txt, NAME.done
set -uo pipefail
cd "$(dirname "$0")"; n=$1; shift
a=$(date -u +%FT%TZ)
LLM_BENCH_NO_UPDATE_CHECK=1 python3 <bench>/llm_decode_bench.py \
  --display-mode plain --host 127.0.0.1 --port 8000 --model current "$@" --output "$n.json" > "$n.stdout.txt" 2>&1
rc=$?; b=$(date -u +%FT%TZ); echo "$a $b" > "$n.window.txt"
docker logs --since "$a" --until "$b" dsv41-recipe-v22 2>&1 | grep 'Running: ' > "$n.engine-load.txt"
foreign=$(docker logs --since "$a" --until "$b" dsv41-recipe-v22 2>&1 | grep 'POST /v1' | grep -vc '127\.0\.0\.1')
maxr=$(grep -o 'Running: [0-9]*' "$n.engine-load.txt" | awk '{print $2}' | sort -n | tail -1)
echo "rc=$rc max_running=$maxr foreign_posts=$foreign window=$a..$b" > "$n.done"
