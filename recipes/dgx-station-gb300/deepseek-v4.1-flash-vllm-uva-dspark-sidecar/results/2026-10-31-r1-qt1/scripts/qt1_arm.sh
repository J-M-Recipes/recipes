#!/usr/bin/env bash
# qt1_arm.sh h2ref <outdir> — Card QT1 P7 captures on h2-REF, called by r1_runner.sh P4 while h2-REF serves :30006.
set -uo pipefail
[[ ${1:-} == h2ref ]] || { echo "usage: qt1_arm.sh h2ref <outdir>"; exit 2; }
OUT=${2:?outdir}; Q=$HOME/dsv41/qt1; mkdir -p "$OUT"
docker ps --format '{{.Names}}' | grep -qx dsv41-vllm-1017-h2-REF || { echo "REFUSE: h2-REF not running"; exit 3; }
cd "$Q" || exit 4
for rep in a b; do
  t0=$(date +%s)
  python3 qt1_tf.py capture --ref tf-G1-A.jsonl --out "$OUT/tf20-h2${rep}.jsonl" --conc 1 > "$OUT/tf-h2${rep}.log" 2>&1
  echo "capture h2${rep} rc=$? $(( $(date +%s) - t0 )) s $(wc -l < "$OUT/tf20-h2${rep}.jsonl") docs"
done
sha256sum "$OUT"/tf20-h2*.jsonl
