#!/bin/bash
# Contention A/B: does a concurrent 93K prefill on the sibling lane (same CRS804 switch) slow this lane's prefill?
# B = SGLang lane, spark1 :8888 ; C = vLLM lane, spark5 :8002. Lanes are on disjoint node sets 1-4 / 5-8.
set -u
S1=milo@spark1; S5=milo@spark5
BT=~/bench-r2/tools/prefill_reps.py; CT=~/ladder/prefill_reps.py
BM=deepseek-v4.1-flash; CM=deepseek-v4.1-flash-spark-vllm
BD=~/bench-r2/contention; CD=~/bench/ladder/contention
L=93335
log(){ echo "[$(date '+%F %T')] $*" | tee -a run.log; }
# phase dir, lane, reps -> runs prefill_reps on that lane, timestamps each line
run(){ local ph=$1 lane=$2 reps=$3; local host tool model dir port
  if [ $lane = B ]; then host=$S1 tool=$BT model=$BM dir=$BD port=8888; else host=$S5 tool=$CT model=$CM dir=$CD port=8002; fi
  ssh -o ConnectTimeout=15 $host "mkdir -p $dir/$ph && cd $dir/$ph && python3 $tool http://127.0.0.1:$port $model $reps $L 2>&1 | while IFS= read -r l; do echo \"\$(date +%T) \$l\"; done" > $ph-$lane.log 2>&1
  log "$ph $lane done: $(tail -1 $ph-$lane.log)"
}
case ${1:-all} in
quiet)
  log "PHASE quiet: B alone x3, then C alone x3"
  run quiet B 3; run quiet C 3 ;;
contB)
  log "PHASE contB: C prefilling continuously (x9), B measured x3 inside that window"
  run contB C 9 & sleep 12; run contB B 3; wait ;;
contC)
  log "PHASE contC: B prefilling continuously (x9), C measured x3 inside that window"
  run contC B 9 & sleep 12; run contC C 3; wait ;;
quiet2)
  log "PHASE quiet2: recheck B alone x3, C alone x3"
  run quiet2 B 3; run quiet2 C 3 ;;
esac
