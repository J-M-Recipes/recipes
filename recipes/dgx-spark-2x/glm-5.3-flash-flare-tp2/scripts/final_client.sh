#!/bin/bash
# final_m4.sh TAG -- client-side winner checks against <rank0>:8003: GSM8K-200 (lil-bench, same as GB300 Flare post) + Hermes 20-hop x3.
set -u
TAG=$1; O=${OUT:-$PWD/results}/$TAG-final; HOST=${HOST:?set HOST=<rank-0 host>}; LIL=${LIL_BENCH:?path to llm-inference-bench v0.7.7 checkout with run.sh}; mkdir -p $O
echo "FINAL START $(date)" | tee -a $O/final.log
$LIL/run.sh --host $HOST --port 8003 --model GLM-5.3-Flash --test-profile gsm8k --profile-runs 200 --output $O/t2-gsm8k200.json > $O/t2-gsm8k200.stdout.txt 2>&1
echo "gsm8k rc=$? $(date)" | tee -a $O/final.log
cd $(dirname $0)/hermes-bench && HERMES_HOME=${HERMES_HOME:?Hermes home with a custom provider flareglm -> http://$HOST:8003/v1} python3 gauntlet_reps.py flare-glm-$TAG 'custom[flareglm]' GLM-5.3-Flash 3 > $O/gauntlet.log 2>&1
echo "gauntlet rc=$? $(date) $(tail -1 $O/gauntlet.log)" | tee -a $O/final.log
echo "FINAL END $(date)" | tee -a $O/final.log
