#!/bin/bash
# final_box.sh -- serve the rule winner (W8k3na) as WIN, confirmation battery, then needles. Leaves it serving.
set -u
WORK=${WORK:-$HOME/flare-prep-glm}   # campaign dir on the rank-0 node (logs, results, tools/)
H=$WORK; exec >> $H/final.log 2>&1
echo "FINAL BOX START $(date)"
bash $H/arm_chain.sh W8 WIN 8 --mtp-draft-tokens 3 --no-mtp-adaptive-verification
D=$H/results/WIN; free -g | awk 'NR==2{print "pre-needle MemAvailable GiB", $7}'
python3 $H/tools/glm_needles.py http://127.0.0.1:8003/v1 32000 128000 300000 600000 > $D/needles.log 2>&1
echo "needles rc=$? $(date)"; cat $D/needles.log | cut -c1-300
free -g | awk 'NR==2{print "post-needle MemAvailable GiB", $7}'
curl -s -m 10 http://127.0.0.1:8003/metrics > $D/metrics-after-needles.txt
echo "FINAL BOX END $(date)"
