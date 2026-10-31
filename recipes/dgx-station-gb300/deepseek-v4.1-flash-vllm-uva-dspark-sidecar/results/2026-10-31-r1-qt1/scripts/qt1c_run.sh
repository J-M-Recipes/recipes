#!/usr/bin/env bash
# qt1c_run.sh — QT1-CONFIRM arms per PREREG-CONFIRM.md (sha 8ae41fd2). Box loopback, C4, max_tokens 256.
set -uo pipefail
cd "$HOME/dsv41/qt1" || exit 1
O=run-confirm; mkdir -p $O; cp run1/promptids.jsonl $O/; sha256sum PREREG-CONFIRM.md > $O/prereg.sha256
curl -sf -m 3 http://127.0.0.1:30006/v1/models | grep -q dsv41-flash-uva || { echo "REFUSE: lane not bound"; exit 2; }
grep -q '"trip": null' "$HOME/dsv41/garner/pg61-work/wdp3.status.json" || { echo "REFUSE: watchdog"; exit 2; }
for spec in "C1 1.0 310000" "C2 1.0 320000" "K95 0.95 330000" "K105 1.05 340000"; do
  set -- $spec
  t0=$(date +%s)
  python3 qt1_gen.py --arm $1 --temp $2 --seed-base $3 --max-tokens 256 --conc 4 --out $O > $O/gen-$1.log 2>&1
  echo "$(date '+%F %T %Z') arm $1 rc=$? $(( $(date +%s) - t0 )) s" >> $O/arms.log
done
grep -q '"trip": null' "$HOME/dsv41/garner/pg61-work/wdp3.status.json" && echo "watchdog trip null at end" >> $O/arms.log
echo DONE >> $O/arms.log
