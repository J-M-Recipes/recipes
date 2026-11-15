#!/bin/bash
# Published copy (2026-11-15) of the reference-box script that ran the Pg61 -> Pg62 cutover at 2026-11-14 23:05:13 CST
# (log: results/2026-11-14-pg62-w1-exact/raw/promote/live/cutover-pg62-20261114-2305.log). Scrubbed: box home path -> $HOME,
# GR/A/MODEL_DIR overridable from the environment; nothing else changed. Box-only dependencies it still names:
# $HOME/dsv41/decode/w0/w0_drive.py and bench/gt_mix_drive.py (jmeadlock/gb300-dsv41-decode), the stock reference
# capture $A/serve-quality-a2 (greedy oracle, first pass), the oracle comparer (results/2026-11-14-pg62-w1-exact/
# scripts/compare.py, at $T/bisect/compare.py on the box), and the lane-owner files under $HOME/dsv41/m3.
# It calls the box's launcher copy at $GR/launcher/launch-pg61.sh without DS41_PG62 (the tree was staged by hand).
# Cutover to Pg62 on :30006 (James 2026-11-14 night: "both together tonight plz").
# Pg62 = Pg61 + (a) W1 FP8 compressed-layer decode attention at 1..16 rows (w1_attn_fp8_ext.so de874506, STOCK_ALL,
#        DS41_W1_COMPRESSED_FP8=1, W13 integrate/ops/formats) + (b) incumbent w1_attn_ext.so rebuilt with the unfused
#        sink term (3bca6cac, STOCK_ALL). Tree: garner/pg62-work (pg61-work untouched = rollback).
# On UP: KV gate, first-pass greedy oracle vs serve-quality-a2, one C1 pass. Oracle/KV failure -> down, lane DARK
# (no automatic Pg61 restore; rollback is promote/rollback-pg62-to-pg61.sh on James's word).
set -uo pipefail
GR=${GR:-$HOME/dsv41/garner}; A=${A:-$HOME/dsv41/allround-20261112}; T=$A/milo-takeover
MODEL_DIR=${MODEL_DIR:-/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170}
O=$T/promote/cutover-$(date +%Y%m%d-%H%M); mkdir -p "$O"; LOG=$GR/promote/cutover-pg62-$(date +%Y%m%d-%H%M).log
exec >>"$LOG" 2>&1
ts(){ date "+%F %T %Z"; }; LO=$HOME/dsv41/m3/LANE-OWNER.txt; LO2=$HOME/dsv41/m3/logs/LANE-OWNER.txt
lo(){ for p in $LO $LO2; do [ -e $p ] && echo "- $(ts) $*" >> $p; done; }
pg(){ env WORK=$GR/pg62-work MODEL_DIR=$MODEL_DIR DS41_HOME=$GR/home bash $GR/launcher/launch-pg61.sh "$1"; }
echo "$(ts) [cutover-pg62] start; receipts $O"
other=$(docker ps --format '{{.Names}}'); [[ -n $other ]] && { echo "$(ts) ABORT: containers running: $other"; exit 3; }
ps -eo args | grep -qE "^bash .*(_runner|run_shadow|run_perf)[^ ]*\.sh" && { echo "$(ts) ABORT: a runner is alive"; exit 4; }
( cd $GR/pg62-work && sha256sum -c --quiet BUILD-SHA256SUMS ) || { echo "$(ts) ABORT: pg62-work hash mismatch"; exit 5; }
( cd $GR/pg61-work && sha256sum -c --quiet BUILD-SHA256SUMS ) || { echo "$(ts) ABORT: pg61-work (rollback) hash mismatch"; exit 5; }
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader; df -h / | tail -1; free -g | sed -n 2p
lo "CUTOVER Pg61 -> Pg62 starting (Milo, James: both fixes together tonight); tree garner/pg62-work; log $LOG"
sudo -S -p '' -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; sleep 45
if ! pg up; then rc=$?; echo "$(ts) [cutover-pg62] UP FAILED rc=$rc"; pg down || true
  lo "CUTOVER FAILED rc=$rc at launch-pg61.sh up (pg62-work); lane DARK; MAINTENANCE kept; see $LOG"; exit $rc; fi
echo "$(ts) [cutover-pg62] UP"
docker logs ds41 > "$O/server.log" 2>&1
kv=$(grep -o 'GPU KV cache size: [0-9,]* tokens' "$O/server.log" | tail -1 | tr -dc 0-9)
grep 'DS41_FASTPATH attn' "$O/server.log" | cut -c1-600 | tail -4
fail=""
[ "$kv" = 11696582 ] || fail="KV $kv"
W0_BASE=http://127.0.0.1:30006 W0_MODEL=dsv41-flash-uva timeout 300 python3 $HOME/dsv41/decode/w0/w0_drive.py parity "$O/r0.jsonl" > "$O/r0.log" 2>&1 || fail="$fail oracle-run"
v=$(python3 $T/bisect/compare.py $A/serve-quality-a2 "$O" 1 2>&1 | tail -1); echo "$(ts) ORACLE $v KV=$kv"
echo "$v" | grep -q "r0 text 8/8 lp-exact 8/8" || fail="$fail oracle($v)"
if [ -n "$fail" ]; then echo "$(ts) [cutover-pg62] GATE FAILED: $fail -> down, DARK"; pg down || true
  lo "CUTOVER GATE FAILED ($fail); Pg62 taken down; lane DARK; MAINTENANCE kept; rollback script promote/rollback-pg62-to-pg61.sh not run"; exit 6; fi
W0_BASE=http://127.0.0.1:30006 W0_MODEL=dsv41-flash-uva timeout 300 python3 $HOME/dsv41/decode/bench/gt_mix_drive.py pass --set A --out "$O/c1-A.json" > "$O/c1-A.log" 2>&1
echo "$(ts) C1-A $(python3 -c "import json;a=json.load(open('$O/c1-A.json'))['agg'];print({k:a[k] for k in ('decode_tok_s','step_ms','tokens_per_step','errors') if k in a})" 2>&1)"
grep -q '"trip": null' $GR/pg62-work/wdp3.status.json && echo "$(ts) watchdog trip null" || echo "$(ts) WATCHDOG NOT CLEAN"
ps -eo pid,ppid,comm,args | awk '$3=="python3" && index($0,"sidecar_watchdog_p3.py") && index($0,"--mode enforce")' | cut -c1-200
[ -e $HOME/dsv41/m3/logs/MAINTENANCE ] && mv $HOME/dsv41/m3/logs/MAINTENANCE $HOME/dsv41/m3/logs/MAINTENANCE.stale-allround-$(date +%Y%m%d-%H%M)
lo "CUTOVER done: Pg62 serving :30006 (garner/pg62-work: FP8 W1 attention fast path de874506 + incumbent sink fix 3bca6cac); KV $kv; oracle 8/8 vs stock; p3 watchdog ENFORCE; soak clock restarted; Pg61 tree kept as rollback (promote/rollback-pg62-to-pg61.sh); MAINTENANCE archived; log $LOG"
echo "$(ts) [cutover-pg62] DONE"
