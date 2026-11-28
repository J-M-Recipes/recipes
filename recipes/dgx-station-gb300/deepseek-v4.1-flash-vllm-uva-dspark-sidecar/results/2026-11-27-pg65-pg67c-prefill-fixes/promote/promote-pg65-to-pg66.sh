#!/bin/bash
# Promote Pg66 = Pg65 + Garner fastpath components w8oproj + lowb_aux (card F1/F1b, Flare-transfer round 2026-11-24).
#   w8oproj: attention wo_a + MXFP8 quantize in one kernel (T 1..8; window layers bitwise, sparse/drafter 2-part split-K).
#   lowb_aux: attention side bf16 GEMMs (kv_score, indexer weights_proj/wk) on deterministic LLBf16 at small T.
#   Both change summation order only. F1+F1b: C1 step -0.149/-0.198 ms vs control, decode logprob drift inside the
#   control envelope, top1 100 %, gauntlet 12/12, C8/C32 flat. Receipts ~/dsv41/flarex/results-F1{,b}/ANALYSIS.txt.
# Only change vs pg65-work: DS41_FASTPATH_COMPONENTS_OVERRIDE (+2 names) and the overlay-dir path line.
# Kill switch without rebuild: drop the two names from pg66-work/policy.env and relaunch; or rollback-pg66-to-pg65.sh.
set -uo pipefail
GR=~/dsv41/garner; W=$GR/pg66-work; B=$GR/pg65-work
MODEL_DIR=/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170
LO1=~/dsv41/m3/LANE-OWNER.txt; LO2=~/dsv41/m3/logs/LANE-OWNER.txt
LOG=$GR/promote/promote-pg66-$(date +%Y%m%d-%H%M).log; exec >>"$LOG" 2>&1
ts(){ date "+%F %T %Z"; }; lo(){ echo "- $(ts) $*" >> $LO1; echo "- $(ts) $*" >> $LO2; }; log(){ echo "$(ts) [pg66] $*"; }
restore65(){ env WORK=$W DS41_HOME=$GR/home bash $GR/launcher/launch-pg61.sh down || true
  sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; sleep 45
  env WORK=$B MODEL_DIR=$MODEL_DIR DS41_HOME=$GR/home bash $GR/launcher/launch-pg61.sh up && lo "Pg65 restored after failed Pg66 promote" || lo "Pg65 RESTORE FAILED; lane DARK; see $LOG"; }
fail(){ log "FAIL: $*"; lo "PROMOTE Pg66 FAILED ($*); restoring Pg65"; [ "${UP:-0}" = 1 ] && restore65; rm -f ~/dsv41/m3/logs/MAINTENANCE; echo "OP promote END rc=1"; exit 1; }
log "start"
COMPS=$(grep '^DS41_FASTPATH_COMPONENTS_OVERRIDE=' $B/policy.env | cut -d= -f2-)
NEW=$(python3 -c "import sys;a=sys.argv[1].split(',');a.insert(a.index('topp_fast')+1,'w8oproj');a.insert(a.index('tail_proj_side')+1,'lowb_aux');print(','.join(a))" "$COMPS")
if [ ! -e $W/BUILD-SHA256SUMS ]; then
  mkdir -p $W && cp -a $B/tree $B/overlay-vllm $B/sidecar_watchdog_p3.py $B/policy.env $W/ || fail "copy"
  chmod -R u+w $W
  sed -i "s#^DS41_OVERLAY_58132_DIR=.*#DS41_OVERLAY_58132_DIR=$W/overlay-vllm#" $W/policy.env
  sed -i "s#^DS41_FASTPATH_COMPONENTS_OVERRIDE=.*#DS41_FASTPATH_COMPONENTS_OVERRIDE=$NEW#" $W/policy.env
  ( cd $W && find tree/fastpath tree/serve tree/config overlay-vllm sidecar_watchdog_p3.py policy.env -type f ! -name '*.pyc' -print0 | sort -z | xargs -0 sha256sum ) > $W/BUILD-SHA256SUMS
  chmod -R a-w $W/tree $W/overlay-vllm; chmod u+w $W
fi
nd=$(diff -rq $B/tree $W/tree | grep -v __pycache__ | wc -l); [ "$nd" = 0 ] || fail "tree drift $nd"
no=$(diff -rq $B/overlay-vllm $W/overlay-vllm | grep -v __pycache__ | wc -l); [ "$no" = 0 ] || fail "overlay drift $no"
np=$(diff $B/policy.env $W/policy.env | grep -c '^[<>]'); [ "$np" = 4 ] || fail "policy drift $np (want 4)"
grep -q "w8oproj" $W/policy.env && grep -q "lowb_aux" $W/policy.env || fail "components missing"
log "pg66-work ready: tree=$nd overlay=$no policy lines=$np"
lo "PROMOTE Pg66 starting (Milo; James 11/24 'test and see if we can improve our recipe'; F1+F1b passed): Pg65 + w8oproj + lowb_aux; Pg65 kept as rollback (promote/rollback-pg66-to-pg65.sh)"
echo "promotion window Pg65 -> Pg66 (Milo, w8oproj + lowb_aux)" > ~/dsv41/m3/logs/MAINTENANCE
env WORK=$B DS41_HOME=$GR/home bash $GR/launcher/launch-pg61.sh down || fail "down"
UP=1
sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; sleep 45
env WORK=$W MODEL_DIR=$MODEL_DIR DS41_HOME=$GR/home bash $GR/launcher/launch-pg61.sh up || fail "up"
e=$(docker inspect -f '{{json .Config.Env}}' ds41)
echo "$e" | grep -q '"VLLM_PREFILL_YIELD=short"' || fail "env yield (Pg65 base lost)"
docker logs ds41 > $W/boot.log 2>&1
tr '\r' '\n' < $W/boot.log > $W/boot.lines  # no grep -q in pipes: pipefail + SIGPIPE on tr reads as a miss (22:35 false FAIL)
for c in w8oproj lowb_aux; do grep -aq "DS41_FASTPATH ACTIVE $c" $W/boot.lines || fail "$c not ACTIVE"; done
nfb=$(grep -acE 'falling back|stock:|DS41_FASTPATH armed' $W/boot.lines); log "fallback lines: $nfb"
grep -aq "w8oproj: engaged at T=8" $W/boot.lines || fail "w8oproj not engaged at T=8"
out=$(curl -sf -m 60 http://127.0.0.1:30006/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"deepseek-v4.1-flash","temperature":0,"max_tokens":8,"messages":[{"role":"user","content":"Reply with exactly: OK"}],"chat_template_kwargs":{"thinking":false}}') || fail "smoke"
echo "$out" | grep -q '"content":"OK' || fail "smoke content: $out"
sleep 20; grep -q '"trip": null' $W/wdp3.status.json || fail "watchdog"
mv ~/dsv41/m3/logs/MAINTENANCE ~/dsv41/m3/logs/MAINTENANCE.stale-pg66-$(date +%Y%m%d-%H%M)
lo "PROMOTE Pg66 DONE: serving :30006 (garner/pg66-work = pg65-work + w8oproj + lowb_aux); p3 watchdog ENFORCE; rollback promote/rollback-pg66-to-pg65.sh; log $LOG"
log "DONE"; echo "OP promote END rc=0"
