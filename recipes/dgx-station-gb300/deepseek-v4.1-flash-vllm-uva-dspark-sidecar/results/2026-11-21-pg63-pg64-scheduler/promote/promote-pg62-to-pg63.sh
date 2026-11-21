#!/bin/bash
# Promote Pg63 = Pg62 + long-prefill cadence (xfer-20261121 card X4: ov/cadence3 scheduler overlay, env-gated
# VLLM_PREFILL_CADENCE=3 VLLM_PREFILL_CADENCE_MIN_REMAINING=8192). Everything else byte-identical to pg62-work.
# Steps: build frozen pg63-work -> stop current -> settle -> launch-pg61 up on pg63-work -> checks. On any failure: Pg62 back up.
set -uo pipefail
GR=~/dsv41/garner; XF=~/dsv41/xfer-20261121; W=$GR/pg63-work
MODEL_DIR=/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170
LO1=~/dsv41/m3/LANE-OWNER.txt; LO2=~/dsv41/m3/logs/LANE-OWNER.txt
LOG=$GR/promote/promote-pg63-$(date +%Y%m%d-%H%M).log; exec >>"$LOG" 2>&1
ts(){ date "+%F %T %Z"; }; lo(){ echo "- $(ts) $*" >> $LO1; echo "- $(ts) $*" >> $LO2; }; log(){ echo "$(ts) [pg63] $*"; }
fail(){ log "FAIL: $*"; lo "PROMOTE Pg63 FAILED ($*); restoring Pg62"; bash $XF/xfer_arm.sh restore; exit 1; }
log "start"
if [ ! -e $W/BUILD-SHA256SUMS ]; then
  mkdir -p $W && cp -a $GR/pg62-work/tree $GR/pg62-work/overlay-vllm $GR/pg62-work/sidecar_watchdog_p3.py $GR/pg62-work/policy.env $W/ || fail "copy"
  chmod -R u+w $W
  cp $XF/serve.sh.arm $W/tree/serve/serve.sh || fail "serve.sh"
  mkdir -p $W/overlay-vllm/v1/core/sched && cp $XF/ov/cadence3/v1/core/sched/scheduler.py $W/overlay-vllm/v1/core/sched/scheduler.py || fail "overlay"
  sed -i "s#^DS41_OVERLAY_58132_DIR=.*#DS41_OVERLAY_58132_DIR=$W/overlay-vllm#" $W/policy.env
  echo "DS41_ARM_ENV=VLLM_PREFILL_CADENCE=3 VLLM_PREFILL_CADENCE_MIN_REMAINING=8192" >> $W/policy.env
  ( cd $W && find tree/fastpath tree/serve tree/config overlay-vllm sidecar_watchdog_p3.py policy.env -type f ! -name '*.pyc' -print0 | sort -z | xargs -0 sha256sum ) > $W/BUILD-SHA256SUMS
  chmod -R a-w $W/tree $W/overlay-vllm; chmod u+w $W
fi
nd=$(diff -rq $GR/pg62-work/tree $W/tree | grep -v __pycache__ | wc -l); [ "$nd" = 1 ] || fail "tree drift $nd"
no=$(diff -rq $GR/pg62-work/overlay-vllm $W/overlay-vllm | grep -v __pycache__ | wc -l); [ "$no" = 1 ] || fail "overlay drift $no"
log "pg63-work ready: tree diffs=$nd (serve.sh) overlay diffs=$no (scheduler.py)"
lo "PROMOTE Pg63 starting (Milo; James 11/20 'if it wins, it lives as a full-time resident'; card X4 passed all pre-registered bars on 2 boots): Pg62 + long-prefill cadence; Pg62 kept as rollback (promote/rollback-pg63-to-pg62.sh)"
echo "promotion window Pg62 -> Pg63 (Milo)" > ~/dsv41/m3/logs/MAINTENANCE
w=$(ps -eo args | awk 'index($0,"sidecar_watchdog_p3.py --server ds41") && $1 ~ /python3/ {print $2}' | head -1); w=${w:+$(dirname $w)}; w=${w:-$GR/pg62-work}
env WORK=$w DS41_HOME=$GR/home bash $GR/launcher/launch-pg61.sh down || fail "down"
sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; sleep 45
env WORK=$W MODEL_DIR=$MODEL_DIR DS41_HOME=$GR/home bash $GR/launcher/launch-pg61.sh up || fail "up"
docker inspect -f '{{json .Config.Env}}' ds41 | grep -q '"VLLM_PREFILL_CADENCE=3"' || fail "env cadence"
docker inspect -f '{{json .Config.Env}}' ds41 | grep -q '"VLLM_PREFILL_CADENCE_MIN_REMAINING=8192"' || fail "env min"
docker inspect -f '{{json .Args}}' ds41 | grep -q '"--long-prefill-token-threshold","6144"' || fail "args"
h1=$(sha256sum $XF/ov/cadence3/v1/core/sched/scheduler.py | cut -c1-64); h2=$(docker exec ds41 sha256sum /usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py | cut -c1-64)
[ "$h1" = "$h2" ] || fail "overlay not mounted"
out=$(curl -sf -m 60 http://127.0.0.1:30006/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"deepseek-v4.1-flash","temperature":0,"max_tokens":8,"messages":[{"role":"user","content":"Reply with exactly: OK"}],"chat_template_kwargs":{"thinking":false}}') || fail "smoke"
echo "$out" | grep -q '"content":"OK' || fail "smoke content: $out"
# (no `| grep -q` here: with pipefail, grep -q's early exit SIGPIPEs docker logs and fails the pipeline — the 09:15 false FAIL)
nen=$(docker logs ds41 2>&1 | grep -ac "xfer prefill cadence enabled: N=3 min_remaining=8192"); [ "${nen:-0}" -ge 1 ] || fail "cadence not enabled in engine log"
sleep 20; grep -q '"trip": null' $W/wdp3.status.json || fail "watchdog"
mv ~/dsv41/m3/logs/MAINTENANCE ~/dsv41/m3/logs/MAINTENANCE.stale-pg63-$(date +%Y%m%d-%H%M)
lo "PROMOTE Pg63 DONE: serving :30006 (garner/pg63-work = pg62-work + long-prefill cadence N=3 min_remaining 8192); p3 watchdog ENFORCE; rollback promote/rollback-pg63-to-pg62.sh; log $LOG"
log "DONE"
