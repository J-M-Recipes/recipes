#!/bin/bash
# Promote Pg65 = Pg64 + DSML token provenance (vLLM PR #65019 port onto v0.31.0, merged with our structural-closer rule).
# Pure-Python parser overlay: parser/deepseek_v41.py (replaced) + parser/parser_manager.py + parser/engine/{token_id_scanner,
# incremental_lexer,streaming_parser_engine,parser_engine,parser_engine_config}.py (new). In-image CPU suite 67/67 (2026-11-22).
# Kill switches (env, no rebuild): VLLM_DSV41_DSML_TOKEN_PROVENANCE=0 (old text behaviour), VLLM_DSV41_STRUCTURAL_DSML=0.
# Base = pg64-work (kept as rollback: promote/rollback-pg65-to-pg64.sh).
set -uo pipefail
GR=~/dsv41/garner; P=~/dsv41/dsml-port/overlay/vllm/parser; W=$GR/pg65-work; B=$GR/pg64-work
MODEL_DIR=/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170
LO1=~/dsv41/m3/LANE-OWNER.txt; LO2=~/dsv41/m3/logs/LANE-OWNER.txt
LOG=$GR/promote/promote-pg65-$(date +%Y%m%d-%H%M).log; exec >>"$LOG" 2>&1
ts(){ date "+%F %T %Z"; }; lo(){ echo "- $(ts) $*" >> $LO1; echo "- $(ts) $*" >> $LO2; }; log(){ echo "$(ts) [pg65] $*"; }
restore64(){ env WORK=$W DS41_HOME=$GR/home bash $GR/launcher/launch-pg61.sh down || true
  sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; sleep 45
  env WORK=$B MODEL_DIR=$MODEL_DIR DS41_HOME=$GR/home bash $GR/launcher/launch-pg61.sh up && lo "Pg64 restored after failed Pg65 promote" || lo "Pg64 RESTORE FAILED; lane DARK; see $LOG"; }
fail(){ log "FAIL: $*"; lo "PROMOTE Pg65 FAILED ($*); restoring Pg64"; [ "${UP:-0}" = 1 ] && restore64; rm -f ~/dsv41/m3/logs/MAINTENANCE; exit 1; }
log "start"
if [ ! -e $W/BUILD-SHA256SUMS ]; then
  mkdir -p $W && cp -a $B/tree $B/overlay-vllm $B/sidecar_watchdog_p3.py $B/policy.env $W/ || fail "copy"
  chmod -R u+w $W
  mkdir -p $W/overlay-vllm/parser/engine
  cp $P/deepseek_v41.py.DRAFT $W/overlay-vllm/parser/deepseek_v41.py || fail "cp parser"
  cp $P/parser_manager.py.DRAFT $W/overlay-vllm/parser/parser_manager.py || fail "cp manager"
  for f in token_id_scanner incremental_lexer streaming_parser_engine parser_engine parser_engine_config; do
    cp $P/engine/$f.py.DRAFT $W/overlay-vllm/parser/engine/$f.py || fail "cp $f"; done
  sed -i "s#^DS41_OVERLAY_58132_DIR=.*#DS41_OVERLAY_58132_DIR=$W/overlay-vllm#" $W/policy.env
  ( cd $W && find tree/fastpath tree/serve tree/config overlay-vllm sidecar_watchdog_p3.py policy.env -type f ! -name '*.pyc' -print0 | sort -z | xargs -0 sha256sum ) > $W/BUILD-SHA256SUMS
  chmod -R a-w $W/tree $W/overlay-vllm; chmod u+w $W
fi
nd=$(diff -rq $B/tree $W/tree | grep -v __pycache__ | wc -l); [ "$nd" = 0 ] || fail "tree drift $nd"
no=$(diff -rq $B/overlay-vllm $W/overlay-vllm | grep -v __pycache__ | wc -l); [ "$no" = 3 ] || fail "overlay drift $no (want 3: deepseek_v41.py differs, Only-in engine, Only-in parser_manager.py)"
log "pg65-work ready: tree diffs=$nd overlay diffs=$no"
lo "PROMOTE Pg65 starting (Milo; James 11/22 'yes' to porting the DSML real-token rule): Pg64 + DSML token provenance parser overlay; Pg64 kept as rollback (promote/rollback-pg65-to-pg64.sh)"
echo "promotion window Pg64 -> Pg65 (Milo, DSML provenance parser)" > ~/dsv41/m3/logs/MAINTENANCE
env WORK=$B DS41_HOME=$GR/home bash $GR/launcher/launch-pg61.sh down || fail "down"
UP=1
sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; sleep 45
env WORK=$W MODEL_DIR=$MODEL_DIR DS41_HOME=$GR/home bash $GR/launcher/launch-pg61.sh up || fail "up"
docker inspect -f '{{json .Config.Env}}' ds41 | grep -q '"VLLM_PREFILL_YIELD=short"' || fail "env yield (Pg64 base lost)"
for f in deepseek_v41.py parser_manager.py engine/token_id_scanner.py engine/incremental_lexer.py engine/streaming_parser_engine.py engine/parser_engine.py engine/parser_engine_config.py; do
  h1=$(sha256sum $W/overlay-vllm/parser/$f | cut -c1-64); h2=$(docker exec ds41 sha256sum /usr/local/lib/python3.12/dist-packages/vllm/parser/$f | cut -c1-64)
  [ "$h1" = "$h2" ] || fail "overlay not mounted: $f"; done
out=$(curl -sf -m 60 http://127.0.0.1:30006/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"deepseek-v4.1-flash","temperature":0,"max_tokens":8,"messages":[{"role":"user","content":"Reply with exactly: OK"}],"chat_template_kwargs":{"thinking":false}}') || fail "smoke"
echo "$out" | grep -q '"content":"OK' || fail "smoke content: $out"
docker logs ds41 > $W/boot.log 2>&1; grep -a -i -E "provenance|DSML" $W/boot.log | tail -5
nprov=$(grep -a -c -i "provenance active" $W/boot.log); log "provenance-active log lines: $nprov"
sleep 20; grep -q '"trip": null' $W/wdp3.status.json || fail "watchdog"
mv ~/dsv41/m3/logs/MAINTENANCE ~/dsv41/m3/logs/MAINTENANCE.stale-pg65-$(date +%Y%m%d-%H%M)
lo "PROMOTE Pg65 DONE: serving :30006 (garner/pg65-work = pg64-work + DSML token-provenance parser); p3 watchdog ENFORCE; rollback promote/rollback-pg65-to-pg64.sh; log $LOG"
log "DONE"
