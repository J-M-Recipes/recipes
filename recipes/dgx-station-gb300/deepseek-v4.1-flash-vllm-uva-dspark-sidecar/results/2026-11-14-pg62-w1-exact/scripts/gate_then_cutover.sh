#!/bin/bash
# Wait for the long-context shadow (both fixes) to end clean, then cut over. Any gate failure: no cutover.
A=$HOME/dsv41/allround-20261112; L=$A/shadow-lc2/run.log; G=$A/milo-takeover/promote/gate.log
exec >>"$G" 2>&1
echo "$(date '+%F %T') GATE waiting on shadow-lc2"
for i in $(seq 1 240); do grep -q "SHADOW END\|ABORT" $L && break; sleep 15; done
grep -q "SHADOW END ok" $L || { echo "$(date '+%F %T') GATE FAIL: shadow-lc2 did not end ok"; exit 1; }
python3 - <<'PY' || { echo "$(date '+%F %T') GATE FAIL: mismatches or missing coverage"; exit 1; }
import json, sys
r = json.load(open('$HOME/dsv41/allround-20261112/shadow-lc2/dump-1/stats.json'))
bp = r['totals_by_path']; print('BY_PATH', bp)
ok = all(bp[p]['calls'] > 1000 and bp[p]['calls_any'] == 0 for p in ('fp8', 'incumbent'))
ov = r['devices']['cuda:0'].get('overflow_by_path', {}); print('overflow', ov)
sys.exit(0 if ok and not any(ov.values()) else 1)
PY
grep -q "LC errors 0" $A/shadow-lc2/lc.log || { echo "$(date '+%F %T') GATE FAIL: lc_drive errors: $(tail -1 $A/shadow-lc2/lc.log)"; exit 1; }
grep -c prompt_tokens $A/shadow-lc2/lc.log
[ -z "$(docker ps -q)" ] || { echo "$(date '+%F %T') GATE FAIL: containers running"; exit 1; }
[ "$(sha256sum < $A/serve-bisect/tree/fastpath/w1attn/w1_attn_ext.so | cut -c1-16)" = 7201643d5575c187 ] || echo "note: test tree incumbent not restored"
echo "$(date '+%F %T') GATE PASS -> cutover"
exec bash $HOME/dsv41/garner/promote/cutover-pg61-to-pg62.sh
