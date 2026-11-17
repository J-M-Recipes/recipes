#!/usr/bin/env bash
# post_arm.sh — post-battery functional / determinism suite for a running lane.
# This is the script that produced results/post/<lane>/ on 2026-11-16, with the campaign's absolute paths made
# relative to the recipe. Run it after the battery, against the lane that is up.
#
#   BASE_URL=http://spark5:8002 bash scripts/post_arm.sh out/post-r7
#
# Env: BASE_URL (default http://127.0.0.1:8002), MODEL (served name), OUT (default ./post).
# Needs: the recipe checked out (gates.py, greedy_probe.py, so_probe.py, the DSML live probe), curl, python3.
set -u
D=$(cd "$(dirname "$0")" && pwd)
REC=$(cd "$D/.." && pwd)
BASE_URL=${BASE_URL:-http://127.0.0.1:8002}
MODEL=${MODEL:-deepseek-v4.1-flash-spark-vllm}
ROOT=${BASE_URL%/}; ROOT=${ROOT%/v1}
OUT=${OUT:-./post}
LIVE=$REC/results/2026-11-16-r7-candidate60/tests/live_probe.py
mkdir -p "$OUT"; cd "$OUT"

echo "== gates"
BASE_URL=$ROOT/v1 python3 "$D/gates.py" "$PWD" > gates.log 2>&1; tail -3 gates.log

echo "== greedy x2 (same boot)"
python3 "$D/greedy_probe.py" "$ROOT" greedy-1.json >/dev/null && python3 "$D/greedy_probe.py" "$ROOT" greedy-2.json >/dev/null
python3 - <<'PY'
import json
a = json.load(open('greedy-1.json'))['outputs']; b = json.load(open('greedy-2.json'))['outputs']
print('greedy same-boot identical', sum(x == y for x, y in zip(a, b)), '/', len(a))
PY

echo "== greedy under load (15 background streams, max_tokens 256, T=0.8)"
python3 - "$ROOT" "$MODEL" <<'PY'
import json, sys, threading, urllib.request, time
root, model = sys.argv[1], sys.argv[2]
stop = False
def bg(i):
    while not stop:
        try:
            body = json.dumps({"model": model, "prompt": f"Story {i}: once upon a time", "max_tokens": 256,
                               "temperature": 0.8}).encode()
            urllib.request.urlopen(urllib.request.Request(root + '/v1/completions', body,
                                  {"Content-Type": "application/json"}), timeout=600).read()
        except Exception:
            pass
ts = [threading.Thread(target=bg, args=(i,), daemon=True) for i in range(15)]
[t.start() for t in ts]; time.sleep(5)
src = json.load(open('greedy-1.json')); outs = []
for p in src['prompts']:
    body = json.dumps({"model": model, "prompt": p, "max_tokens": 128, "temperature": 0, "seed": 1}).encode()
    d = json.load(urllib.request.urlopen(urllib.request.Request(root + '/v1/completions', body,
                                        {"Content-Type": "application/json"}), timeout=600))
    outs.append(d['choices'][0]['text'])
stop = True
json.dump({"prompts": src['prompts'], "outputs": outs}, open('greedy-load.json', 'w'))
print('greedy under-load identical to idle', sum(x == y for x, y in zip(src['outputs'], outs)), '/', len(outs))
PY

echo "== so_probe (spark-ds41f#1194 guard, 32 concurrent structured-output requests, 2 waves)"
python3 "$D/so_probe.py" "$ROOT" "$MODEL" 32 2 | tail -1

echo "== DSML live probe"
python3 "$LIVE" "$ROOT/v1" "$MODEL" > dsml-live.log 2>&1; tail -6 dsml-live.log
echo "POST_DONE $(date)"
