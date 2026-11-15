#!/usr/bin/env bash
# Milo 2026-11-14: shadow arm per E-RUNBOOK steps 1-10 (+ optional full w0 pass -> dump-1). Ends DARK.
set -u
R=$HOME/dsv41/allround-20261112; T=$R/milo-takeover; W=$R/serve-bisect; GR=$HOME/dsv41/garner
DRV=$HOME/dsv41/decode/w0/w0_drive.py
MODEL=/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170
IMAGE=sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3
FP=$W/tree/fastpath/w1attn; SH=$W/home/state/prof/shadow; O=$R/shadow-lc1
mkdir -p "$O"; exec >>"$O/run.log" 2>&1
say(){ echo "$(date '+%F %T') $*"; }
lane(){ env WORK="$W" DS41_HOME="$W/home" MODEL_DIR="$MODEL" IMAGE="$IMAGE" DS41_IMAGE="$IMAGE" timeout "$2" bash "$GR/launcher/launch-pg61.sh" "$1"; }
down(){ lane down 240 >>"$O/down.log" 2>&1; docker ps --format '{{.Names}}' | grep -qxE 'ds41|ds41-sidecar' && docker stop -t 10 ds41 ds41-sidecar; say "DOWN running=[$(docker ps --format '{{.Names}}' | tr '\n' ' ')]"; }
rollback(){ cp "$O/integrate.py.bisect" "$FP/integrate.py"; cp "$O/fp8_mask.json.bisect" "$FP/fp8_mask.json"; cp "$O/policy.env.bisect" "$W/policy.env"; say "ROLLED BACK tree (so left as stockall)"; }
sentinel(){ rm -f "$SH/$1.done" "$SH/$1.error"; touch "$SH/$1"; for i in $(seq "$2"); do [ -e "$SH/$1.done" ] || [ -e "$SH/$1.error" ] && break; sleep 1; done; say "$1: $(cat "$SH/$1.done" "$SH/$1.error" "$SH/NOT_READY" 2>/dev/null | head -c 300)"; [ -e "$SH/$1.done" ]; }
stats(){ python3 - "$1" <<'PY'
import json, sys
r = json.load(open(sys.argv[1])); print('TOTALS', r['totals'])
al = r['devices']['cuda:0']['sites']
import collections
w = collections.Counter()
for x in al: w[x['width']] += x['calls']
print('calls_by_width', dict(sorted(w.items())))
ss = [s for s in al if s['calls_with_any_mismatch']]
print('sites_with_mismatch', len(ss))
for s in ss[:40]:
    print({k: s.get(k) for k in ('layer','width','calls','calls_with_any_mismatch','calls_with_valid_row_mismatch','mismatching_data_bytes','mismatching_scale_bytes','first_mismatch_seq','row_mismatch_calls','snapshot_file')})
PY
}
say "SHADOW START"
[ -z "$(docker ps -q)" ] || { say "ABORT containers running"; exit 1; }
pgrep -f "bisect_runner.sh|spec_runner" >/dev/null && { say "ABORT runner alive"; exit 1; }
grep -qi allround $HOME/dsv41/m3/logs/MAINTENANCE || { say "ABORT no maintenance flag"; exit 1; }
exec 9>>"$R/gpu-owner.lock"; flock -n 9 || { say "ABORT lock"; exit 1; }
cd "$T/results/E" && sha256sum *.DRAFT.py > "$O/SHIPPED.sha256"
cp -n "$FP/integrate.py" "$O/integrate.py.bisect"; cp -n "$FP/fp8_mask.json" "$O/fp8_mask.json.bisect"; cp -n "$W/policy.env" "$O/policy.env.bisect"
cp "$T/results/E/integrate_shadow.DRAFT.py" "$FP/integrate.py" && python3 -m py_compile "$FP/integrate.py" || { say "ABORT compile"; rollback; exit 1; }
cp "$FP/w1_attn_fp8_ext.so" "$O/w1_attn_fp8_ext.so.before"; cp "$FP/w1_attn_fp8_ext.so.stockall" "$FP/w1_attn_fp8_ext.so"
say "so $(sha256sum < "$FP/w1_attn_fp8_ext.so" | cut -c1-16)"
printf '%s\n' '{"label": "shadow-lc1", "layers": null, "widths": null, "pdl": true, "fp8_plan": null, "fp8_extra": "shadow", "shadow_widths": [6, 8], "shadow_records": true, "shadow_valid_only": true, "shadow_dir": "/prof/shadow"}' > "$FP/fp8_mask.json"
sed -i 's/^DS41_W1_COMPRESSED_FP8=.*/DS41_W1_COMPRESSED_FP8=1/' "$W/policy.env"
mkdir -p "$SH" 2>/dev/null || sudo -n mkdir -p "$SH"; sudo -n chmod 777 "$SH"; sudo -n rm -rf "$SH"/* 2>/dev/null
cp "$FP/fp8_mask.json" "$W/policy.env" "$O/"
say "A1 $(docker run --rm --network none --entrypoint python3 $IMAGE -c "import hashlib,vllm.models.deepseek_v41.nvidia.flash_mla_mega_attn as m;print(hashlib.sha256(open(m.__file__,'rb').read()).hexdigest())" 2>&1 | tail -1 | cut -c1-80) overlay: $(find $HOME/dsv41/garner/pg61-work/overlay-vllm -name flash_mla_mega_attn.py -exec sha256sum {} + 2>/dev/null | cut -c1-16 | tr '\n' ' ')"
say "A2/A3 $(grep -n 'DS41_CACHE_DIR:=\|/prof"\|:/fastpath' "$W/tree/serve/serve.sh" | cut -c1-120 | tr '\n' '|')"
sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; sleep 30
say "boot"
if ! lane up 1800 > "$O/up.log" 2>&1; then say "BOOTFAIL"; docker logs ds41 > "$O/server.log" 2>&1; down; rollback; say "SHADOW END bootfail"; exit 1; fi
docker logs ds41 > "$O/server.log" 2>&1
say "KV $(grep -o 'GPU KV cache size: [0-9,]* tokens' "$O/server.log" | tail -1)"
grep 'FP8_SHADOW\|FP8_MASK\|captured decode attention calls' "$O/server.log" | cut -c1-700 > "$O/engage.txt"; cat "$O/engage.txt" | cut -c1-400
if ! grep -q 'FP8_SHADOW capture done' "$O/engage.txt"; then say "no capture-done line"; fi
sentinel RESET 60 || say "RESET not acknowledged"
python3 - <<'PY' 2>&1 | tee "$O/request.log"
import hashlib, importlib.util, json, os
R = '$HOME/dsv41/allround-20261112'; DRV = '$HOME/dsv41/decode/w0/w0_drive.py'
src = open(DRV, 'rb').read(); assert hashlib.sha256(src).hexdigest().startswith('6c1fb1be1080'), 'driver drift'
os.environ['W0_BASE'] = 'http://127.0.0.1:30006'; os.environ['W0_MODEL'] = 'dsv41-flash-uva'
spec = importlib.util.spec_from_file_location('w0', DRV); w0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(w0)
ref = next(json.loads(l) for l in open(R + '/serve-quality-a2/r0.jsonl') if json.loads(l)['tag'] == 'prose')
body = w0.chat_body(ref['prompt'], 16, stream=False, logprobs=True)
raw = w0.post('/v1/chat/completions', body, timeout=600); open(R + '/shadow-lc1/response.json', 'wb').write(raw)
lp = json.loads(raw)['choices'][0]['logprobs']['content']
a = [(t['token'], t['logprob']) for t in ref['logprobs'][:len(lp)]]; b = [(t['token'], t['logprob']) for t in lp]
print('PROSE16 tokens', len(b), 'identical_to_A2_r0_prefix', a == b, 'first_diff', next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None))
PY
if sentinel DUMP 300; then stats "$SH/dump-0/stats.json"; fi
# full w0 pass in the same boot -> dump-1 (call-time records keep snapshots exact)
sentinel RESET 60
W0_BASE=http://127.0.0.1:30006 W0_MODEL=dsv41-flash-uva timeout 300 python3 "$DRV" parity "$O/r0-full.jsonl" > "$O/r0-full.log" 2>&1; say "full w0 rc=$?"
timeout 3000 python3 "$T/results/E/lc_drive.py" "$O/lc.json" > "$O/lc.log" 2>&1; say "lc rc=$? $(tail -1 "$O/lc.log")"; grep -c prompt_tokens "$O/lc.log"; cut -c1-150 "$O/lc.log" | head -30
mkdir -p "$O/full"; cp "$O/r0-full.jsonl" "$O/full/r0.jsonl" 2>/dev/null && say "$(python3 "$T/bisect/compare.py" "$R/serve-quality-a2" "$O/full" 1 | cut -c1-160)"
if sentinel DUMP 300; then stats "$SH/dump-1/stats.json"; fi
sudo -n cp -r "$SH"/dump-* "$O/" 2>/dev/null || cp -r "$SH"/dump-* "$O/"; sudo -n chown -R milo "$O" 2>/dev/null
docker logs ds41 > "$O/server-final.log" 2>&1; grep 'FP8_SHADOW dump' "$O/server-final.log" | tail -2 | cut -c1-300
grep -q '"trip": null' "$W/wdp3.status.json" && say WATCHDOG-OK
down; rollback
say "SHADOW END ok"
