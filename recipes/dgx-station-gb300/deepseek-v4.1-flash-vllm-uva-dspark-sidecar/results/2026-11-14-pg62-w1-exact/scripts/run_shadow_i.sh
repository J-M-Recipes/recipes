#!/usr/bin/env bash
# Desk I DRAFT (not run by the desk). Station, as milo. One shadow boot of serve-bisect with H's integrate (unchanged)
# and a chosen INCUMBENT w1_attn_ext.so; FP8 path on the .so.stockall build exactly as shadow-h1. Ends DARK, tree rolled
# back (integrate.py, fp8_mask.json, policy.env AND w1_attn_ext.so).
#   usage: run_shadow_i.sh <label> <build> <mode> [shadow_widths_json]
#     build: shipped | <name in $T/I-builds> ($T/I-builds/<name>/build/w1_attn_ext.so)
#     mode:  fixture  - prose16 -> DUMP, then per A2 prompt: RESET, that prompt, DUMP (dumps right after the first
#                       mismatching call's request: its pages are not yet reused, so snapshots rebuild exactly)
#            validate - shadow-h1's workload: prose16 -> DUMP, RESET, full w0 + gt set A + set B + C8 -> DUMP
#     shadow_widths_json default [6, 7, 8] (7 = the drafter's width; shadow-h1 snapshotted only 6 and 8)
# Delta to results/H/run_shadow_h.sh: label/out dir, SHIPPED line fixed to results/H, incumbent .so backup/install/
# rollback with sha checks, widths parameter, fixture mode, per-path stats reader. Same lock / maintenance / drop
# caches / lane up / watchdog / down sequence.
set -u
R=$HOME/dsv41/allround-20261112; T=$R/milo-takeover; I=$T/results/I; W=$R/serve-bisect; GR=$HOME/dsv41/garner
DRV=$HOME/dsv41/decode/w0/w0_drive.py
MODEL=/models/DeepSeek-V4.1-Flash-1017-1bae0e1088993b9b82de08f59de5be5a34e42170
IMAGE=sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3
SHIPPED_SHA=7201643d5575c18706639a0b40b7e1f7482ef2535ebc4c54c55920116743e35f
label=${1:?label}; build=${2:?build}; mode=${3:?mode}; widths=${4:-[6, 7, 8]}
case $mode in fixture|validate) ;; *) echo "mode must be fixture or validate" >&2; exit 2 ;; esac
FP=$W/tree/fastpath/w1attn; SH=$W/home/state/prof/shadow; O=$R/shadow-$label
[ -e "$O" ] && { echo "exists: $O" >&2; exit 2; }
if [ "$build" = shipped ]; then NEWSO=""; else NEWSO=$T/I-builds/$build/build/w1_attn_ext.so; [ -f "$NEWSO" ] || { echo "no $NEWSO" >&2; exit 2; }; fi
mkdir -p "$O"; exec >>"$O/run.log" 2>&1
say(){ echo "$(date '+%F %T') $*"; }
lane(){ env WORK="$W" DS41_HOME="$W/home" MODEL_DIR="$MODEL" IMAGE="$IMAGE" DS41_IMAGE="$IMAGE" timeout "$2" bash "$GR/launcher/launch-pg61.sh" "$1"; }
down(){ lane down 240 >>"$O/down.log" 2>&1; docker ps --format '{{.Names}}' | grep -qxE 'ds41|ds41-sidecar' && docker stop -t 10 ds41 ds41-sidecar; say "DOWN running=[$(docker ps --format '{{.Names}}' | tr '\n' ' ')]"; }
rollback(){ cp "$O/integrate.py.bisect" "$FP/integrate.py"; cp "$O/fp8_mask.json.bisect" "$FP/fp8_mask.json"; cp "$O/policy.env.bisect" "$W/policy.env"
            cp "$O/w1_attn_ext.so.before" "$FP/w1_attn_ext.so"
            say "ROLLED BACK tree (incumbent so $(sha256sum < "$FP/w1_attn_ext.so" | cut -c1-16); fp8 so left as stockall)"; }
sentinel(){ rm -f "$SH/$1.done" "$SH/$1.error"; touch "$SH/$1"; for i in $(seq "$2"); do [ -e "$SH/$1.done" ] || [ -e "$SH/$1.error" ] && break; sleep 1; done; say "$1: $(cat "$SH/$1.done" "$SH/$1.error" "$SH/NOT_READY" 2>/dev/null | head -c 300)"; [ -e "$SH/$1.done" ]; }
stats(){ python3 - "$1" <<'PY'
import json, sys
r = json.load(open(sys.argv[1])); dev = r['devices']['cuda:0']
print('BY_PATH', json.dumps(r.get('totals_by_path'))[:600], 'OVERFLOW', dev['overflow_by_path'])
for g in r.get('totals_by_group', []):
    if g['path'] == 'incumbent':
        print('INC', g['prefix'], 'sites', g['sites'], 'calls', g['calls'], 'sites_mis', g['calls_any'], 'saved', g['saved'])
for s in dev['sites']:
    if s['calls_with_any_mismatch']:
        print({k: s.get(k) for k in ('path', 'prefix', 'width', 'calls', 'calls_with_any_mismatch', 'mismatching_data_bytes',
                                      'mismatching_scale_bytes', 'first_mismatch_seq', 'snapshot_file')})
PY
}
say "SHADOW-I START label=$label build=$build mode=$mode widths=$widths"
[ -z "$(docker ps -q)" ] || { say "ABORT containers running"; exit 1; }
pgrep -f "bisect_runner.sh|spec_runner|prefill_runner" >/dev/null && { say "ABORT runner alive"; exit 1; }
grep -qi allround $HOME/dsv41/m3/logs/MAINTENANCE || { say "ABORT no maintenance flag"; exit 1; }
exec 9>>"$R/gpu-owner.lock"; flock -n 9 || { say "ABORT lock"; exit 1; }
( cd "$T/results/H" && sha256sum *.DRAFT.py; cd "$I" && sha256sum *.DRAFT.* *.sh *.py ) > "$O/SHIPPED.sha256"
cp -n "$FP/integrate.py" "$O/integrate.py.bisect"; cp -n "$FP/fp8_mask.json" "$O/fp8_mask.json.bisect"; cp -n "$W/policy.env" "$O/policy.env.bisect"
cp -n "$FP/w1_attn_ext.so" "$O/w1_attn_ext.so.before"
before=$(sha256sum < "$O/w1_attn_ext.so.before" | cut -c1-64)
[ "$before" = "$SHIPPED_SHA" ] || { say "ABORT serve-bisect incumbent is ${before:0:16}, not the shipped ${SHIPPED_SHA:0:16} (tree not in the expected state)"; exit 1; }
cp "$T/results/H/integrate_shadow_H.DRAFT.py" "$FP/integrate.py" && python3 -m py_compile "$FP/integrate.py" || { say "ABORT compile"; rollback; exit 1; }
cp "$FP/w1_attn_fp8_ext.so.stockall" "$FP/w1_attn_fp8_ext.so"
[ -n "$NEWSO" ] && cp "$NEWSO" "$FP/w1_attn_ext.so"
say "fp8 so $(sha256sum < "$FP/w1_attn_fp8_ext.so" | cut -c1-16)  inc so $(sha256sum < "$FP/w1_attn_ext.so" | cut -c1-16) (build=$build)"
printf '{"label": "shadow-%s", "layers": null, "widths": null, "pdl": true, "fp8_plan": null, "fp8_extra": "shadow", "shadow_incumbent": true, "shadow_widths": %s, "shadow_records": true, "shadow_valid_only": true, "shadow_dir": "/prof/shadow"}\n' "$label" "$widths" > "$FP/fp8_mask.json"
python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$FP/fp8_mask.json" || { say "ABORT mask json"; rollback; exit 1; }
sed -i 's/^DS41_W1_COMPRESSED_FP8=.*/DS41_W1_COMPRESSED_FP8=1/' "$W/policy.env"
mkdir -p "$SH" 2>/dev/null || sudo -n mkdir -p "$SH"; sudo -n chmod 777 "$SH"; sudo -n rm -rf "$SH"/* 2>/dev/null
cp "$FP/fp8_mask.json" "$W/policy.env" "$O/"
sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'; sleep 30
say "boot"
if ! lane up 1800 > "$O/up.log" 2>&1; then say "BOOTFAIL"; docker logs ds41 > "$O/server.log" 2>&1; down; rollback; say "SHADOW-I END bootfail"; exit 1; fi
docker logs ds41 > "$O/server.log" 2>&1
say "KV $(grep -o 'GPU KV cache size: [0-9,]* tokens' "$O/server.log" | tail -1)"
grep 'FP8_SHADOW\|FP8_MASK\|captured decode attention calls' "$O/server.log" | cut -c1-700 > "$O/engage.txt"; cut -c1-400 "$O/engage.txt"
grep -q 'FP8_SHADOW incumbent installed' "$O/engage.txt" || say "WARNING no incumbent shadow install line"
sentinel RESET 60 || say "RESET not acknowledged"
# one prompt -> response file + prefix compare against A2 r0 (same code as run_shadow_h.sh, parameterised by tag)
one(){ python3 - "$1" "$2" "$O" <<'PY'
import hashlib, importlib.util, json, os, sys
tag, ntok, O = sys.argv[1], int(sys.argv[2]), sys.argv[3]
R = '$HOME/dsv41/allround-20261112'; DRV = '$HOME/dsv41/decode/w0/w0_drive.py'
src = open(DRV, 'rb').read(); assert hashlib.sha256(src).hexdigest().startswith('6c1fb1be1080'), 'driver drift'
os.environ['W0_BASE'] = 'http://127.0.0.1:30006'; os.environ['W0_MODEL'] = 'dsv41-flash-uva'
spec = importlib.util.spec_from_file_location('w0', DRV); w0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(w0)
ref = next(json.loads(l) for l in open(R + '/serve-quality-a2/r0.jsonl') if json.loads(l)['tag'] == tag)
raw = w0.post('/v1/chat/completions', w0.chat_body(ref['prompt'], ntok, stream=False, logprobs=True), timeout=600)
open(f'{O}/response-{tag}-{ntok}.json', 'wb').write(raw)
lp = json.loads(raw)['choices'][0]['logprobs']['content']
a = [(t['token'], t['logprob']) for t in ref['logprobs'][:len(lp)]]; b = [(t['token'], t['logprob']) for t in lp]
print(tag, ntok, 'tokens', len(b), 'identical_to_A2_r0_prefix', a == b,
      'first_diff', next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None))
PY
}
one prose 16 | tee -a "$O/request.log"
if sentinel DUMP 300; then stats "$SH/dump-0/stats.json"; fi
if [ "$mode" = fixture ]; then
  for tag in $(python3 -c "import json;print(' '.join(json.loads(l)['tag'] for l in open('$R/serve-quality-a2/r0.jsonl')))"); do
    sentinel RESET 60 || say "RESET not acknowledged before $tag"
    one "$tag" 256 | tee -a "$O/request.log"
    if sentinel DUMP 300; then say "dump after $tag:"; stats "$(ls -d "$SH"/dump-* | sort -V | tail -1)/stats.json"; fi
  done
else
  sentinel RESET 60
  W0_BASE=http://127.0.0.1:30006 W0_MODEL=dsv41-flash-uva timeout 300 python3 "$DRV" parity "$O/r0-full.jsonl" > "$O/r0-full.log" 2>&1; say "full w0 rc=$?"
  GT=$HOME/dsv41/decode/bench/gt_mix_drive.py
  for c in "pass --set A" "pass --set B" "conc --n 8 --tag shadow-$label-c8"; do W0_BASE=http://127.0.0.1:30006 W0_MODEL=dsv41-flash-uva timeout 900 python3 "$GT" $c --out "$O/gt-$(echo $c | tr -c 'a-zA-Z0-9' _).json" > /dev/null 2>&1; say "gt [$c] rc=$?"; done
  mkdir -p "$O/full"; cp "$O/r0-full.jsonl" "$O/full/r0.jsonl" 2>/dev/null && say "vs A2: $(python3 "$T/bisect/compare.py" "$R/serve-quality-a2" "$O/full" 1 | cut -c1-160)"
  [ -f "$R/shadow-h1/r0-full.jsonl" ] && { mkdir -p "$O/ref-h1"; cp "$R/shadow-h1/r0-full.jsonl" "$O/ref-h1/r0.jsonl"; say "vs shadow-h1 (all-stock values): $(python3 "$T/bisect/compare.py" "$O/ref-h1" "$O/full" 1 | cut -c1-160)"; }
  if sentinel DUMP 300; then stats "$SH/dump-1/stats.json"; fi
fi
sudo -n cp -r "$SH"/dump-* "$O/" 2>/dev/null || cp -r "$SH"/dump-* "$O/"; sudo -n chown -R milo "$O" 2>/dev/null
docker logs ds41 > "$O/server-final.log" 2>&1; grep 'FP8_SHADOW dump' "$O/server-final.log" | tail -3 | cut -c1-400
grep -q '"trip": null' "$W/wdp3.status.json" && say WATCHDOG-OK
down; rollback
say "SHADOW-I END ok"
