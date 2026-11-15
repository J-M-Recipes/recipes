#!/usr/bin/env bash
set -u
R=$HOME/dsv41/allround-20261112; T=$R/milo-takeover; FP=$R/serve-bisect/tree/fastpath/w1attn; O=$R/perf-i1; mkdir -p $O
exec >>"$O/run.log" 2>&1
say(){ echo "$(date '+%F %T') $*"; }
say "PERF-I START"; [ -z "$(docker ps -q)" ] || { say "ABORT containers"; exit 1; }
[ "$(sha256sum < $FP/w1_attn_ext.so | cut -c1-16)" = 7201643d5575c187 ] || { say "ABORT incumbent not shipped"; exit 1; }
[ "$(sha256sum < $FP/w1_attn_fp8_ext.so | cut -c1-16)" = de87450648e96214 ] || { say "ABORT fp8 so not stockall"; exit 1; }
cp $FP/w1_attn_ext.so $O/w1_attn_ext.so.shipped; cp $FP/fp8_mask.json $O/fp8_mask.json.before
printf '{"label": "both-fixed", "layers": null, "widths": null, "pdl": true, "fp8_plan": null, "fp8_extra": null}\n' > $FP/fp8_mask.json
cp $T/I-builds/stock-all/build/w1_attn_ext.so $FP/w1_attn_ext.so; say "installed incumbent $(sha256sum < $FP/w1_attn_ext.so | cut -c1-16)"
cd $T/spec
printf 'todo\tA6\tstock\tserve-spec\ntodo\tF2\tstock\tserve-bisect\ntodo\tA7\tstock\tserve-spec\n' > arms2.tsv; bash spec_runner2.sh; say "perf cells done"
[ -z "$(docker ps -q)" ] || { say "ABORT containers after perf"; }
printf 'todo\tP-F2\tstock\tserve-bisect\n' > arms3.tsv; bash prefill_runner.sh; say "prefill done"
cp $O/w1_attn_ext.so.shipped $FP/w1_attn_ext.so; cp $O/fp8_mask.json.before $FP/fp8_mask.json
say "PERF-I END restored incumbent $(sha256sum < $FP/w1_attn_ext.so | cut -c1-16) running=$(docker ps -q | wc -l)"
