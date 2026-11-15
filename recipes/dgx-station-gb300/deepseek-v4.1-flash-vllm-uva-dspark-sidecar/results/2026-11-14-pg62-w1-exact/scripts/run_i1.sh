#!/usr/bin/env bash
set -u
R=$HOME/dsv41/allround-20261112; T=$R/milo-takeover; I=$T/results/I; PG=$HOME/dsv41/garner/pg61-work/tree
IMAGE=sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3
GPU=nvidia.com/gpu=${GB300_UUID:?set GB300_UUID=GPU-...}
mkdir -p $T/I-probes $T/I-cmp; exec >>"$T/I-run1.log" 2>&1
say(){ echo "$(date '+%F %T') $*"; }
iprobe(){ site=$1; out=$2; shift 2
  timeout 1800 docker run --rm --name dsv41-I-probe --network none --device $GPU --ulimit core=0 --shm-size 2g \
    -e PYTHONDONTWRITEBYTECODE=1 -v $I:/i:ro -v $T/I-builds:/b:ro -v $PG/fastpath/w1attn:/ship:ro \
    -v "$(dirname "$site")":/s:ro -v $T/I-probes:/out --entrypoint python3 $IMAGE /i/incumbent_probe.DRAFT.py \
    /s/"$(basename "$site")" --ops /ship/ops.py --shipped /ship/w1_attn_ext.so --out /out/$out.json "$@" > $T/I-probes/$out.log 2>&1
  say "PROBE $out rc=$?"; grep -E "VERDICT|reproduc|DIFF|excluded|Traceback|Error|INCONCLUSIVE" $T/I-probes/$out.log | cut -c1-420 | head -30; }
say "I1 START"; [ -z "$(docker ps -q)" ] || { say "I1 ABORT containers"; exit 1; }
exec 9>>"$R/gpu-owner.lock"; flock -n 9 || { say "I1 ABORT lock"; exit 1; }
cd $I && sha256sum * 2>/dev/null > SHIPPED.sha256
for b in "control" "stock-all -DW1_VARIANT_STOCK_ALL" "sink-unfused -DW1_VARIANT_SINK_UNFUSED"; do
  set -- $b; bash $I/build_incumbent.sh "$@" 2>&1 | tail -9 | cut -c1-220; done
for s in site-0087-inc-swa-L01-w8 site-0126-inc-swa-L00-w6 site-0127-inc-swa-L01-w6; do
  iprobe $R/shadow-h1/dump-1/$s.pt $s --so control=/b/control/build/w1_attn_ext.so --so stock-all=/b/stock-all/build/w1_attn_ext.so --so sink-unfused=/b/sink-unfused/build/w1_attn_ext.so
done
say "I1 END running=$(docker ps -q | wc -l)"
