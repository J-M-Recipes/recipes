#!/usr/bin/env bash
set -u
R=$HOME/dsv41/allround-20261112; T=$R/milo-takeover; F=$T/results/F
export SRC=$R/w1-hardening-build/source/kernels/w1-attn
IMAGE=sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3
GPU=nvidia.com/gpu=${GB300_UUID:?set GB300_UUID=GPU-...}
exec >>"$T/F-run.log" 2>&1
say(){ echo "$(date '+%F %T') $*"; }
probe(){ VB=$T/F-builds/$1; shift
  timeout 900 docker run --rm --name dsv41-F-probe --network none --device $GPU --ulimit core=0 --shm-size 2g \
    -e PYTHONDONTWRITEBYTECODE=1 -v $T/results/E:/e:ro -v $F:/f:ro -v $R/shadow-e1:/o:ro -v $VB:/vb \
    -v $VB/fastpath:/fastpath:ro --entrypoint python3 $IMAGE /f/f_probe.py /o/dump-1/site-0088-L14-w8.pt \
    --fastpath /fastpath --repro /e/shadow_repro.DRAFT.py --out /vb/F-probe.json "$@" > $VB/probe.log 2>&1
  say "PROBE $(basename $VB) rc=$?"; grep -E "TARGET|VERDICT|DIAG|byte |Traceback|Error" $VB/probe.log | cut -c1-900; }
say "F START"; [ -z "$(docker ps -q)" ] || { say "F ABORT containers"; exit 1; }
exec 9>>"$R/gpu-owner.lock"; flock -n 9 || { say "F ABORT lock"; exit 1; }
for f in stock w1; do say "SASS $f"; awk -v re="$([ $f = stock ] && echo 'core_attn|Decode' || echo .)" '/Function :/ { fn = $0; keep = (fn ~ re); n = 0; next } keep && (/1\.44269502/ || /0x3fb8aa3b/) { print substr(fn,1,150); print "   " $0; n = 4; hits++; next } keep && n > 0 { print "   " $0; n-- } END { print "L2E_HITS " hits + 0 }' $T/sass/$f.sass | cut -c1-200 | head -60; done
mkdir -p $T/F-builds/w13/fastpath && cp -a $R/serve-bisect/tree/fastpath/. $T/F-builds/w13/fastpath/ && cp $R/serve-bisect/tree/fastpath/w1attn/w1_attn_fp8_ext.so.w13 $T/F-builds/w13/fastpath/w1attn/w1_attn_fp8_ext.so
probe w13
bash $F/build_variant.sh control | tail -3; probe control
bash $F/build_variant.sh stock-all -DW1_VARIANT_STOCK_ALL | tail -3; probe stock-all
bash $F/build_variant.sh diag2 -DW1_DIAG -DW1_DIAG_ROW=1 -DW1_DIAG_HEAD=31 -DW1_DIAG_DIM=284 -DW1_DIAG_WHAT=2 | tail -3; probe diag2 --diag-what 2
bash $F/build_variant.sh diag1 -DW1_DIAG -DW1_DIAG_ROW=1 -DW1_DIAG_HEAD=31 -DW1_DIAG_DIM=284 -DW1_DIAG_WHAT=1 | tail -3; probe diag1 --diag-what 1
bash $F/build_variant.sh diag0 -DW1_DIAG -DW1_DIAG_ROW=1 -DW1_DIAG_HEAD=31 -DW1_DIAG_DIM=284 -DW1_DIAG_WHAT=0 | tail -3; probe diag0 --diag-what 0
say "F END running=$(docker ps -q | wc -l)"
