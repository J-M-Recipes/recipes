#!/usr/bin/env bash
set -u
R=$HOME/dsv41/allround-20261112; T=$R/milo-takeover; F=$T/results/F
export SRC=$R/w1-hardening-build/source/kernels/w1-attn
IMAGE=sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3
GPU=nvidia.com/gpu=${GB300_UUID:?set GB300_UUID=GPU-...}
exec >>"$T/F-run2.log" 2>&1
say(){ echo "$(date '+%F %T') $*"; }
probe(){ VB=$T/F-builds/$1; shift
  timeout 900 docker run --rm --name dsv41-F-probe --network none --device $GPU --ulimit core=0 --shm-size 2g \
    -e PYTHONDONTWRITEBYTECODE=1 -v $T/results/E:/e:ro -v $F:/f:ro -v $R/shadow-e1:/o:ro -v $VB:/vb \
    -v $VB/fastpath:/fastpath:ro --entrypoint python3 $IMAGE /f/f_probe.py /o/dump-1/site-0088-L14-w8.pt \
    --fastpath /fastpath --repro /e/shadow_repro.DRAFT.py --out /vb/F-probe.json "$@" > $VB/probe.log 2>&1
  say "PROBE $(basename $VB) rc=$?"; grep -E "TARGET|VERDICT|DIAG|byte |Traceback|Error" $VB/probe.log | cut -c1-900; }
say "F START"; [ -z "$(docker ps -q)" ] || { say "F ABORT containers"; exit 1; }
exec 9>>"$R/gpu-owner.lock"; flock -n 9 || { say "F ABORT lock"; exit 1; }








for v in "sink-unfused -DW1_VARIANT_SINK_UNFUSED" "ex2-noftz -DW1_VARIANT_EX2_NOFTZ" "oscdiv-noftz -DW1_VARIANT_OSC_DIV_NOFTZ" "outdiv-rn -DW1_VARIANT_OUT_DIV_RN"; do set -- $v; bash $F/build_variant.sh $1 $2 | tail -2; probe $1; done
say "F END running=$(docker ps -q | wc -l)"
