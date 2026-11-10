#!/bin/bash
# Card HC1 cells, identical for every arm (Milo, 2026-11-10). Env: HC_OUT (file prefix), HC_ARM (tag).
# Cells on :30006 over loopback: two unmeasured warm passes, C1 mixed passes A/B twice, C8 x3 and C32 x3 (a new salt
# tag per rep so no rep is a prefix-cache hit; the same tags in every arm), first token by prompt length, tool gauntlet.
# Prints one line per cell; never exits non-zero for a failed cell (the line says so).
set -u
: "${HC_OUT:?}" "${HC_ARM:?}"
DEC=$HOME/dsv41/decode; cd "$DEC" || exit 9
export W0_BASE=http://127.0.0.1:30006 W0_MODEL=dsv41-flash-uva
say() { echo "$(date '+%F %T %Z') [hc1 $HC_ARM] $*"; }
run() { local name=$1; shift; timeout 1800 python3 bench/gt_mix_drive.py "$@" --out "$HC_OUT-$name.json" > "$HC_OUT-$name.log" 2>&1
        say "$name rc=$? $(tail -1 "$HC_OUT-$name.log" | cut -c1-300)"; }
run warm-a pass --set A
run warm-b pass --set B
run c1-a1 pass --set A
run c1-b1 pass --set B
for r in 1 2 3; do run c8-r$r conc --n 8 --tag hc1-c8-r$r; done
for r in 1 2 3; do run c32-r$r conc --n 32 --tag hc1-c32-r$r; done
run c1-a2 pass --set A
run c1-b2 pass --set B
run ttft ttft --sizes 32,300,1000,8000,64000 --reps 3
GDIR=$HOME/pin-hot-experts/round7/gauntlet; GIMG=sha256:00d577a6a63281e15336029d5bcee4e9a2cf182214a4f20ba6111b1c8e79893d
rm -rf "$HC_OUT-gauntlet"   # run_suite.py creates its output dir itself and refuses an existing one
( cd "$GDIR" && timeout 1500 python3 run_suite.py --base-url "$W0_BASE/v1" --model "$W0_MODEL" --output "$HC_OUT-gauntlet" \
    --tag "hc1-$HC_ARM" --cache-salt "hc1-$HC_ARM" --sandbox-image "$GIMG" --workers 4 ) > "$HC_OUT-gauntlet.log" 2>&1
say "gauntlet rc=$? ok $(grep -c '"success": true' "$HC_OUT-gauntlet.log") fail $(grep -c '"success": false' "$HC_OUT-gauntlet.log")"
say "cells done"
