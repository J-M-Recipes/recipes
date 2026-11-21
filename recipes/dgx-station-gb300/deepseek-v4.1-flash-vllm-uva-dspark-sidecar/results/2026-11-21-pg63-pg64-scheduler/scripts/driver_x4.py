#!/usr/bin/env python3
"""Card X4 driver: A6 (live) -> G1 -> G2 -> restore -> A7; each arm = battery + post + C32 x5."""
import json, os, subprocess, importlib.util
B = "~/hermes/research/dsv41-gb300-owen-transfer-20261121/cards"
spec = importlib.util.spec_from_file_location("drv", f"{B}/driver_lib.py"); drv = importlib.util.module_from_spec(spec); spec.loader.exec_module(drv)
BENCH = os.path.expanduser("~/hermes/work/dsv41-5way-2026-10-26")
OUT = f"{B}/receipts/x4-c32.json"
def c32x5(arm):
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res[arm] = []
    for i in range(1, 6):
        lab = f"x4-{arm}-c32-r{i}"
        subprocess.run(f"DSV41_ENCODING_DIR=tools/encoding python3 tools/v41bench_tf.py --base http://gb300:30006/v1 --model deepseek-v4.1-flash --label {lab} --levels 32 --prefill '' > /dev/null 2>&1", shell=True, cwd=BENCH)
        try:
            b = json.load(open(f"{BENCH}/bench-{lab}.json")); bs = [x for x in b["batches"] if x["c"] == 32]
            r = {"mean": sum(x["agg_tok_s"] for x in bs) / len(bs), "cells": {x["category"]: [x["agg_tok_s"], x["ttft_mean_s"]] for x in bs}}
        except Exception as e: r = {"err": str(e)}
        res[arm].append(r); drv.log(f"C32x5 {arm} r{i}: mean={r.get('mean', 0):.1f} stalls>0.5s={sum(1 for v in r.get('cells', {}).values() if v[1] > 0.5)}")
        json.dump(res, open(OUT, "w"), indent=1)
drv.log("X4 START")
drv.battery("gb300-pg62-A6"); c32x5("A6")
try:
    for name in ("G1", "G2"):
        if drv.box_op(f"arm {name} cadence3 'VLLM_PREFILL_CADENCE=3 VLLM_PREFILL_CADENCE_MIN_REMAINING=8192'", name) != 0:
            drv.log(f"arm {name} failed"); continue
        drv.battery(f"gb300-{name}"); c32x5(name)
finally:
    rc = drv.box_op("restore", "restore")
    if rc == 0:
        drv.battery("gb300-pg62-A7"); c32x5("A7"); drv.log("X4 END (lane = Pg62 restored)")
    else: drv.log(f"X4 END RESTORE FAILED rc={rc}")
