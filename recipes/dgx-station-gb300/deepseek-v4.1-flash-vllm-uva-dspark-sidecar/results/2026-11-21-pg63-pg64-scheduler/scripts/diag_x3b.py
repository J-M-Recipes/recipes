#!/usr/bin/env python3
"""X3b diagnostic (no promotion): where does the C32 'math' stall in cadence arms come from?
Per arm: fresh boot -> C32 x3 (cold, no long traces before) -> 1M decode trace -> C32 x2 -> save full engine log.
Arms: E (cadence3 + env), F (cadence3 overlay, NO env = stock code path), then restore Pg62 and run the same sequence (control)."""
import json, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(__file__))
B = "~/hermes/research/dsv41-gb300-owen-transfer-20261121/cards"
os.environ.setdefault("WAIT_PID", "")
import importlib.util
spec = importlib.util.spec_from_file_location("drv", f"{B}/driver_lib.py"); drv = importlib.util.module_from_spec(spec); spec.loader.exec_module(drv)
BENCH = os.path.expanduser("~/hermes/work/dsv41-5way-2026-10-26")
def c32(label):
    subprocess.run(f"DSV41_ENCODING_DIR=tools/encoding python3 tools/v41bench_tf.py --base http://gb300:30006/v1 --model deepseek-v4.1-flash --label {label} --levels 32 --prefill '' > /dev/null 2>&1", shell=True, cwd=BENCH)
    try:
        b = json.load(open(f"{BENCH}/bench-{label}.json"))
        line = " ".join(f"{x['category'][:5]}={round(x['agg_tok_s'])}/{x['ttft_mean_s']:.2f}s" for x in b["batches"] if x["c"] == 32)
        mean = sum(x["agg_tok_s"] for x in b["batches"] if x["c"] == 32) / 9
    except Exception as e: line, mean = f"ERR {e}", 0
    drv.log(f"C32 {label}: mean={mean:.1f} | {line}")
def seq(tag):
    for i in (1, 2, 3): c32(f"diag-{tag}-cold-r{i}")
    r = subprocess.run(["python3", f"{B}/tools/mixed_trace.py", "http://gb300:30006", "deepseek-v4.1-flash", "1000000", "4.0",
                        f"{B}/logs/diag-{tag}-m1m.json", "400", "off"], capture_output=True, text=True)
    drv.log(f"1M trace {tag}: {r.stdout.strip()[:300]}")
    for i in (1, 2): c32(f"diag-{tag}-post1m-r{i}")
    drv.ssh(f"docker logs ds41 > ~/dsv41/xfer-20261121/logs/engine-diag-{tag}.log 2>&1", 300)
    subprocess.run(["scp", "-q", f"milo@gb300:~/dsv41/xfer-20261121/logs/engine-diag-{tag}.log", f"{B}/logs/"])
drv.log("DIAG START")
try:
    for name, ovs, env in [("E3", "cadence3", "'VLLM_PREFILL_CADENCE=3 VLLM_PREFILL_CADENCE_MIN_REMAINING=8192'"), ("F", "cadence3", "-")]:
        if drv.box_op(f"arm {name} {ovs} {env}", name) != 0: drv.log(f"arm {name} failed"); continue
        seq(name)
finally:
    rc = drv.box_op("restore", "restore")
    if rc == 0: seq("A5"); drv.log("DIAG END (lane = Pg62 restored)")
    else: drv.log(f"DIAG END RESTORE FAILED rc={rc}")
