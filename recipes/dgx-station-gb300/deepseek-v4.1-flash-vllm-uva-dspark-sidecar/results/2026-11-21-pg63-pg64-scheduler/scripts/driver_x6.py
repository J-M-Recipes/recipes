#!/usr/bin/env python3
"""Card X5 driver (Pg63-based): K1 live -> D1 -> Y1 -> D2 -> Y2 -> restore -> K2.
Each arm: battery + post suite + C32x5 + det_check + phase-locked yield probes (P1 413K@0.8 x5, P2 8K-prompt short @0.8 x2)."""
import json, os, subprocess, time, importlib.util
B = "~/hermes/research/dsv41-gb300-owen-transfer-20261121/cards"
spec = importlib.util.spec_from_file_location("drv", f"{B}/driver_lib.py"); drv = importlib.util.module_from_spec(spec); spec.loader.exec_module(drv)
XF = drv.XF; ROOT = "http://gb300:30006"
BENCH = os.path.expanduser("~/hermes/work/dsv41-5way-2026-10-26"); OUT = f"{B}/receipts/x6.json"
CAD = "VLLM_PREFILL_CADENCE=3 VLLM_PREFILL_CADENCE_MIN_REMAINING=8192"
ARMS = [("Y3", "yieldcad", CAD + " VLLM_PREFILL_YIELD=short VLLM_PREFILL_YIELD_LIMIT_S=2", "-")]
def box_op2(args, tag):
    lf = f"{XF}/logs/op2-{tag}-{int(time.time())}.log"
    drv.ssh(f"cd {XF}; (setsid nohup bash {XF}/xfer_arm2.sh {args} > {lf} 2>&1 < /dev/null &)")
    drv.log(f"box op2 '{args}' log {lf}"); t0 = time.time()
    while time.time() - t0 < 3600:
        time.sleep(20)
        try: r = drv.ssh(f"grep -a 'OP .* END rc=' {lf} | tail -1", 60).stdout.strip()
        except Exception as e: drv.log(f"poll error {e}"); continue
        if r:
            drv.log(f"box op2 done: {r}"); subprocess.run(["scp", "-q", f"{drv.BOX}:{lf}", f"{B}/logs/"]); return int(r.rsplit("rc=", 1)[1])
    return 99
def save(arm, key, val):
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res.setdefault(arm, {})[key] = val; json.dump(res, open(OUT, "w"), indent=1)
def extras(arm):
    c = []
    for i in range(1, 6):
        lab = f"x5-{arm}-c32-r{i}"
        subprocess.run(f"DSV41_ENCODING_DIR=tools/encoding python3 tools/v41bench_tf.py --base {ROOT}/v1 --model deepseek-v4.1-flash --label {lab} --levels 32 --prefill '' > /dev/null 2>&1", shell=True, cwd=BENCH)
        try:
            bs = [x for x in json.load(open(f"{BENCH}/bench-{lab}.json"))["batches"] if x["c"] == 32]
            c.append({"mean": sum(x["agg_tok_s"] for x in bs) / len(bs), "stalls": sum(1 for x in bs if x["ttft_mean_s"] > 0.5)})
        except Exception as e: c.append({"err": str(e)})
    save(arm, "c32x5", c); drv.log(f"{arm} C32x5 {[round(x.get('mean', 0)) for x in c]} stalls {[x.get('stalls') for x in c]}")
    r = subprocess.run(["python3", f"{B}/tools/det_check.py", ROOT, f"{B}/logs/det-{arm}.json"], capture_output=True, text=True)
    save(arm, "det", r.stdout.strip()); drv.log(f"{arm} det {r.stdout.strip()}")
    p1 = []
    for i in range(5):
        r = subprocess.run(["python3", f"{B}/tools/phase_trace.py", ROOT, "413000", "0.8", "0", f"{B}/logs/p1-{arm}-{i}.json"], capture_output=True, text=True)
        try: p1.append(json.loads(r.stdout.strip().splitlines()[-1]))
        except Exception: p1.append({"err": r.stderr[-300:]})
        time.sleep(3)
    save(arm, "p1", p1); drv.log(f"{arm} P1 ttft {[x.get('ttft') for x in p1]} long {[x.get('long_done') for x in p1]} solo {[x.get('solo_s') for x in p1]}")
    p2 = []
    for i in range(2):
        r = subprocess.run(["python3", f"{B}/tools/phase_trace.py", ROOT, "413000", "0.8", "8192", f"{B}/logs/p2-{arm}-{i}.json"], capture_output=True, text=True)
        try: p2.append(json.loads(r.stdout.strip().splitlines()[-1]))
        except Exception: p2.append({"err": r.stderr[-300:]})
        time.sleep(3)
    save(arm, "p2", p2); drv.log(f"{arm} P2(8K) ttft {[x.get('ttft') for x in p2]}")
    eng = drv.ssh("docker logs ds41 2>&1 | grep -acE 'xfer prefill cadence enabled'; docker logs ds41 2>&1 | grep -ac yield; docker logs ds41-sidecar 2>&1 | grep -aiE 'determin|ROUTE_BUFFER' | tail -2", 120).stdout
    save(arm, "engagement", eng); drv.log(f"{arm} engagement: {eng.strip()[:300]}")
drv.log("X6 START")
try:
    for name, ovs, env, side in ARMS:
        a = "-" if env == "-" else f"'{env}'"; s = "-" if side == "-" else f"'{side}'"
        if box_op2(f"arm {name} {ovs} {a} {s}", name) != 0: drv.log(f"arm {name} failed"); continue
        drv.battery(f"gb300-{name}"); extras(name)
finally:
    rc = box_op2("restore", "restore")
    if rc == 0: drv.log("X6 END (lane = Pg63 restored)")
    else: drv.log(f"X6 END RESTORE FAILED rc={rc}")
