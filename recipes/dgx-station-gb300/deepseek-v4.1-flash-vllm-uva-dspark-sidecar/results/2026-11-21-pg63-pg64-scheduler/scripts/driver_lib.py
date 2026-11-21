#!/usr/bin/env python3
"""xfer-20261121 card driver (runs on the M4, detached). usage: driver.py ARMSPEC [ARMSPEC...]
ARMSPEC = name:ovsets:armenv   (ovsets comma list or none; armenv K=V joined by '+', or '-')
e.g. B:cadence:VLLM_PREFILL_CADENCE=2   GR:g59131,r63540:-
Waits for WAIT_PID (A1 measurement) first. After the arms it always restores Pg62 and runs the A2 control.
A failed arm boot is logged and the driver moves on to the next arm. Arm C (cadence 3) is skipped when B's
decode-during-413K median is < 1.5x A1's (pre-registered skip rule, W1 card 6.2)."""
import glob, json, os, statistics as st, subprocess, sys, time
B = "~/hermes/research/dsv41-gb300-owen-transfer-20261121/cards"
BOX = "milo@gb300"; XF = "~/dsv41/xfer-20261121"
LOG = open(f"{B}/logs/driver.log", "a", buffering=1)
def log(m): LOG.write(time.strftime("%F %T ") + m + "\n")
def ssh(cmd, timeout=120):
    return subprocess.run(["ssh", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=30", BOX, cmd],
                          capture_output=True, text=True, timeout=timeout)
def box_op(args, tag):
    lf = f"{XF}/logs/op-{tag}-{int(time.time())}.log"
    ssh(f"mkdir -p {XF}/logs; cd {XF}; (setsid nohup bash {XF}/xfer_arm.sh {args} > {lf} 2>&1 < /dev/null &)")
    log(f"box op '{args}' log {lf}")
    t0 = time.time()
    while time.time() - t0 < 3600:
        time.sleep(20)
        try: r = ssh(f"grep -a 'OP .* END rc=' {lf} | tail -1", 60).stdout.strip()
        except Exception as e: log(f"poll error {e}"); continue
        if r:
            log(f"box op done: {r}"); rc = int(r.rsplit("rc=", 1)[1])
            subprocess.run(["scp", "-q", f"{BOX}:{lf}", f"{B}/logs/"]); return rc
    log("box op TIMEOUT"); return 99
def battery(label):
    cmd = ["python3", "battery.py", "--lane", label, "--api-root", "http://gb300:30006", "--model",
           "deepseek-v4.1-flash", "--engine", "vllm", "--out", f"{B}/receipts", "--allow-1m", "--reps", "3", "--no-ssh",
           "--kv-pool-tokens", "11696582", "--phases", "smoke,c1,c16,prefill,needle,mixed_short,mixed_warm,agent_turn,qeval"]
    r = subprocess.run(cmd, cwd=os.path.expanduser("~/hermes/work/dsv41-5way-2026-10-26"),
                       stdout=open(f"{B}/logs/battery-{label}.log", "w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    log(f"battery {label} rc={r.returncode}")
    r = subprocess.run(["bash", f"{B}/tools/post_arm_gb300.sh", label], stdout=open(f"{B}/logs/post-{label}.log", "w"),
                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    log(f"post {label} rc={r.returncode}")
    # engine-log engagement lines for this arm
    r = ssh("docker logs ds41 2>&1 | grep -aE 'xfer (prefill cadence|#59131)' | tail -5", 60)
    log(f"engagement {label}: " + (r.stdout.strip().replace("\n", " || ") or "none"))
def dpp(label):
    v = []
    for p in glob.glob(f"{B}/receipts/post/{label}/trace-d-*.json"):
        try: v.append(json.load(open(p))["ev"].get("decode_during_prefill_tok_s"))
        except Exception: pass
    v = [x for x in v if x]; return st.median(v) if v else None
