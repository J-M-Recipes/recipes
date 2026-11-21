import os, subprocess, sys, time
B = "~/hermes/research/dsv41-gb300-owen-transfer-20261121/cards"
LOG = open(f"{B}/logs/driver.log", "a", buffering=1)
def log(m): LOG.write(time.strftime("%F %T ") + m + "\n")
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
label=sys.argv[1]; log("MEASURE START "+label); battery(label); log("MEASURE END "+label)
