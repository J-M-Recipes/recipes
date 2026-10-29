import glob
import json
import os
import subprocess

R = "/home/milo/dsv41/m3/tr2/results-cardTR2"
ARMS = {"A": "dsv41-vllm-1017-TR2-A", "B": "dsv41-vllm-1017-TR2-B",
        "A2": "dsv41-vllm-1017-TR2-A2", "BV": "dsv41-vllm-1017-TR2-BV"}


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def last_stats(container):
    out = sh("docker logs %s 2>&1 | tr '\\r' '\\n' | grep 'L2 stats' | tail -1" % container)
    return out.strip()


def hash_hit(container):
    out = sh("docker logs %s 2>&1 | tr '\\r' '\\n' | grep -cE 'Config cache hit|Loaded [0-9]+ configs'" % container)
    try:
        return 1 if int(out.strip()) > 0 else 0
    except ValueError:
        return 0


for arm, cname in ARMS.items():
    d = os.path.join(R, arm)
    os.makedirs(d, exist_ok=True)
    line = last_stats(cname)
    with open(os.path.join(d, "l2-boot-stats.txt"), "w") as fh:
        fh.write(line + "\n")
    fl = os.path.join(d, "facts.json")
    if os.path.exists(fl):
        try:
            f = json.load(open(fl))
        except Exception:
            f = {}
    else:
        f = {"arm": arm, "name": cname}
    f["hash_hit"] = hash_hit(cname)
    f["hash_hit_note"] = "recomputed from the container log after the card"
    json.dump(f, open(fl, "w"), indent=1)
    print(arm, "stats:", line[:150])
    print(arm, "hash_hit:", f["hash_hit"])

subprocess.run(["python3", "/home/milo/dsv41/m3/tr2/tr2_verdict.py", R])