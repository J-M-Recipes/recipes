#!/usr/bin/env python3
"""Scheduler occupancy from each arm's server.log (vLLM 10 s engine stats lines)."""
import collections, json, re, sys
from pathlib import Path
raw = Path(sys.argv[1]); out = {}
for a in ["JM_K4", "INCOAI_K4", "JM_K7", "INCOAI_K7", "INCOAI_bf58_K4"]:
    t = (raw / a / "boot/server.log").read_text(errors="replace")
    run = [int(x) for x in re.findall(r"Running: (\d+) reqs", t)]
    wait = [int(x) for x in re.findall(r"Waiting: (\d+) reqs", t)]
    kv = [float(x) for x in re.findall(r"GPU KV cache usage: ([0-9.]+)%", t)]
    c = collections.Counter(run)
    out[a] = {"max_running": max(run) if run else None, "running_hist_top": c.most_common(4),
              "max_waiting": max(wait) if wait else None, "max_kv_usage_pct": max(kv) if kv else None}
    print(a, json.dumps(out[a]))
(raw.parent / "sched-stats.json").write_text(json.dumps(out, indent=2) + "\n")
