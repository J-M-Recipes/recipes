#!/usr/bin/env python3
"""Concurrent decode: C streams at once, each a distinct short prompt + N forced tokens (ignore_eos, temperature 0).
Reports aggregate tok/s and per-stream inter-token gap (median / p90) over the steady part of each stream.
Usage: conc_bench.py PORT N_TOKENS C1,C2,... [OUT.json]"""
import json
import statistics
import sys
import threading
import time
import urllib.request

port, n_tok = sys.argv[1], int(sys.argv[2])
cs = [int(c) for c in sys.argv[3].split(",")]
out_path = sys.argv[4] if len(sys.argv) > 4 else None
url = f"http://127.0.0.1:{port}/v1/completions"
topics = ["the transistor", "the steam engine", "the printing press", "radio", "the jet engine", "the laser",
          "the integrated circuit", "the internet"]


def stream(i, res):
    b = json.dumps({"model": "k3", "prompt": f"Write a long, detailed history of {topics[i % len(topics)]}.",
                    "max_tokens": n_tok, "temperature": 0, "ignore_eos": True, "stream": True}).encode()
    ts = []
    with urllib.request.urlopen(urllib.request.Request(url, b, {"Content-Type": "application/json"}),
                                timeout=900) as r:
        for line in r:
            if line.startswith(b"data: ") and not line.startswith(b"data: [DONE]"):
                ts.append(time.perf_counter())
    res[i] = ts


report = []
for c in cs:
    res = [None] * c
    th = [threading.Thread(target=stream, args=(i, res)) for i in range(c)]
    t0 = time.perf_counter()
    for t in th:
        t.start()
    for t in th:
        t.join()
    wall = time.perf_counter() - t0
    ntok = sum(len(r) for r in res)
    gaps = []
    for r in res:
        g = [(b - a) * 1e3 for a, b in zip(r[8:], r[9:])]        # skip prefill + first steps
        gaps += g
    gaps.sort()
    row = {"C": c, "tokens": ntok, "wall_s": round(wall, 2), "agg_tok_s": round(ntok / wall, 2),
           "gap_ms_median": round(statistics.median(gaps), 2), "gap_ms_p90": round(gaps[int(0.9 * len(gaps))], 2)}
    report.append(row)
    print(json.dumps(row), flush=True)
if out_path:
    json.dump(report, open(out_path, "w"), indent=1)
