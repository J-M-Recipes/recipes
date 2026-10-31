#!/usr/bin/env python3
"""Compare two k3pod_probe.py outputs: greedy text, and per-position logprob agreement.
Usage: k3pod_compare.py REF.json POD.json"""
import json, math, sys

ref, pod = (json.load(open(p)) for p in sys.argv[1:3])
worst = 0.0; n = 0; top1 = 0; diffs = []
for a, b in zip(ref, pod):
    same = a["text"] == b["text"]
    print(f"{'SAME' if same else 'DIFF'} text  {a['prompt'][:40]!r}")
    pa, pb = a.get("prompt_logprobs") or [], b.get("prompt_logprobs") or []
    seqs = [(x, y) for x, y in zip(pa, pb) if x and y]
    seqs += list(zip(a["top_logprobs"], b["top_logprobs"]))
    for x, y in seqs:
        # x, y: {token_id_str: {logprob, rank, decoded}} (prompt) or {token: logprob} (completion)
        fx = {k: (v["logprob"] if isinstance(v, dict) else v) for k, v in x.items()}
        fy = {k: (v["logprob"] if isinstance(v, dict) else v) for k, v in y.items()}
        common = set(fx) & set(fy)
        for k in common:
            d = abs(fx[k] - fy[k]); diffs.append(d); worst = max(worst, d)
        n += 1
        top1 += max(fx, key=fx.get) == max(fy, key=fy.get)
diffs.sort()
print(f"positions {n}, top-1 agreement {top1}/{n}")
print(f"|dlogprob| over shared top-20 entries: median {diffs[len(diffs)//2]:.5f}  p99 {diffs[int(len(diffs)*.99)]:.5f}  max {worst:.5f}  (n={len(diffs)})")
