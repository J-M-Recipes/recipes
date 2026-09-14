#!/usr/bin/env python3
"""live_layer_skew.py — per-layer cache-hit skew of the LIVE K=2 gate container
under real agent traffic (Milo's own research session, 2026-09-13).

Reads the gzip'd engine log (SLOT_CACHE STATS windows) and the per-layer slot
budget JSON, then answers: do the worst-hit layers coincide with the smallest
per-layer slot budgets (budget mismatch), or is it workload routing skew
(map-fit mismatch)?

Usage: python3 live_layer_skew.py <k2gate.log.gz> <slots-7360-ctx256k.json>
Reproduces the numbers cited in RESEARCH-REPORT.md section 5 (124 windows,
layer 12 worst in 58, layer 40 best in 67, Spearman -0.171/-0.179).
"""
import collections
import gzip
import json
import re
import sys


def main(log_path, slots_path):
    pat = re.compile(r"SLOT_CACHE STATS window.*?best=(\d+):([\d.]+) worst=(\d+):([\d.]+)")
    best_c, worst_c, hits, n = collections.Counter(), collections.Counter(), [], 0
    with gzip.open(log_path, "rt", errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if m:
                n += 1
                best_c[int(m.group(1))] += 1
                worst_c[int(m.group(3))] += 1
                hits.append(float(m.group(4)))

    slots = json.load(open(slots_path))
    per_layer = {int(k): v for k, v in slots["per_layer"].items()}

    print(f"windows parsed: {n}, mean worst-layer hit: {sum(hits)/len(hits):.3f}")
    print("\nMost-often-WORST layers (live agent traffic) vs slot budget:")
    for lyr, cnt in worst_c.most_common(8):
        print(f"  layer {lyr:3d}: worst in {cnt:3d} windows, slots={per_layer.get(lyr)}")
    print("\nMost-often-BEST layers vs slot budget:")
    for lyr, cnt in best_c.most_common(8):
        print(f"  layer {lyr:3d}: best in {cnt:3d} windows, slots={per_layer.get(lyr)}")

    layers = sorted(per_layer)
    w = [worst_c.get(l, 0) for l in layers]
    b = [best_c.get(l, 0) for l in layers]
    s = [per_layer[l] for l in layers]

    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for i, idx in enumerate(order):
            r[idx] = i
        return r

    rw, rs, rb = rank(w), rank(s), rank(b)
    n = len(layers)
    sp_w = 1 - 6 * sum((a - c) ** 2 for a, c in zip(rw, rs)) / (n * (n ** 2 - 1))
    sp_b = 1 - 6 * sum((a - c) ** 2 for a, c in zip(rb, rs)) / (n * (n ** 2 - 1))
    print(f"\nSpearman rank corr (n={n}): worst-frequency vs slot-budget = {sp_w:+.3f}, "
          f"best-frequency vs budget = {sp_b:+.3f}")
    print("(near-zero = budget size is not the driver -> workload routing skew / map-fit mismatch)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
