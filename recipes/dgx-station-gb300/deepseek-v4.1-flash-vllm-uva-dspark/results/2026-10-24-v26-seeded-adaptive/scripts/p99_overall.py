#!/usr/bin/env python3
"""p99_overall.py — whole-boot host-step interval percentiles from the hook v26ar.2 step ring (desk-side, 2026-10-25).

Reads cardAR6/<boot>/steps.bin(.gz) and swaps.jsonl(.gz) with the same parser and swap attribution as ar6_verdict.py G2
(consecutive host steps only, 0 < dt <= 250 ms; a step is swap-bearing if the ring row reports n_applied > 0 or a
swaps.jsonl apply_ns falls inside it). Prints, per boot: step count, swap-bearing share, p99 of all steps, p99 of the
swap-free steps alone, and p99 with each swap-bearing step replaced by its 5 s bin's swap-free median (counterfactual).

Usage (from the bundle root): python3 scripts/p99_overall.py [boot0 bootA]
This is a host-step interval (one engine step = one DSpark verify window), not a client inter-token gap.
"""
import bisect
import gzip
import json
import math
import statistics
import sys
from array import array
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "cardAR6"


def _read(p: Path) -> bytes:
    if p.exists():
        return p.read_bytes()
    gz = p.with_name(p.name + ".gz")
    return gzip.decompress(gz.read_bytes())


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(math.ceil(q * len(xs))) - 1))]


def steps(d: Path):
    raw = _read(d / "steps.bin")
    raw = raw[: len(raw) - len(raw) % 32]
    a = array("q")
    a.frombytes(raw)
    if sys.byteorder != "little":
        a.byteswap()
    rows, offs = [], []
    for i in range(0, len(a), 4):
        hs, t, x, y = a[i], a[i + 1], a[i + 2], a[i + 3]
        if hs >= 0:
            rows.append((hs, t, x, y))
        elif hs == -2:
            offs.append(x - t)
    return rows, statistics.median(offs)


def main(boots):
    for b in boots:
        d = BASE / b
        rows, off = steps(d)
        swaps = [json.loads(l) for l in _read(d / "swaps.jsonl").decode().splitlines() if l.strip()]
        apply = sorted(int(s["apply_ns"]) for s in swaps if s.get("apply_ns") is not None)
        bins = {}
        for k in range(len(rows) - 1):
            if rows[k + 1][0] != rows[k][0] + 1:
                continue
            dt = (rows[k + 1][1] - rows[k][1]) / 1e6
            if dt <= 0 or dt > 250:
                continue
            n = max(rows[k][2], bisect.bisect_left(apply, rows[k + 1][1]) - bisect.bisect_left(apply, rows[k][1]))
            bins.setdefault(int(((rows[k][1] + off) / 1e9) // 5), []).append((dt, n))
        alld, nond, cf, nsw = [], [], [], 0
        for v in bins.values():
            non = [x for x, n in v if n == 0]
            med = statistics.median(non) if non else None
            for x, n in v:
                alld.append(x)
                if n == 0:
                    nond.append(x)
                    cf.append(x)
                else:
                    nsw += 1
                    cf.append(med if med is not None else x)
        print(f"{b}: host steps {len(alld)}  swap-bearing {nsw} ({100 * nsw / len(alld):.2f}%)  "
              f"p99 all {pct(alld, .99):.2f} ms  p99 swap-free only {pct(nond, .99):.2f} ms  "
              f"p99 counterfactual {pct(cf, .99):.2f} ms  p50 all {pct(alld, .50):.2f} ms")


if __name__ == "__main__":
    main(sys.argv[1:] or ["boot0", "bootA"])
