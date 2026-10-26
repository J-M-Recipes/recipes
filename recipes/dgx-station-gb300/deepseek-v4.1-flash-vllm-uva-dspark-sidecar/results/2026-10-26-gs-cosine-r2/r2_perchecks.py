#!/usr/bin/env python3
"""Per-check dump for the R2 re-judge (report-only helper; uses judge_r2.py's own regex + scope, imported unmodified)."""
import csv, sys, importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location("j", sys.argv[1]); j = importlib.util.module_from_spec(spec); spec.loader.exec_module(j)
out = Path(sys.argv[2])
for arg in sys.argv[3:]:
    label, log = arg.split("=", 1)
    t = Path(log).read_text(errors="replace"); i = t.find("Application startup complete")
    rows = [dict(L=int(a), T=int(b), rows=int(c), rel=float(d), cos=float(e), share=float(f)) for a, b, c, d, e, f in j.RX.findall(t[i:])]
    g = [r for r in rows if r["rows"] >= 32]
    for r in g: r["e"] = r["rel"] * r["share"]
    with open(out / f"perchecks-{label}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["idx", "L", "T", "rows", "rel", "cos", "share", "e"]); w.writeheader()
        for k, r in enumerate(g): w.writerow(dict(idx=k, **r))
    print(f"== {label}: {len(g)} gated checks of {len(rows)} post-startup")
    print(" top-5 by e:", [(r["L"], r["T"], r["rows"], r["cos"], r["share"], round(r["e"], 5)) for r in sorted(g, key=lambda r: -r["e"])[:5]])
    print(" bottom-5 by cos:", [(r["L"], r["T"], r["rows"], r["cos"], r["share"], round(r["e"], 5)) for r in sorted(g, key=lambda r: r["cos"])[:5]])
    print(" below 0.999:", sum(r["cos"] < 0.999 for r in g))
    per = {}
    for r in g: per.setdefault(r["L"], []).append(r)
    print(" per-layer (L n min_cos max_e):", " ".join(f"{L}:{len(v)}:{min(x['cos'] for x in v):.5f}:{max(x['e'] for x in v):.5f}" for L, v in sorted(per.items())))
