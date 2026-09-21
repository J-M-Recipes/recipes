#!/usr/bin/env python3
"""tf_split.py — per-class split of a tf_logprob compare JSON (the split OPEN-FINDINGS says must always be shown).
Usage: python3 tf_split.py tf-compare-<ref>-vs-<cand>.json [...]
Classes by doc id prefix: parity-agent-* = agent/tool, parity-heldout-* = heldout, *-10k-* = prose."""
import json, sys

def cls(k):
    if k.startswith("parity-agent"): return "agent_tool"
    if k.startswith("parity-heldout"): return "heldout"
    if "-10k-" in k: return "prose_10k"
    return "other"

for fn in sys.argv[1:]:
    j = json.load(open(fn)); per = j["per_doc"]; agg = {}
    for k, v in per.items():
        c = cls(k); a = agg.setdefault(c, {"docs": 0, "positions": 0, "sum_abs": 0.0, "flips": 0})
        a["docs"] += 1; a["positions"] += v["positions"]; a["sum_abs"] += v["mean_abs_dlp"] * v["positions"]; a["flips"] += v["top1_flips"]
    print(f"{j['ref']} -> {j['cand']}  (corpus: {j['positions']} pos, flips {j['top1_flip_rate']*100:.2f}%, mean|Δ| {j['mean_abs_dlogprob']:.4f})")
    for c, a in sorted(agg.items()):
        print(f"  {c:10s} docs={a['docs']:2d} pos={a['positions']:6d} mean|Δlp|={a['sum_abs']/a['positions']:.4f} top1_flips={a['flips']/a['positions']*100:.2f}%")
