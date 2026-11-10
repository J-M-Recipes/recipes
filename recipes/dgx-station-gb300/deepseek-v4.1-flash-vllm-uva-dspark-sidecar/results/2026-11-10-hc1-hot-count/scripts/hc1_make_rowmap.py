#!/usr/bin/env python3
"""Card HC1: build a Garner-format rowmap (layers[L] = {hot, cold}, both sorted) with N hot experts per layer from
Garner's h255 rowmap plus one of our hot maps (layer -> list of hot expert ids, Garner's 255 as a subset).
Usage: hc1_make_rowmap.py <garner-h255.json> <our-hot-map.json> <out.json>"""
import json
import sys

EXPERTS = 384
src, ours, out = sys.argv[1:4]
g = json.load(open(src))
h = json.load(open(ours))
L = g["layers"]
assert len(L) == 40 and set(L) == set(h), "layer sets differ"
n = {len(v) for v in h.values()}
assert len(n) == 1, n
n_hot = n.pop()
doc = {"layers": {}, "meta": dict(g["meta"])}
for k in sorted(L, key=int):
    gh, gc = L[k]["hot"], L[k]["cold"]
    assert set(L[k]) == {"hot", "cold"} and gh == sorted(gh) and gc == sorted(gc)
    hot = sorted(h[k])
    assert len(set(hot)) == n_hot and set(gh) <= set(hot), f"layer {k}: Garner's hot set is not a subset"
    cold = [e for e in gc if e not in set(hot)]
    assert sorted(hot + cold) == list(range(EXPERTS)), f"layer {k}: not a partition"
    doc["layers"][k] = {"hot": hot, "cold": cold}
doc["meta"]["n_hot"] = n_hot
doc["meta"]["hc1"] = f"Garner's h255 hot set kept; {n_hot - 255} experts per layer promoted from its cold list (card HC1)"
json.dump(doc, open(out, "w"))
print(f"rowmap {out}: 40 layers, hot {n_hot}, cold {EXPERTS - n_hot}")
