#!/usr/bin/env python3
"""fit_l39static.py <mixed.json> <static.json> <out.json> — v25m2: mixed-traffic map with layer 39 restored to the static split.
Why: Card GV's only failure was the bring-up cosine gate at layer 39 (min 0.99811 vs 0.999); layer 39 T=128 is the
worst check on every map. Layers 0-38 keep the mixed fit; layer 39 takes rowmap-static-v1's hot/cold lists verbatim."""
import json
import sys

mx, st, out = sys.argv[1:4]
a, b = json.load(open(mx)), json.load(open(st))
L = dict(a["layers"])
L["39"] = {"hot": list(b["layers"]["39"]["hot"]), "cold": list(b["layers"]["39"]["cold"])}
for k, v in L.items():
    assert len(v["hot"]) == 295 and len(v["cold"]) == 89 and not set(v["hot"]) & set(v["cold"]), k
    assert sorted(v["hot"] + v["cold"]) == list(range(384)), k
s = dict(a["summary"])
s["source"] = s["source"] + " | v2-l39s 2026-10-24: layer 39 = rowmap-static-v1 (Card GV cosine gate failed at L39 only)"
s["layer39"] = "static-v1"
json.dump({"layers": L, "summary": s}, open(out, "w"))
print("wrote", out, "L39 hot overlap with mixed:", len(set(a["layers"]["39"]["hot"]) & set(L["39"]["hot"])))
