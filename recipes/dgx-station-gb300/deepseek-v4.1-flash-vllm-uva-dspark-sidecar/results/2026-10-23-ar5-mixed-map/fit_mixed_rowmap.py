#!/usr/bin/env python3
"""Fit a mixed-traffic static rowmap (295 hot / 89 cold per layer) from the AR1+AR2
cumulative decode counts. Experts that stay hot keep their rows; promoted experts take
the vacated rows. Ties (equal counts) prefer the existing static-map membership.

Usage: fit_mixed_rowmap.py <counts.npz> <rowmap-static-v1.json> <out.json>
"""
import json
import sys

import numpy as np

z = np.load(sys.argv[1])
cnt = z["cnt"].astype(np.int64)  # [40, 384] cumulative routed decode lanes since boot
static = json.load(open(sys.argv[2]))
L, E = cnt.shape
out_layers = {}
tot = cold_static = cold_mixed = 0
moved = []
for l in range(L):
    s = static["layers"][str(l)]
    hot_s, cold_s = list(s["hot"]), list(s["cold"])
    n_hot = len(hot_s)
    in_static_hot = np.zeros(E, dtype=bool)
    in_static_hot[hot_s] = True
    # sort by count desc, then static-hot first, then id
    order = sorted(range(E), key=lambda e: (-int(cnt[l, e]), not in_static_hot[e], e))
    hot_new_set = set(order[:n_hot])
    stay = [e for e in hot_s if e in hot_new_set]
    promote = [e for e in order[:n_hot] if not in_static_hot[e]]
    vacated = [r for r, e in enumerate(hot_s) if e not in hot_new_set]
    hot_new = list(hot_s)
    for r, e in zip(vacated, promote):
        hot_new[r] = e
    demote = [hot_s[r] for r in vacated]
    cold_new = list(cold_s)
    pos = {e: r for r, e in enumerate(cold_s)}
    for e_in, e_out in zip(promote, demote):
        cold_new[pos[e_in]] = e_out
    assert len(set(hot_new)) == n_hot and set(hot_new) | set(cold_new) == set(range(E))
    assert not (set(hot_new) & set(cold_new))
    out_layers[str(l)] = {"hot": hot_new, "cold": cold_new}
    t = int(cnt[l].sum())
    tot += t
    cold_static += int(cnt[l, cold_s].sum())
    cold_mixed += int(cnt[l, cold_new].sum())
    moved.append(len(promote))
summary = {
    "hot_n": n_hot, "cold_n": E - n_hot, "L_main": L,
    "source": f"AR1+AR2 cumulative decode counts {sys.argv[1]} (v26ar boot 2026-10-23: A/B agent, parity agent/tools, "
              "shift frozen+adaptive JP/DE+agent); fitted top-295 per layer, ties -> static membership",
    "decode_selections_main": tot,
    "in_sample_cold_share_static": round(cold_static / tot, 5),
    "in_sample_cold_share_mixed": round(cold_mixed / tot, 5),
    "experts_moved_per_layer": {"mean": round(float(np.mean(moved)), 2), "min": int(min(moved)), "max": int(max(moved)),
                                "total": int(sum(moved))},
}
json.dump({"layers": out_layers, "summary": summary}, open(sys.argv[3], "w"), indent=1)
print(json.dumps(summary, indent=1))
