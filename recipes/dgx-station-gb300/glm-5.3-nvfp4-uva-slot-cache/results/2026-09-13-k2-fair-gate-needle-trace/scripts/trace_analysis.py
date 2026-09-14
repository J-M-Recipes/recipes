#!/usr/bin/env python3
"""trace_analysis.py <trace_dir> <slots.json>
Three questions on the frozen decode routing trace:
 A. static-oracle pin vs LRU at the live per-layer slot budget (how much does explicit allocation buy?)
 B. cross-layer predictability: given layer L's top-8 set at step t, how well does it predict L+1's set?
    (would a "route L, prefetch L+1" scheme have anything to work with?)
 C. temporal predictability: given step t's set at layer L, how much of step t+1's set is the same expert?
    (would MTP-lookahead prefetch of the *previous* step's routing help?)
 D. hybrid: static top-N pin + LRU tail at the same total budget vs pure LRU.
"""
import sys, os, glob, json, numpy as np
from collections import OrderedDict, Counter

d, slots_path = sys.argv[1], sys.argv[2]
slots = json.load(open(slots_path))["per_layer"]
data = []; steps = []
for f in sorted(glob.glob(os.path.join(d, "trace-*.i16"))):
    st = [l.split() for l in open(f.replace(".i16", ".steps"))]
    st = [(float(a), int(b), int(c), int(e)) for a, b, c, e in st]
    L, K = st[0][2], st[0][3]
    arr = np.fromfile(f, dtype=np.int16); n = arr.size // (L * K)
    data.append(arr[: n * L * K].reshape(n, L, K)); steps.extend(st)
X = np.concatenate(data); ntag = sum(nt for _, nt, _, _ in steps); X = X[:ntag]
tags = np.concatenate([np.full(nt, 0 if nt <= 8 else 1, dtype=np.int8) for _, nt, _, _ in steps])[: X.shape[0]]
Xd = X[tags == 0]
moe = [l for l in range(X.shape[1]) if (X[:, l, :] >= 0).any() and len(np.unique(Xd[:, l, :])) > 8]
E = int(Xd.max()) + 1
print(f"decode tokens={Xd.shape[0]} moe_layers={len(moe)} experts={E} topk={K}")

def lru_hit(Y, cap):
    cache = OrderedDict(); hits = tot = 0
    for row in Y:
        for e in row:
            if e < 0: continue
            tot += 1
            if e in cache: cache.move_to_end(e); hits += 1
            else:
                cache[e] = 1
                if len(cache) > cap: cache.popitem(last=False)
    return hits / tot

def hybrid_hit(Y, cap, pinned):
    """pinned set always resident; LRU over the remaining cap-len(pinned) slots."""
    pin = set(pinned); lcap = cap - len(pin)
    cache = OrderedDict(); hits = tot = 0
    for row in Y:
        for e in row:
            if e < 0: continue
            tot += 1
            if e in pin: hits += 1; continue
            if e in cache: cache.move_to_end(e); hits += 1
            else:
                cache[e] = 1
                if lcap > 0 and len(cache) > lcap: cache.popitem(last=False)
                elif lcap <= 0: cache.clear()
    return hits / tot

sub = max(1, Xd.shape[0] // 8000)
rows = {"layer": [], "S": [], "static": [], "lru": [], "hyb50": [], "hyb75": [], "xlayer": [], "temporal": []}
for l in moe:
    S = int(slots[str(l)])
    Y = Xd[:, l, :]; Ys = Y[::sub]
    cnt = Counter(Y[Y >= 0].ravel().tolist()); tot = sum(cnt.values())
    order = [e for e, _ in cnt.most_common()]
    static = sum(cnt[e] for e in order[:S]) / tot
    lru = lru_hit(Ys, S)
    h50 = hybrid_hit(Ys, S, order[: S // 2]); h75 = hybrid_hit(Ys, S, order[: int(S * 0.75)])
    # cross-layer: fraction of L+1's experts at step t that also appear in L's set at step t (same expert id)
    nxt = moe[moe.index(l) + 1] if moe.index(l) + 1 < len(moe) else None
    xl = np.nan
    if nxt is not None:
        Z = Xd[:, nxt, :]
        xl = np.mean([len(set(a[a >= 0]) & set(b[b >= 0])) / max(1, (b >= 0).sum()) for a, b in zip(Y[::sub], Z[::sub])])
    # temporal: fraction of step t+1's experts that were in step t's set
    tm = np.mean([len(set(a[a >= 0]) & set(b[b >= 0])) / max(1, (b >= 0).sum()) for a, b in zip(Y[:-1:sub], Y[1::sub])])
    for k, v in zip(rows, (l, S, static, lru, h50, h75, xl, tm)): rows[k].append(v)

def m(k): return float(np.nanmean(rows[k]))
print("\nPER-LAYER (first 12 + last 4):")
print(f"{'L':>3} {'S':>4} {'static':>7} {'lru':>6} {'hyb50':>6} {'hyb75':>6} {'xlayer':>7} {'temporal':>8}")
idx = list(range(len(moe)))[:12] + list(range(len(moe)))[-4:]
for i in idx:
    print(f"{rows['layer'][i]:>3} {rows['S'][i]:>4} {rows['static'][i]:>7.3f} {rows['lru'][i]:>6.3f} {rows['hyb50'][i]:>6.3f} {rows['hyb75'][i]:>6.3f} {rows['xlayer'][i]:>7.3f} {rows['temporal'][i]:>8.3f}")
summary = {"decode_tokens": int(Xd.shape[0]), "moe_layers": len(moe), "total_slots": int(sum(rows["S"])),
           "mean_static_oracle": m("static"), "mean_lru": m("lru"), "mean_hybrid_pin50": m("hyb50"), "mean_hybrid_pin75": m("hyb75"),
           "mean_crosslayer_overlap": m("xlayer"), "mean_temporal_overlap": m("temporal"), "random_overlap_baseline": K / E,
           "misses_per_layer_step": {"static": K * (1 - m("static")), "lru": K * (1 - m("lru")), "hyb75": K * (1 - m("hyb75"))}}
print("\nSUMMARY", json.dumps(summary, indent=1))
json.dump({"summary": summary, "per_layer": rows}, open(os.path.join(os.path.dirname(slots_path), "trace_analysis.json"), "w"), indent=1, default=float)
