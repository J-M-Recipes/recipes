#!/usr/bin/env python3
"""draft_corr.py ID_RING.pt [slots.json] -- does the MTP-block (layer-78) draft routing at step t predict
main-layer routing at step t+1?  Rows n0..n.  For each main MoE layer L:
overlap(draft_t, main_L_{t+1}) vs chance (8/256) vs temporal(main_L_t, main_L_{t+1}) vs same-step(draft_t, main_L_t).
Also: fraction of step-t+1 LRU misses (per-layer LRU replay at the live slot map) that draft_t would have named."""
import sys, torch, numpy as np, json
from collections import OrderedDict
d = torch.load(sys.argv[1]); ids = d["ids"].numpy(); dr = d["draft"].numpy(); n, n0, R = d["n"], d["n0"], d["ring"]
rows = [(n0 + i) % R for i in range(n - n0)]
X = np.transpose(ids[:, rows, :], (1, 0, 2)); D = dr[rows]
moe = [l for l in range(X.shape[1]) if (X[:, l, :] >= 0).any()]
ok = np.all(X[:, moe, :] >= 0, axis=(1, 2)) & np.all(D >= 0, axis=1)
X, D = X[ok], D[ok]; T = X.shape[0]
print(f"steps={T} (rows with valid draft {int(ok.sum())}/{len(ok)}) moe_layers={len(moe)}")
def ov(a, b): return np.mean([len(set(a[i]) & set(b[i])) / 8 for i in range(len(a))])
res = {}
for l in moe:
    m = X[:, l, :]
    res[l] = dict(draft_next=ov(D[:-1], m[1:]), draft_same=ov(D, m), temporal=ov(m[:-1], m[1:]))
arr = lambda k: np.array([res[l][k] for l in moe])
print(f"chance={8/256:.4f}")
for k in ("draft_next", "draft_same", "temporal"):
    a = arr(k); print(f"{k:11s} mean={a.mean():.4f} min={a.min():.4f} max={a.max():.4f}  layers>2x chance: {int((a>2*8/256).sum())}/{len(moe)}")
slots = json.load(open(sys.argv[2]))["per_layer"] if len(sys.argv) > 2 else {}
cov_num = cov_den = 0
for l in moe:
    S = int(slots.get(str(l), 96)); cache = OrderedDict(); m = X[:, l, :]
    for t in range(T):
        misses = set()
        for e in m[t]:
            e = int(e)
            if e in cache: cache.move_to_end(e)
            else:
                misses.add(e); cache[e] = 1
                if len(cache) > S: cache.popitem(last=False)
        if t > 0 and misses:
            cov_num += len(misses & set(int(x) for x in D[t-1])); cov_den += len(misses)
print(f"LRU@map: draft_t names {cov_num}/{cov_den} = {cov_num/max(1,cov_den):.4f} of step-t+1 misses (perfect lookahead = 1.0)")
json.dump({str(l): res[l] for l in moe}, open("draft_corr_per_layer.json", "w"), indent=1)
