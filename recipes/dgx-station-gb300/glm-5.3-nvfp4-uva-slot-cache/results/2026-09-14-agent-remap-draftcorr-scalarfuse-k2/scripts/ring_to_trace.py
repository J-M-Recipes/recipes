#!/usr/bin/env python3
"""ring_to_trace.py ID_RING.pt <out_dir>  — convert the ID_RING dump into alloc_slots.py trace format
(trace-<pid>.i16 [n, L, K] int16 + trace-<pid>.steps 'time ntok L K'), rows n0..n only (the marked corpus).
Rows with any -1 in a MoE layer (partial step) are dropped. Prints coverage stats."""
import sys, os, torch, numpy as np
d = torch.load(sys.argv[1]); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
ids = d["ids"].numpy()            # [L, ring, 8] int16
L, R, K = ids.shape; n, n0 = d["n"], d["n0"]
assert n - n0 <= R, "ring wrapped; shrink corpus or grow ring"
rows = [(n0 + i) % R for i in range(n - n0)]
X = np.transpose(ids[:, rows, :], (1, 0, 2))      # [n, L, K]
moe = [l for l in range(L) if (X[:, l, :] >= 0).any()]
full = np.all(X[:, moe, :] >= 0, axis=(1, 2))
X = X[full]
print(f"rows marked={n-n0} complete={X.shape[0]} moe_layers_seen={len(moe)} L={L} K={K}")
X.astype(np.int16).tofile(os.path.join(out, "trace-agent.i16"))
with open(os.path.join(out, "trace-agent.steps"), "w") as f:
    for i in range(X.shape[0]): f.write(f"{i:.3f} 1 {L} {K}\n")
# quick per-layer distinct-expert count as sanity
for l in (3, 12, 40, 77):
    if l in moe: print(f"layer {l}: distinct experts={len(np.unique(X[:, l, :]))}")
