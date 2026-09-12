#!/usr/bin/env python3
"""Unique routed experts per decode step as a function of speculation depth K, from the T5 raw routing arrays.

A DSpark step at depth K verifies the last accepted token + K drafts = K+1 consecutive positions. Every expert any of
them routes to must be fetched (from HBM or Grace). We approximate the verify window with K+1 consecutive *accepted*
decode tokens (upper bound on acceptance — rejected drafts route too, so real windows touch at least this many).
Report per layer: mean unique experts per step vs K, its ratio to K=0 (top_k=6), and the Grace share under the v12
positional offload (last ~24% of rows; T2b offloaded 61.17 of ~252 GiB) vs a usage-chosen cold set of the same size.
"""
import json, numpy as np
z = np.load("/w/results/hist-T5-hist.npz", allow_pickle=True)
meta = z["meta"].item(); meta = json.loads(meta) if isinstance(meta, str) else meta
E = 384; TOPK = 6; GRACE_FRAC = 0.2423
cold_pos = np.arange(int(round(E * (1 - GRACE_FRAC))), E)                    # positional tail = what v12 does
cnt = z["cnt_dec"]; cnt = cnt.reshape(-1, E) if cnt.ndim == 1 else cnt          # [L, E]
L = cnt.shape[0]
cold_use = np.argsort(cnt, axis=1)[:, : len(cold_pos)]                           # per layer coldest by usage
def cold_share(ids_l, cold_l):  # fraction of a set of unique expert ids that live in Grace
    return np.isin(ids_l, cold_l).mean() if len(ids_l) else 0.0
out = {}
for cat_filter in (None, "prose", "shell_ops", "code", "tool_json", "structured"):
    rows = {K: [] for K in (0, 1, 2, 3, 5, 7)}
    grace_pos = {K: [] for K in rows}; grace_use = {K: [] for K in rows}
    for m in meta:
        if cat_filter and m["cat"] != cat_filter: continue
        a = z[f"req{m['i']}"]                       # [T, L, topk]
        dec = a[m["P"]:]                            # decode rows only
        for K in rows:
            w = K + 1
            for s in range(0, len(dec) - w + 1, w):
                win = dec[s:s + w]                  # [w, L, topk]
                per_layer_unique = []
                gp = []; gu = []
                for l in range(L):
                    ids = np.unique(win[:, l, :])
                    per_layer_unique.append(len(ids))
                    gp.append(cold_share(ids, cold_pos)); gu.append(cold_share(ids, cold_use[l]))
                rows[K].append(np.mean(per_layer_unique)); grace_pos[K].append(np.mean(gp)); grace_use[K].append(np.mean(gu))
    res = {}
    for K in rows:
        u = float(np.mean(rows[K])) if rows[K] else float("nan")
        res[K] = {"tokens": K + 1, "unique_experts_per_layer": round(u, 2), "vs_K0": round(u / max(np.mean(rows[0]), 1e-9), 2),
                  "per_token": round(u / (K + 1), 2), "grace_share_positional": round(float(np.mean(grace_pos[K])), 3),
                  "grace_share_usage_cold": round(float(np.mean(grace_use[K])), 3), "n_windows": len(rows[K])}
    out[cat_filter or "ALL"] = res
json.dump(out, open("/w/results/t5_unique_vs_k.json", "w"), indent=1)
for cat, res in out.items():
    print(f"== {cat}")
    print("  K  tok  uniq/layer  vsK0  per-tok  grace(pos)  grace(usage)  n")
    for K, r in res.items():
        print(f"  {K}  {r['tokens']:>3}  {r['unique_experts_per_layer']:>9}  {r['vs_K0']:>4}  {r['per_token']:>6}  {r['grace_share_positional']:>9}  {r['grace_share_usage_cold']:>11}  {r['n_windows']}")
