#!/usr/bin/env python3
"""Full-model per-layer check of the K3 pod against an independent fp32 reference.

Inputs: chunks written by k3pod.py in record mode (K3POD_DUMP_DIR): per (layer, decode token) the MoE input x,
16 global ids, weights, the Station's share and the workers' share, from a real full-model run.
Reference: y = sum_j w_j * expert_j(x), fp32, dequantized NVFP4 weights straight from the checkpoint, SiTU,
unquantized activation (ref_expert.py). Split by owner (plan): Station share vs its experts, workers' share vs
theirs, and the total. A wrong / missing / double-counted expert shows up as cosine far below the ~0.99x that
activation quantization alone produces.

Usage (vLLM container, GPU): pod_ref_check.py DUMP_DIR PLAN OUT.json [max_tokens_per_layer]
"""
import glob, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, "/situ")
from ref_expert import dequantize_nvfp4, situ  # noqa: E402

DUMP, PLANP, OUT = sys.argv[1:4]
MAXT = int(sys.argv[4]) if len(sys.argv) > 4 else 10_000
LO = int(sys.argv[5]) if len(sys.argv) > 5 else 0          # layer range [LO, HI] (fresh process per batch:
HI = int(sys.argv[6]) if len(sys.argv) > 6 else 999        #  one long run was OOM-killed after ~36 layers)
CK = "/ckpt"
dev = "cuda"
idx = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]
plan = json.load(open(PLANP))

# owner: 0 station, 1 remote
remote = {}
for t, v in plan["targets"].items():
    for L, ids in v["layers"].items():
        remote.setdefault(int(L), set()).update(ids)

# ---- load dump
rows = {}
for f in sorted(glob.glob(os.path.join(DUMP, "chunk*.npz"))):
    d = np.load(f)
    lay, x, ids, w, st, rm = (d[k] for k in ("layer", "x", "ids", "w", "station", "remote"))   # load each array once
    for i in range(len(lay)):
        if not (LO <= int(lay[i]) <= HI):
            continue
        rows.setdefault(int(lay[i]), []).append((x[i], ids[i], w[i], st[i], rm[i]))
print("layers", len(rows), "layer-tokens", sum(len(v) for v in rows.values()), flush=True)

_handles = {}


def handle(shard):
    h = _handles.get(shard)
    if h is None:
        h = _handles[shard] = safe_open(f"{CK}/{shard}", "pt")
    return h


def load_expert(L, e):
    base = f"language_model.model.layers.{L}.block_sparse_moe.experts.{e}."
    t = {}
    for p in ("w1", "w2", "w3"):
        for s in ("weight", "weight_scale", "weight_scale_2"):
            k = base + p + "." + s
            t[p + "." + s] = handle(idx[k]).get_tensor(k)
    return e, t


def deq(t):
    W = {}
    for p in ("w1", "w2", "w3"):
        W[p] = dequantize_nvfp4(t[p + ".weight"].to(dev), t[p + ".weight_scale"].to(dev),
                                t[p + ".weight_scale_2"].to(dev).float().reshape(()))
    return W


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a, b, dim=-1).min()), \
        float(torch.nn.functional.cosine_similarity(a, b, dim=-1).mean())


def rel(a, b):
    r = (a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-12)
    return float(r.mean()), float(r.max())


res = json.load(open(OUT)) if os.path.exists(OUT) else {}
res = {int(k): v for k, v in res.items()}
pool = ThreadPoolExecutor(16)
t_all = time.time()
for L in sorted(rows):
    t0 = time.time()
    R = rows[L][:MAXT]
    X = torch.tensor(np.stack([r[0] for r in R]), device=dev)
    IDS = np.stack([r[1] for r in R]); Wt = torch.tensor(np.stack([r[2] for r in R]), device=dev)
    ST = torch.tensor(np.stack([r[3] for r in R]), device=dev); RM = torch.tensor(np.stack([r[4] for r in R]), device=dev)
    need = sorted({int(e) for e in IDS.ravel() if 0 <= e < 896})
    ref_st = torch.zeros_like(ST); ref_rm = torch.zeros_like(RM)
    rem = remote.get(L, set())
    for e, t in pool.map(lambda e: load_expert(L, e), need):
        W = deq(t)
        hit = torch.tensor(IDS == e, device=dev)             # [n, 16]
        rows_hit = hit.any(1)
        if not bool(rows_hit.any()):
            continue
        xs = X[rows_hit]
        h = situ(xs @ W["w1"].T, xs @ W["w3"].T)
        y = h @ W["w2"].T                                     # [m, H]
        wsel = (Wt[rows_hit] * hit[rows_hit].float()).sum(1, keepdim=True)
        tgt = ref_rm if e in rem else ref_st
        tgt[rows_hit] += wsel * y
    tot_pod, tot_ref = ST + RM, ref_st + ref_rm
    has_st = ref_st.norm(dim=-1) > 0; has_rm = ref_rm.norm(dim=-1) > 0
    # tokens with no Station (or no remote) expert must show an exactly-zero share
    leak_st = float(ST[~has_st].abs().max()) if bool((~has_st).any()) else 0.0
    leak_rm = float(RM[~has_rm].abs().max()) if bool((~has_rm).any()) else 0.0
    r = {"n": len(R), "experts_read": len(need),
         "total_cos_min_mean": cos(tot_pod, tot_ref), "total_rel_mean_max": rel(tot_pod, tot_ref),
         "station_cos_min_mean": cos(ST[has_st], ref_st[has_st]) if bool(has_st.any()) else None,
         "remote_cos_min_mean": cos(RM[has_rm], ref_rm[has_rm]) if bool(has_rm.any()) else None,
         "leak_station_max": leak_st, "leak_remote_max": leak_rm,
         "scale_pod_over_ref": float((tot_pod * tot_ref).sum() / (tot_ref * tot_ref).sum())}
    res[L] = r
    print(f"L{L:2d} n={r['n']} experts={r['experts_read']} total cos min/mean {r['total_cos_min_mean'][0]:.5f}/"
          f"{r['total_cos_min_mean'][1]:.5f} station {r['station_cos_min_mean'][1] if r['station_cos_min_mean'] else float('nan'):.5f} "
          f"remote {r['remote_cos_min_mean'][1] if r['remote_cos_min_mean'] else float('nan'):.5f} "
          f"rel {r['total_rel_mean_max'][0]:.4f} scale {r['scale_pod_over_ref']:.4f} leak {leak_st:.1e}/{leak_rm:.1e} "
          f"({time.time() - t0:.1f} s)", flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
print(f"done {len(res)} layers in {time.time() - t_all:.0f} s; worst total cos mean "
      f"{min(v['total_cos_min_mean'][1] for v in res.values()):.5f}, worst token "
      f"{min(v['total_cos_min_mean'][0] for v in res.values()):.5f}")
