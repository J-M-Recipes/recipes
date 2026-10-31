#!/usr/bin/env python3
"""Kernel-faithful check of selected layers: compare each tier's recorded share with an emulation of how that tier
quantizes activations, built from the checkpoint bytes.

  Station (stock vLLM TRT-LLM NVFP4 path): x quantized to NVFP4 with global scale = max over the layer's experts
    of w1.input_scale; SiTU output quantized with that expert's w2.input_scale.
  Workers (B12x, input_global_scale = fc2_input_scale = 1): dynamic per-16 block scales, global scale 1.
Block scales round to e4m3 (with subnormals), values to e2m1. Weights dequantized exactly. fp32 math otherwise.
If a tier matches its emulation far better than the fp32 reference, its deviation from fp32 is the NVFP4 recipe.

Usage: pod_emu_check.py DUMP_DIR PLAN OUT.json L1,L2,...
"""
import glob, json, os, sys
import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, "/situ")
from ref_expert import dequantize_nvfp4, situ  # noqa: E402

DUMP, PLANP, OUT, LAYS = sys.argv[1], sys.argv[2], sys.argv[3], [int(v) for v in sys.argv[4].split(",")]
CK, dev = "/ckpt", "cuda"
idx = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]
plan = json.load(open(PLANP))
remote = {}
for t, v in plan["targets"].items():
    for L, ids in v["layers"].items():
        remote.setdefault(int(L), set()).update(ids)
_h = {}


def get(k):
    s = idx[k]
    if s not in _h:
        _h[s] = safe_open(f"{CK}/{s}", "pt")
    return _h[s].get_tensor(k)


MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=dev)
MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=dev)


def q_nvfp4(x, gs):
    """fake-quantize rows of x (fp32 [m, K]) to NVFP4 with global scale gs (real = e2m1 * e4m3 * gs)."""
    m, K = x.shape
    xb = x.reshape(m, K // 16, 16)
    amax = xb.abs().amax(-1, keepdim=True)
    s = (amax / 6.0 / gs).clamp(max=448.0).to(torch.float8_e4m3fn).float()
    den = s * gs
    v = torch.where(den > 0, xb / den.clamp_min(1e-30), torch.zeros_like(xb))
    mag = MAG[torch.bucketize(v.abs(), MID)]
    return (torch.sign(v) * mag * den).reshape(m, K)


def cosm(a, b):
    c = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    return round(float(c.min()), 5), round(float(c.mean()), 5)


rows = {}
for f in sorted(glob.glob(os.path.join(DUMP, "chunk*.npz"))):
    d = np.load(f)
    lay, x, ids, w, st, rm = (d[k] for k in ("layer", "x", "ids", "w", "station", "remote"))
    for i in range(len(lay)):
        if int(lay[i]) in LAYS:
            rows.setdefault(int(lay[i]), []).append((x[i], ids[i], w[i], st[i], rm[i]))

res = {}
for L in LAYS:
    R = rows[L]
    X = torch.tensor(np.stack([r[0] for r in R]), device=dev)
    IDS = np.stack([r[1] for r in R]); Wt = torch.tensor(np.stack([r[2] for r in R]), device=dev)
    ST = torch.tensor(np.stack([r[3] for r in R]), device=dev); RM = torch.tensor(np.stack([r[4] for r in R]), device=dev)
    base = f"language_model.model.layers.{L}.block_sparse_moe.experts."
    gs1 = max(float(get(f"{base}{e}.w1.input_scale")) for e in range(896))      # layer-wide a1 global scale
    need = sorted({int(e) for e in IDS.ravel() if 0 <= e < 896})
    rem = remote.get(L, set())
    acc = {k: torch.zeros_like(ST) for k in ("f_st", "f_rm", "e_st", "e_rm", "w_st")}
    Xq_st = q_nvfp4(X, gs1)
    Xq_rm = q_nvfp4(X, 1.0)
    for e in need:
        Wd = {p: dequantize_nvfp4(get(f"{base}{e}.{p}.weight").to(dev), get(f"{base}{e}.{p}.weight_scale").to(dev),
                                  get(f"{base}{e}.{p}.weight_scale_2").to(dev).float().reshape(())) for p in ("w1", "w2", "w3")}
        hit = torch.tensor(IDS == e, device=dev); rh = hit.any(1)
        wsel = (Wt[rh] * hit[rh].float()).sum(1, keepdim=True)

        def fwd(xin, gs2):
            h = situ(xin @ Wd["w1"].T, xin @ Wd["w3"].T)
            if gs2 is not None:
                h = q_nvfp4(h, gs2)
            return h @ Wd["w2"].T
        side = "rm" if e in rem else "st"
        acc["f_" + side][rh] += wsel * fwd(X[rh], None)
        if side == "st":
            acc["e_st"][rh] += wsel * fwd(Xq_st[rh], float(get(f"{base}{e}.w2.input_scale")))
            acc["w_st"][rh] += wsel * fwd(Xq_rm[rh], 1.0)        # same experts, worker-style quantization
        else:
            acc["e_rm"][rh] += wsel * fwd(Xq_rm[rh], 1.0)
    hs = acc["f_st"].norm(dim=-1) > 0; hr = acc["f_rm"].norm(dim=-1) > 0
    r = {"x_norm_mean": float(X.norm(dim=-1).mean()), "a1_global_scale": gs1,
         "station_vs_fp32": cosm(ST[hs], acc["f_st"][hs]), "station_vs_emu_station": cosm(ST[hs], acc["e_st"][hs]),
         "station_vs_emu_worker_style": cosm(ST[hs], acc["w_st"][hs]),
         "remote_vs_fp32": cosm(RM[hr], acc["f_rm"][hr]), "remote_vs_emu_worker": cosm(RM[hr], acc["e_rm"][hr])}
    res[L] = r
    print(f"L{L}: " + json.dumps(r), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
