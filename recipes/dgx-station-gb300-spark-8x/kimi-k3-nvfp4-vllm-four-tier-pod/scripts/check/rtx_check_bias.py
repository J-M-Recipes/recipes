#!/usr/bin/env python3
"""Is worker.py's small-batch boot check biased? It compares rel(graph, eager1) with rel(eager2, eager1) where eager2
runs right after eager1. Measure, per sample: g = graph, e1, e2 back to back (worker order), then a graph replay, then
e3 (not back to back). Report median rel(g,e1), rel(e2,e1) [consecutive], rel(e3,e1) [separated], rel(g2,g) [graph-graph].
Also warms each wrapper with one eager call after capture (the first eager call after capture on the last-captured
layer was seen to return garbage once). Usage: rtx_check_bias.py [N=60]"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, "/w/a3")
import worker as W
from flashinfer.fused_moe import B12xMoEWrapper

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
E = int(os.environ.get("E", "56")); H, I, NL = 3584, 3072, 4
W.E = E
layers, _, _ = W.load_bank(os.environ.get("BANK", "/bank/k3bank-rtx0.ldb"), NL)
one = torch.ones((), dtype=torch.float32, device="cuda")
BK, M = (2, 4, 8), 8
wr = {b: B12xMoEWrapper(num_experts=E, top_k=1, hidden_size=H, intermediate_size=I, use_cuda_graph=True,
                        max_num_tokens=b, num_local_experts=E, output_dtype=torch.bfloat16, activation="situ",
                        quant_mode="nvfp4") for b in BK}
x8 = torch.zeros((M, H), dtype=torch.bfloat16, device="cuda")
gat = {b: torch.zeros((b,), dtype=torch.int64, device="cuda") for b in BK}
sid = {b: torch.zeros((b, 1), dtype=torch.int32, device="cuda") for b in BK}
swt = {b: torch.zeros((b, 1), dtype=torch.float32, device="cuda") for b in BK}
tix = {b: torch.zeros((b,), dtype=torch.int64, device="cuda") for b in BK}
acc = torch.zeros((M + 1, H), dtype=torch.float32, device="cuda")


def body(b, li):
    Lr = layers[li]
    y = wr[b].run(x8.index_select(0, gat[b]), Lr["w13"], Lr["w13_sf"], Lr["w2"], Lr["w2_sf"], sid[b], swt[b],
                  w1_alpha=Lr["w1_alpha"], w2_alpha=Lr["w2_alpha"], fc2_input_scale=one, input_global_scale=one)
    acc.zero_()
    acc.index_add_(0, tix[b], y.float())


gs = torch.cuda.Stream(); pool = torch.cuda.graph_pool_handle()
with torch.cuda.stream(gs):
    for b in BK:
        x8.normal_(); sid[b][:, 0] = torch.arange(b, device="cuda", dtype=torch.int32) % E
        swt[b].fill_(0.5); gat[b].copy_(torch.arange(b, device="cuda") % M); tix[b].copy_(gat[b]); body(b, 0)
    torch.cuda.synchronize()
    graphs = {}
    for b in BK:
        for li in range(NL):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool, stream=gs):
                body(b, li)
            graphs[(b, li)] = g
torch.cuda.synchronize()
if os.environ.get("WARM", "1") == "1":
    for b in BK:
        for li in range(NL):
            body(b, li)
    torch.cuda.synchronize()


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


rng = np.random.default_rng(7)
R = {"g_e1": [], "e2_e1": [], "e3_e1": [], "g2_g": []}
for i in range(N):
    b = BK[i % 3]; li = int(rng.integers(0, NL))
    x8.normal_(); sid[b][:, 0] = torch.tensor(rng.integers(0, E, b), dtype=torch.int32, device="cuda")
    swt[b][:, 0] = torch.tensor(rng.random(b), dtype=torch.float32, device="cuda")
    gat[b].copy_(torch.tensor(rng.integers(0, M, b), device="cuda")); tix[b].copy_(gat[b]); torch.cuda.synchronize()
    graphs[(b, li)].replay(); torch.cuda.synchronize(); g = acc[:M].clone()
    body(b, li); torch.cuda.synchronize(); e1 = acc[:M].clone()
    body(b, li); torch.cuda.synchronize(); e2 = acc[:M].clone()
    graphs[(b, li)].replay(); torch.cuda.synchronize(); g2 = acc[:M].clone()
    body(b, li); torch.cuda.synchronize(); e3 = acc[:M].clone()
    R["g_e1"].append(rel(g, e1)); R["e2_e1"].append(rel(e2, e1)); R["e3_e1"].append(rel(e3, e1)); R["g2_g"].append(rel(g2, g))
for k, v in R.items():
    v = np.array(v)
    print(f"{k:6s} median {np.median(v):.2e}  p90 {np.quantile(v, 0.9):.2e}  max {v.max():.2e}  exact-zero {int((v == 0).sum())}/{len(v)}", flush=True)
print(f"worker-check ratio median(g_e1)/median(e2_e1) = {np.median(R['g_e1']) / max(np.median(R['e2_e1']), 1e-12):.2f} "
      f"(limit ~1.5); fair ratio median(g_e1)/median(e3_e1) = {np.median(R['g_e1']) / max(np.median(R['e3_e1']), 1e-12):.2f}", flush=True)
