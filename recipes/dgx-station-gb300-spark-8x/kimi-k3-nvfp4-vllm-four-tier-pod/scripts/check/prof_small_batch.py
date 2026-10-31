#!/usr/bin/env python3
"""K3-5 prototype: small-batch decode on a Spark worker. For a decode step with C concurrent tokens, a Spark serves
P (token, local expert) pairs. Pad P up to a bucket (2/4/8/16/32) with weight-0 copies of a real pair (same expert ->
no extra weight bytes), run one top_k=1 B12x call, captured as a CUDA graph per (bucket, layer). Times graph replay
vs eager vs the M=1 per-token graphs replayed one after another (today's option)."""
import os, sys, time
import numpy as np
import torch
sys.path.insert(0, "/w/a3")
import worker as W
from flashinfer.fused_moe import B12xMoEWrapper
E, H, I = 62, 3584, 3072
W.E = E
NL = 2
layers, _, _ = W.load_bank(os.environ.get("BANK", "/bank/k3bank-spark0.ldb"), NL)
one = torch.ones((), dtype=torch.float32, device="cuda")
rng = np.random.default_rng(11)
BUCKETS = (2, 4, 8, 16, 32)
wr = {b: B12xMoEWrapper(num_experts=E, top_k=1, hidden_size=H, intermediate_size=I, use_cuda_graph=True,
                        max_num_tokens=b, num_local_experts=E, output_dtype=torch.bfloat16, activation="situ",
                        quant_mode="nvfp4") for b in BUCKETS}
xs = {b: torch.zeros((b, H), dtype=torch.bfloat16, device="cuda") for b in BUCKETS}
ids = {b: torch.zeros((b, 1), dtype=torch.int32, device="cuda") for b in BUCKETS}
ws = {b: torch.zeros((b, 1), dtype=torch.float32, device="cuda") for b in BUCKETS}
ys = {b: torch.zeros((b, H), dtype=torch.bfloat16, device="cuda") for b in BUCKETS}


def body(b, L):
    Lr = layers[L]
    y = wr[b].run(xs[b], Lr["w13"], Lr["w13_sf"], Lr["w2"], Lr["w2_sf"], ids[b], ws[b], w1_alpha=Lr["w1_alpha"],
                  w2_alpha=Lr["w2_alpha"], fc2_input_scale=one, input_global_scale=one)
    ys[b].copy_(y)


gs = torch.cuda.Stream()
pool = torch.cuda.graph_pool_handle()
t0 = time.perf_counter()
with torch.cuda.stream(gs):
    for b in BUCKETS:
        xs[b].normal_(); ids[b][:, 0] = torch.arange(b, device="cuda") % E; ws[b].fill_(0.5)
        body(b, 0)
    torch.cuda.synchronize()
    graphs = {}
    for b in BUCKETS:
        for L in range(NL):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool, stream=gs):
                body(b, L)
            graphs[(b, L)] = g
torch.cuda.synchronize()
print(f"built {len(graphs)} graphs in {time.perf_counter() - t0:.1f} s", flush=True)


def timed(f, n=200):
    f(); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        f(); torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e6


print(f"{'C':>2} {'P':>3} {'bkt':>3} | graph us  eager us | rel graph-vs-eager", flush=True)
for C in (2, 4, 8):
    for trial in range(3):
        # this Spark's share of C tokens: each token's 16 picks land here with p = 62/896
        pairs = []
        for t in range(C):
            k = rng.binomial(16, E / 896)
            for e in rng.choice(E, k, replace=False):
                pairs.append((t, int(e), float(rng.random())))
        P = len(pairs)
        if P == 0:
            print(f"{C:2d} {P:3d}   - | (no local pairs)", flush=True); continue
        b = next(x for x in BUCKETS if x >= P)
        L = trial % NL
        xin = (torch.randn((C, H), device="cuda") * 0.1).to(torch.bfloat16)
        rows = [p[0] for p in pairs] + [pairs[0][0]] * (b - P)
        eid = [p[1] for p in pairs] + [pairs[0][1]] * (b - P)
        wv = [p[2] for p in pairs] + [0.0] * (b - P)
        xs[b].copy_(xin[torch.tensor(rows, device="cuda")])
        ids[b][:, 0] = torch.tensor(eid, dtype=torch.int32, device="cuda")
        ws[b][:, 0] = torch.tensor(wv, device="cuda")
        tg = timed(lambda: graphs[(b, L)].replay())
        yg = ys[b][:P].float().clone()
        te = timed(lambda: body(b, L), 50)
        ye = ys[b][:P].float().clone()
        rel = float((yg - ye).norm() / ye.norm().clamp_min(1e-12))
        print(f"{C:2d} {P:3d} {b:3d} | {tg:8.1f} {te:8.1f} | {rel:.2e}", flush=True)
