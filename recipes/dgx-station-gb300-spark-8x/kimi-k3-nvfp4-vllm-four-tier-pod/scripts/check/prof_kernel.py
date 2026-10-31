#!/usr/bin/env python3
"""Where is the GB10 B12x MoE kernel's ~115 us fixed cost? Times k=1/2/4 at M=1 on 2 real bank layers;
run under nsys to get the per-kernel breakdown."""
import os, sys, time, torch
sys.path.insert(0, "/w/a3")
import worker as W
from flashinfer.fused_moe import B12xMoEWrapper
W.E = 62
layers, _, _ = W.load_bank(os.environ.get("BANK", "/bank/k3bank-spark0.ldb"), 2)
one = torch.ones((), dtype=torch.float32, device="cuda")
x = torch.randn((1, 3584), dtype=torch.bfloat16, device="cuda") * 0.1
for k in (1, 2, 4):
    wr = B12xMoEWrapper(num_experts=62, top_k=k, hidden_size=3584, intermediate_size=3072, use_cuda_graph=False,
                        num_local_experts=62, output_dtype=torch.bfloat16, activation="situ", quant_mode="nvfp4")
    ids = [torch.randperm(62, device="cuda")[:k].to(torch.int32).view(1, k) for _ in range(32)]
    w = torch.full((1, k), 1.0 / k, dtype=torch.float32, device="cuda")
    def run(i):
        Lr = layers[i % 2]
        return wr.run(x, Lr["w13"], Lr["w13_sf"], Lr["w2"], Lr["w2_sf"], ids[i % 32], w, w1_alpha=Lr["w1_alpha"],
                      w2_alpha=Lr["w2_alpha"], fc2_input_scale=one, input_global_scale=one)
    for i in range(50): run(i)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(f"k{k}")
    t0 = time.perf_counter()
    for i in range(200):
        run(i); torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 200
    torch.cuda.nvtx.range_pop()
    print(f"k={k}: {dt * 1e6:.1f} us per call (host-synchronous)", flush=True)
