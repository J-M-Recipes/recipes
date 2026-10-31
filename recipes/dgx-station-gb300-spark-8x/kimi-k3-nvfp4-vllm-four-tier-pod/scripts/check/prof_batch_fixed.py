#!/usr/bin/env python3
"""K3-4: one B12x wrapper sized for the largest chunk (CAP pairs, top_k=1); run it with varying pair counts.
Does it recompile per count, and what does each count cost?"""
import os, sys, time
import numpy as np
import torch
sys.path.insert(0, "/w/a3")
import worker as W
from flashinfer.fused_moe import B12xMoEWrapper
E, H, I = 62, 3584, 3072
W.E = E
layers, _, _ = W.load_bank(os.environ.get("BANK", "/bank/k3bank-spark0.ldb"), 2)
one = torch.ones((), dtype=torch.float32, device="cuda")
CAP = int(os.environ.get("CAP", "4096"))
t0 = time.perf_counter()
wr = B12xMoEWrapper(num_experts=E, top_k=1, hidden_size=H, intermediate_size=I, use_cuda_graph=False,
                    max_num_tokens=CAP, num_local_experts=E, output_dtype=torch.bfloat16, activation="situ",
                    quant_mode="nvfp4")
print(f"wrapper CAP={CAP} built in {time.perf_counter() - t0:.2f} s", flush=True)
rng = np.random.default_rng(3)
for P in [1, 2, 7, 50, 211, 333, 421, 600, 1100, 1999, 2600]:
    if P > CAP:
        break
    Lr = layers[P % 2]
    x = (torch.randn((P, H), device="cuda") * 0.1).to(torch.bfloat16)
    ids = torch.tensor(rng.integers(0, E, P), dtype=torch.int32, device="cuda").view(-1, 1)
    w = torch.rand((P, 1), device="cuda")
    def run():
        return wr.run(x, Lr["w13"], Lr["w13_sf"], Lr["w2"], Lr["w2_sf"], ids, w, w1_alpha=Lr["w1_alpha"],
                      w2_alpha=Lr["w2_alpha"], fc2_input_scale=one, input_global_scale=one)
    t1 = time.perf_counter(); y = run(); torch.cuda.synchronize(); first = time.perf_counter() - t1
    t1 = time.perf_counter()
    for _ in range(10):
        y = run()
    torch.cuda.synchronize()
    print(f"P={P:5d}: first call {first * 1e3:8.1f} ms, steady {(time.perf_counter() - t1) / 10 * 1e3:7.2f} ms, "
          f"finite {bool(torch.isfinite(y).all())}", flush=True)
