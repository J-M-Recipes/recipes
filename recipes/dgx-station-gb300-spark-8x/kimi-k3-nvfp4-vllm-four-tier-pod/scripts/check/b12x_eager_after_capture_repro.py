#!/usr/bin/env python3
"""Repro: B12xMoEWrapper(use_cuda_graph=True) — the first eager .run() after CUDA-graph capture, on the same weights as
the LAST graph captured with that wrapper, can return a wrong result. Later eager calls and all graph replays are fine.

Self-contained: random NVFP4 payloads (packed E2M1 bytes, E4M3 block scales converted with convert_sf_to_mma_layout),
activation "silu", no model weights needed.

For each bucket size M (rows; top_k=1, one expert per row) the script:
  1. builds a wrapper with use_cuda_graph=True, max_num_tokens=M; one eager warm-up call on layer 0;
  2. captures one CUDA graph per layer (layers 0..NL-1, distinct weight tensors), in that order;
  3. sets fresh inputs, then calls run() eagerly REPS times on layer NL-1 (the last captured), then on layer 0;
  4. compares every eager call and a graph replay against a REFERENCE wrapper (use_cuda_graph=False, never captured,
     same inputs/weights), and prints output norms.
Observed (bug): on the first trial, every eager call on layer NL-1 returns an output that is far from the reference
(|out| ~ 0) while the graph replay matches it; later trials and layer 0 are fine.

usage: python b12x_eager_after_capture_repro.py [--trials 5] [--reps 8] [--layers 4] [--buckets 2,4,8] [--warm]
       --warm: one extra eager call per layer after capture (workaround) — expect no deviation.
"""
import argparse
import torch
import flashinfer
from flashinfer.fused_moe import B12xMoEWrapper
from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import convert_sf_to_mma_layout

ap = argparse.ArgumentParser()
ap.add_argument("--trials", type=int, default=5)
ap.add_argument("--reps", type=int, default=8)
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--experts", type=int, default=16)
ap.add_argument("--hidden", type=int, default=3584)
ap.add_argument("--inter", type=int, default=3072)
ap.add_argument("--buckets", default="2,4,8")
ap.add_argument("--warm", action="store_true")
a = ap.parse_args()
E, H, I, NL = a.experts, a.hidden, a.inter, a.layers
dev = "cuda"
print(f"flashinfer {flashinfer.__version__} torch {torch.__version__} {torch.cuda.get_device_name()} "
      f"cc {torch.cuda.get_device_capability()} | E={E} H={H} I={I} layers={NL} warm={a.warm}", flush=True)
g = torch.Generator(device=dev).manual_seed(0)


def sf(m, k):
    s = (torch.rand((E, m, k // 16), generator=g, device=dev) * 1.5 + 0.5).to(torch.float8_e4m3fn).view(torch.uint8)
    return convert_sf_to_mma_layout(s.reshape(E * m, k // 16), m=m, k=k, num_groups=E, sf_vec_size=16), s


layers = []
for _ in range(NL):
    w13_sf, k1 = sf(2 * I, H)
    w2_sf, k2 = sf(H, I)
    layers.append(dict(
        w13=torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=dev, generator=g),
        w2=torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=dev, generator=g),
        w13_sf=w13_sf, w2_sf=w2_sf, keep=(k1, k2),
        a1=torch.full((E,), 2e-3, device=dev), a2=torch.full((E,), 2e-3, device=dev)))
one = torch.ones((), dtype=torch.float32, device=dev)


def rel(x, y):
    return float((x.float() - y.float()).norm() / y.float().norm().clamp_min(1e-12))


bad_first = bad_other = n_first = n_other = 0
refw = B12xMoEWrapper(num_experts=E, top_k=1, hidden_size=H, intermediate_size=I, use_cuda_graph=False,
                      num_local_experts=E, output_dtype=torch.bfloat16, activation="silu", quant_mode="nvfp4")
for M in [int(v) for v in a.buckets.split(",")]:
    w = B12xMoEWrapper(num_experts=E, top_k=1, hidden_size=H, intermediate_size=I, use_cuda_graph=True,
                       max_num_tokens=M, num_local_experts=E, output_dtype=torch.bfloat16, activation="silu",
                       quant_mode="nvfp4")
    x = torch.zeros((M, H), dtype=torch.bfloat16, device=dev)
    ids = torch.zeros((M, 1), dtype=torch.int32, device=dev)
    wt = torch.ones((M, 1), dtype=torch.float32, device=dev)
    out = torch.zeros((M, H), dtype=torch.bfloat16, device=dev)

    def body(L):
        p = layers[L]
        out.copy_(w.run(x, p["w13"], p["w13_sf"], p["w2"], p["w2_sf"], ids, wt, w1_alpha=p["a1"], w2_alpha=p["a2"],
                        fc2_input_scale=one, input_global_scale=one))

    x.normal_(generator=g); ids[:, 0] = torch.arange(M, device=dev, dtype=torch.int32) % E
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        body(0); torch.cuda.synchronize()
        graphs = []
        for L in range(NL):
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr, stream=s):
                body(L)
            graphs.append(gr)
    torch.cuda.synchronize()
    fa = getattr(w, "_folded_w1_alpha", None)
    if fa is not None:
        print(f"M={M} after capture: wrapper._folded_w1_alpha (key = last captured layer) = {fa[:4].tolist()}  "
              f"expected w1_alpha*input_global_scale = {(layers[NL - 1]['a1'][:4] * one).tolist()}", flush=True)
    if a.warm:
        for L in range(NL):
            body(L)
        torch.cuda.synchronize()
    for t in range(a.trials):
        x.copy_(torch.randn((M, H), generator=g, device=dev).to(torch.bfloat16))
        ids[:, 0] = torch.randint(0, E, (M,), device=dev, dtype=torch.int32, generator=g)
        wt[:, 0] = torch.rand((M,), device=dev, generator=g)
        torch.cuda.synchronize()
        for L in (NL - 1, 0):
            p = layers[L]
            ref = refw.run(x, p["w13"], p["w13_sf"], p["w2"], p["w2_sf"], ids, wt, w1_alpha=p["a1"],
                           w2_alpha=p["a2"], fc2_input_scale=one, input_global_scale=one).clone()
            torch.cuda.synchronize()
            runs = []
            for r in range(a.reps):
                body(L); torch.cuda.synchronize(); runs.append(out.clone())
            graphs[L].replay(); torch.cuda.synchronize(); gdev = rel(out, ref)
            devs = [rel(rr, ref) for rr in runs]
            wrong = [d > 0.1 for d in devs]                 # kernel noise vs reference is ~1e-2
            if L == NL - 1 and t == 0:
                n_first += 1; bad_first += any(wrong)
            else:
                n_other += 1; bad_other += any(wrong)
            print(f"M={M} trial={t} layer={L}{' (last captured)' if L == NL - 1 else '                '}: |ref| {ref.float().norm():.3e}"
                  f"  eager |out| #0 {runs[0].float().norm():.3e}  eager rel-vs-ref #0 {devs[0]:.2e} max {max(devs):.2e}"
                  f" ({sum(wrong)}/{a.reps} wrong)  graph rel-vs-ref {gdev:.2e}{'   <-- EAGER WRONG' if any(wrong) else ''}",
                  flush=True)
print(f"SUMMARY eager wrong on (first trial, last-captured layer): {bad_first}/{n_first}; "
      f"eager wrong anywhere else: {bad_other}/{n_other}", flush=True)
