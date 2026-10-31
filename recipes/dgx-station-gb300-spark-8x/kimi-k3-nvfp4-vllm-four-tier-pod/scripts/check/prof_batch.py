#!/usr/bin/env python3
"""K3-4 prototype: can the worker's B12x NVFP4 MoE kernel serve a whole prefill chunk in one call?

Loads 2 real bank layers, draws realistic per-worker routing for an M-token chunk (each of 16 global picks lands on
this worker with p = 62/896; tokens with no local pick are not sent), and times:
  loop    - today: one M=1 call per token (eager, host-synchronous)
  pairs   - every (token, local expert) pair as its own row, top_k=1, weight folded in; index_add per token
  padded  - one call, M rows, top_k = max local picks; short rows padded with unused experts at weight 0
and checks pairs/padded against loop (rel L2; the kernel's own run-to-run spread is ~1-2% on GB10).
"""
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
rng = np.random.default_rng(7)
_wr = {}


def wrapper(k, m):
    key = (k, m)
    if key not in _wr:
        _wr[key] = B12xMoEWrapper(num_experts=E, top_k=k, hidden_size=H, intermediate_size=I, use_cuda_graph=False,
                                  max_num_tokens=m, num_local_experts=E, output_dtype=torch.bfloat16,
                                  activation="situ", quant_mode="nvfp4")
    return _wr[key]


def call(Lr, x, ids, w):
    return wrapper(ids.shape[1], max(x.shape[0], 1)).run(
        x, Lr["w13"], Lr["w13_sf"], Lr["w2"], Lr["w2_sf"], ids, w, w1_alpha=Lr["w1_alpha"],
        w2_alpha=Lr["w2_alpha"], fc2_input_scale=one, input_global_scale=one)


def draw(M):
    toks = []
    while len(toks) < M:
        k = rng.binomial(16, E / 896)
        if k:
            toks.append((rng.choice(E, k, replace=False), rng.dirichlet(np.ones(16))[:k].astype(np.float32)))
    return toks


def timeit(f, n):
    f(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        y = f()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3, y


def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


print(f"{'M':>4} {'pairs':>5} {'kmax':>4} | {'loop ms':>8} {'pairs ms':>8} {'padded ms':>9} | rel pairs  rel padded", flush=True)
for M in [int(v) for v in os.environ.get("MS", "1,8,32,64,128,256").split(",")]:
    Lr = layers[M % 2]
    toks = draw(M)
    x = (torch.randn((M, H), device="cuda") * 0.1).to(torch.bfloat16)
    ids1 = [torch.tensor(t[0], dtype=torch.int32, device="cuda").view(1, -1) for t in toks]
    w1 = [torch.tensor(t[1], device="cuda").view(1, -1) for t in toks]

    def loop():
        return torch.cat([call(Lr, x[i:i + 1], ids1[i], w1[i]) for i in range(M)])

    # pairs
    tix = np.concatenate([np.full(len(t[0]), i) for i, t in enumerate(toks)])
    pid = torch.tensor(np.concatenate([t[0] for t in toks]), dtype=torch.int32, device="cuda").view(-1, 1)
    pw = torch.tensor(np.concatenate([t[1] for t in toks]), device="cuda").view(-1, 1)
    tix_t = torch.tensor(tix, device="cuda")
    P = len(tix)

    def pairs():
        yp = call(Lr, x.index_select(0, tix_t), pid, pw)
        out = torch.zeros((M, H), dtype=torch.float32, device="cuda")
        out.index_add_(0, tix_t, yp.float())
        return out.to(torch.bfloat16)

    # padded
    kmax = max(len(t[0]) for t in toks)
    pad_ids = np.zeros((M, kmax), np.int32); pad_w = np.zeros((M, kmax), np.float32)
    for i, (e, wv) in enumerate(toks):
        rest = [j for j in range(E) if j not in set(e.tolist())][:kmax - len(e)]
        pad_ids[i] = np.concatenate([e, rest]); pad_w[i, :len(e)] = wv
    pid2 = torch.tensor(pad_ids, device="cuda"); pw2 = torch.tensor(pad_w, device="cuda")

    def padded():
        return call(Lr, x, pid2, pw2)

    n = 5 if M >= 64 else 20
    t_loop, y_loop = timeit(loop, max(2, n // 4))
    t_pairs, y_pairs = timeit(pairs, n)
    t_pad, y_pad = timeit(padded, n)
    print(f"{M:4d} {P:5d} {kmax:4d} | {t_loop:8.2f} {t_pairs:8.2f} {t_pad:9.2f} | {rel(y_pairs, y_loop):.2e}   "
          f"{rel(y_pad, y_loop):.2e}", flush=True)
