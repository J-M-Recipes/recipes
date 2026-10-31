#!/usr/bin/env python3
"""RTX (sm_120) small-batch CUDA graph: kernel noise or a real graph error?

For each trial (bucket b, layer, random routing of b (token, expert) pairs over 8 tokens), run the worker's exact
small-batch body (gather rows -> top_k=1 B12x -> index_add into an f32 accumulator) REPS times eagerly and REPS times
as a captured CUDA graph, and compare:
  noise_e   median rel(eager_i, mean_eager)     run-to-run spread of the eager kernel
  noise_g   median rel(graph_i, mean_eager)     graph runs against the eager centre
  spread_g  median rel(graph_i, mean_graph)     run-to-run spread of the graph
  bias      rel(mean_graph, mean_eager)         systematic offset (noise alone -> ~ spread / sqrt(REPS))
  ref_e/g   rel(mean_eager / mean_graph, fp32)  both against the independent fp32 reference from the checkpoint
Usage (in the worker image, on the RTX): rtx_sb_check.py [REPS=20] [TRIALS=24]
"""
import json, os, sys, time
import numpy as np
import torch
from safetensors import safe_open
sys.path.insert(0, "/w/a3")
import worker as W
from ref_expert import dequantize_nvfp4, situ
from flashinfer.fused_moe import B12xMoEWrapper

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
TRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
E, H, I, NL = 56, 3584, 3072, 4
W.E = E
plan = json.load(open("/w/a3/plan-4tier-v1.json"))["targets"]["rtx0"]["layers"]
layers, _, _ = W.load_bank("/bank/k3bank-rtx0.ldb", NL)
held = {li: sorted(plan[str(li + 1)]) for li in range(NL)}          # bank layer li = model layer li+1
one = torch.ones((), dtype=torch.float32, device="cuda")
BK = tuple(int(v) for v in os.environ.get("BKTS", "2,4,8,16,32").split(","))
M = 8
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

CK = "/ckpt"
idx = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]
_h = {}


def expert_fp32(layer, e):
    base = f"language_model.model.layers.{layer}.block_sparse_moe.experts.{e}."
    Wt = {}
    for p in ("w1", "w2", "w3"):
        k = lambda s: base + p + "." + s
        sh = idx[k("weight")]
        h = _h.get(sh) or _h.setdefault(sh, safe_open(f"{CK}/{sh}", "pt"))
        Wt[p] = dequantize_nvfp4(h.get_tensor(k("weight")).cuda(), h.get_tensor(k("weight_scale")).cuda(),
                                 h.get_tensor(k("weight_scale_2")).cuda().float().reshape(()))
    return Wt


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


rng = np.random.default_rng(20261031)
out = []
print(f"{'b':>2} {'L':>2} | noise_e  noise_g  spread_g  bias     | ref_e    ref_g", flush=True)
for tr in range(TRIALS):
    b = BK[tr % len(BK)]; li = int(rng.integers(0, NL))
    toks = rng.integers(0, M, b); loc = rng.integers(0, E, b); wv = rng.random(b).astype(np.float32)
    xin = (torch.randn((M, H), device="cuda") * float(rng.choice([0.05, 0.2, 1.0]))).to(torch.bfloat16)
    x8.copy_(xin); gat[b].copy_(torch.tensor(toks, device="cuda")); tix[b].copy_(gat[b])
    sid[b][:, 0] = torch.tensor(loc, dtype=torch.int32, device="cuda"); swt[b][:, 0] = torch.tensor(wv, device="cuda")
    torch.cuda.synchronize()
    E_, G_ = [], []
    for r in range(REPS):                                 # interleave so drift hits both equally
        body(b, li); torch.cuda.synchronize(); E_.append(acc[:M].clone())
        graphs[(b, li)].replay(); torch.cuda.synchronize(); G_.append(acc[:M].clone())
    Es, Gs = torch.stack(E_), torch.stack(G_)
    me, mg = Es.mean(0), Gs.mean(0)
    ref = torch.zeros((M, H), dtype=torch.float32, device="cuda")
    for t, l, w in zip(toks, loc, wv):
        Wt = expert_fp32(li + 1, held[li][int(l)])
        xv = xin[int(t)].float().view(1, -1)
        hdn = situ(xv @ Wt["w1"].T, xv @ Wt["w3"].T)
        ref[int(t)] += float(w) * (hdn @ Wt["w2"].T)[0]
    rows = sorted(set(int(t) for t in toks)); me, mg, ref = me[rows], mg[rows], ref[rows]
    Es, Gs = Es[:, rows], Gs[:, rows]
    re_ = [rel(e, ref) for e in Es]; rg_ = [rel(g, ref) for g in Gs]
    base = min(np.median(re_), np.median(rg_))
    bad_e = sum(v > 1.5 * base + 0.05 for v in re_); bad_g = sum(v > 1.5 * base + 0.05 for v in rg_)
    r = {"b": b, "layer": li + 1, "noise_e": float(np.median([rel(e, me) for e in Es])),
         "noise_g": float(np.median([rel(g, me) for g in Gs])), "spread_g": float(np.median([rel(g, mg) for g in Gs])),
         "bias": rel(mg, me), "ref_e": rel(me, ref), "ref_g": rel(mg, ref),
         "max_g_vs_me": float(max(rel(g, me) for g in Gs)), "max_e_vs_me": float(max(rel(e, me) for e in Es)),
         "ref_runs_e": re_, "ref_runs_g": rg_, "bad_e": int(bad_e), "bad_g": int(bad_g)}
    out.append(r)
    print(f"{b:2d} {li + 1:2d} | {r['noise_e']:.2e} {r['noise_g']:.2e} {r['spread_g']:.2e} {r['bias']:.2e} | "
          f"{r['ref_e']:.2e} {r['ref_g']:.2e} | bad runs vs fp32: eager {bad_e}/{REPS} graph {bad_g}/{REPS} "
          f"(worst eager {max(re_):.2f}, graph {max(rg_):.2f})", flush=True)
json.dump(out, open(os.environ.get("OUT", "/w/a3/rtx_sb_check.json"), "w"), indent=1)
print(f"TOTAL bad runs vs fp32: eager {sum(o['bad_e'] for o in out)}/{REPS * len(out)}, "
      f"graph {sum(o['bad_g'] for o in out)}/{REPS * len(out)}", flush=True)
k = lambda key: float(np.median([o[key] for o in out]))
print(f"MEDIAN noise_e {k('noise_e'):.2e} noise_g {k('noise_g'):.2e} spread_g {k('spread_g'):.2e} bias {k('bias'):.2e} "
      f"ref_e {k('ref_e'):.2e} ref_g {k('ref_g'):.2e}", flush=True)
