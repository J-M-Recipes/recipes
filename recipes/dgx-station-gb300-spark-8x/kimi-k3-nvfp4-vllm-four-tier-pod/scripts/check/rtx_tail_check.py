"""RTX graph-vs-eager tail: reproduce the worker's exact data path and compare distributions by k.
Worker path: ids/w in a cudaHostRegister'd scratch page (dev_tensor views), x in a static device tensor, graph writes y
into a registered reply slot. Compares graph-vs-eager (ge), eager-vs-eager (ee), graph-vs-graph (gg)."""
import ctypes, mmap, os, struct, sys, torch, numpy as np
sys.path.insert(0, "/w/a3")
import worker as W
from flashinfer.fused_moe import B12xMoEWrapper
BANK = os.environ["BANK"]
with open(BANK, "rb") as f:
    W.E = E = struct.unpack("<IIII", f.read(24)[8:24])[2]
NL, H, TOPK = 3, 3584, 16
layers, _, _ = W.load_bank(BANK, NL)
one = torch.ones((), dtype=torch.float32, device="cuda")
buf = mmap.mmap(-1, 4 * 65536)
ptr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
assert int(torch.cuda.cudart().cudaHostRegister(ptr, 4 * 65536, 0)) == 0
ids_np = np.frombuffer((ctypes.c_int32 * 16).from_address(ptr), dtype=np.int32)
w_np = np.frombuffer((ctypes.c_float * 16).from_address(ptr + 64), dtype=np.float32)
reply = W.dev_tensor(ptr + 4096, (1, H), "<i2", torch.bfloat16)
x_static = torch.zeros((1, H), dtype=torch.bfloat16, device="cuda")
rs = np.random.default_rng(int(os.environ.get("SEED", "0")))
out = {}
for k in (1, 2, 4, 8, 16):
    wr = B12xMoEWrapper(num_experts=E, top_k=k, hidden_size=H, intermediate_size=3072, use_cuda_graph=os.environ.get("WRAP_GRAPH", "1") == "1",
                        max_num_tokens=1, num_local_experts=E, output_dtype=torch.bfloat16, activation="situ", quant_mode="nvfp4")
    ids_t = W.dev_tensor(ptr, (1, k), "<i4"); w_t = W.dev_tensor(ptr + 64, (1, k), "<f4")
    def run(L):
        Lr = layers[L]
        return wr.run(x_static, Lr["w13"], Lr["w13_sf"], Lr["w2"], Lr["w2_sf"], ids_t, w_t, w1_alpha=Lr["w1_alpha"],
                      w2_alpha=Lr["w2_alpha"], fc2_input_scale=one, input_global_scale=one)
    ids_np[:k] = np.arange(k); w_np[:k] = 1.0 / k
    if os.environ.get("WRAP_GRAPH", "1") != "1":
        ee = []
        r = lambda u, v: float((u.float() - v.float()).norm() / v.float().norm())
        for t in range(150):
            L = t % NL
            ids_np[:k] = rs.choice(E, k, replace=False); w_np[:k] = rs.random(k).astype(np.float32)
            x_static.normal_(); torch.cuda.synchronize()
            e1 = run(L).clone(); e2 = run(L).clone(); torch.cuda.synchronize()
            ee.append(r(e2, e1))
        q = lambda v: (round(float(np.median(v)), 4), round(float(np.percentile(v, 95)), 4), round(float(max(v)), 4))
        print(f"k={k:2d}  (median, p95, max)  eager(graph=False wrapper)-eager {q(ee)}", flush=True)
        continue
    s = torch.cuda.Stream(); gs = {}
    with torch.cuda.stream(s):
        run(0); torch.cuda.synchronize()
        for L in range(NL):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                reply.copy_(run(L))
            gs[L] = g
    torch.cuda.synchronize()
    ge, ee, gg = [], [], []
    r = lambda u, v: float((u.float() - v.float()).norm() / v.float().norm())
    for t in range(150):
        L = t % NL
        ids_np[:k] = rs.choice(E, k, replace=False); w_np[:k] = rs.random(k).astype(np.float32)
        x_static.normal_(); torch.cuda.synchronize()
        gs[L].replay(); torch.cuda.synchronize(); g1 = reply.clone()
        gs[L].replay(); torch.cuda.synchronize(); g2 = reply.clone()
        e1 = run(L).clone(); e2 = run(L).clone(); torch.cuda.synchronize()
        ge.append(r(g1, e1)); ee.append(r(e2, e1)); gg.append(r(g2, g1))
    q = lambda v: (round(float(np.median(v)), 4), round(float(np.percentile(v, 95)), 4), round(float(max(v)), 4))
    out[k] = {"ge": q(ge), "ee": q(ee), "gg": q(gg)}
    print(f"k={k:2d}  (median, p95, max)  graph-eager {q(ge)}  eager-eager {q(ee)}  graph-graph {q(gg)}", flush=True)
