#!/usr/bin/env python3
"""Loopback test of the graph-resident expert step (idea: @javierbits).
Captures, per 'layer': x -> pinned host (UVA), WriteValue32(ready=1), a stand-in GPU kernel, WaitValue32(done==1),
WriteValue32(done=0), add = rows.sum(). The C proxy (nconn=0) copies x into rows[0]. Checks correctness on every
replay with fresh inputs and times the per-layer round trip vs. the same graph without the memops."""
import ctypes, sys, time
import torch
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor as uva

NL, H = 93, 3584
LAYERS = list(range(1, 93))
lib = ctypes.CDLL(sys.argv[1] if len(sys.argv) > 1 else "/k3a3/libk3hop.so.new")
cu = ctypes.CDLL("libcuda.so.1")
for f in ("cuStreamWriteValue32_v2", "cuStreamWaitValue32_v2"):
    getattr(cu, f).argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint]
    getattr(cu, f).restype = ctypes.c_int
WAIT_EQ = 1

dev = "cuda"
flags = torch.zeros((NL, 32), dtype=torch.int32, pin_memory=True)
hx = torch.zeros((NL, H), dtype=torch.bfloat16, pin_memory=True)
hids = torch.zeros((NL, 16), dtype=torch.int32, pin_memory=True)
hw = torch.zeros((NL, 16), dtype=torch.float32, pin_memory=True)
rows = torch.zeros((NL, 1, H), dtype=torch.bfloat16, pin_memory=True)
dflags, dx, dids, dw, drows = (uva(t) for t in (flags, hx, hids, hw, rows))
fbase = dflags.data_ptr()

lib.k3proxy_start.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                              ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                              ctypes.c_int, ctypes.c_int]
lib.k3proxy_served.restype = ctypes.c_longlong
lib.k3proxy_work_s.restype = ctypes.c_double
rc = lib.k3proxy_start(None, 0, NL, H, None, flags.data_ptr(), hx.data_ptr(), hids.data_ptr(), hw.data_ptr(),
                       rows.data_ptr(), 2_000_000, 71)
assert rc == 0, rc

x_in = torch.randn((len(LAYERS), H), device=dev, dtype=torch.bfloat16)   # static graph input
ids_in = torch.randint(0, 896, (1, 16), device=dev, dtype=torch.int64)
w_in = torch.rand((1, 16), device=dev, dtype=torch.float32)
out = torch.zeros((len(LAYERS), H), device=dev, dtype=torch.float32)
Wm = torch.randn((H, 256), device=dev, dtype=torch.bfloat16)
junk = torch.zeros((1, 256), device=dev, dtype=torch.bfloat16)


def step(with_proxy):
    s = torch.cuda.current_stream().cuda_stream
    for i, L in enumerate(LAYERS):
        x = x_in[i:i + 1]
        if with_proxy:
            dx[L:L + 1].copy_(x)
            dids[L:L + 1].copy_(ids_in)
            dw[L:L + 1].copy_(w_in)
            assert cu.cuStreamWriteValue32_v2(s, fbase + (L * 32) * 4, 1, 0) == 0
        torch.matmul(x, Wm, out=junk)                          # stand-in for the Station's own experts
        if with_proxy:
            assert cu.cuStreamWaitValue32_v2(s, fbase + (L * 32 + 16) * 4, 1, WAIT_EQ) == 0
            assert cu.cuStreamWriteValue32_v2(s, fbase + (L * 32 + 16) * 4, 0, 0) == 0
            torch.sum(drows[L].float(), dim=0, keepdim=True, out=out[i:i + 1])
        else:
            out[i:i + 1].copy_(x.float())


res = {}
graphs = {}
for name, wp in (("baseline", False), ("proxy", True)):
    st = torch.cuda.Stream()
    with torch.cuda.stream(st):
        step(wp); torch.cuda.synchronize()                      # eager warmup (exercises the proxy for real)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=st):
            step(wp)
    torch.cuda.synchronize()
    graphs[name] = g
    bad = 0
    for r in range(200):                                        # correctness with fresh inputs every replay
        x_in.copy_(torch.randn_like(x_in)); g.replay(); torch.cuda.synchronize()
        bad += 0 if torch.equal(out, x_in.float()) else 1
    print(f"{name}: mismatching replays {bad}/200", flush=True)
for rnd in range(2):
    for name in ("baseline", "proxy"):
        g = graphs[name]
        for _ in range(20): g.replay()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(200): g.replay()
        e1.record(); e1.synchronize()
        res[name] = e0.elapsed_time(e1) / 200 / 1e3
        print(f"round {rnd} {name}: {res[name] * 1e3:.3f} ms per 92-layer step = {res[name] / 92 * 1e6:.2f} us/layer", flush=True)
lib.k3proxy_work_s()
print(f"proxy round trip overhead: {(res['proxy'] - res['baseline']) / 92 * 1e6:.1f} us/layer; "
      f"served {lib.k3proxy_served()}, ")
lib.k3proxy_stop()
