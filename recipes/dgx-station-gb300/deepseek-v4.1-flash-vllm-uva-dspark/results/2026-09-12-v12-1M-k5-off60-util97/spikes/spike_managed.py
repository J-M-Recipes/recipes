#!/usr/bin/env python3
"""Managed-memory spike (NEXT-STEPS research round 2).
Does Grace-Blackwell access-counter migration move GPU-hot pages from LPDDR5X to HBM automatically,
with the pointer unchanged? If yes, vLLM's UVA offloader could use cudaMallocManaged instead of pinned
memory and get an adaptive hot/cold split for free.

Variants, 256 MiB each (> B300 L2), bandwidth via torch reduce (float32 accumulate), 1 iter per sample:
  P   pinned cudaHostAlloc (UVA today)                        control ~350 GB/s
  M0  cudaMallocManaged, CPU first-touch, no advice           read 30x, print every pass -> migration curve
  M1  managed + cudaMemAdvise(PreferredLocation=CPU numa0) + AccessedBy(GPU)   (the "stay in Grace but let GPU read" advice)
  M2  managed + SetAccessedBy only
  M3  managed + cudaMemPrefetchAsync to GPU                   (explicit move; ceiling for what migration can reach)
  S   system malloc via numpy (ATS, first-touch on CPU)       read 30x -> does system memory migrate?
Also: after M0's 30 passes, cudaMemRangeGetAttribute(LastPrefetchLocation/PreferredLocation) and a CPU
read timing to infer where the pages ended up.
"""
import json, time, ctypes, numpy as np, torch
from cuda.bindings import runtime as rt

def chk(res, what):
    err = res[0]
    if err != rt.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{what}: {rt.cudaGetErrorName(err)[1]}")
    return res[1:] if len(res) > 1 else None

torch.cuda.init(); _ = torch.empty(1, device="cuda"); dev = 0
N = 256 << 20
class CAI:
    def __init__(self, ptr, nbytes): self.__cuda_array_interface__ = {"shape": (nbytes // 2,), "typestr": "<f2", "data": (int(ptr), False), "version": 3, "strides": None}
def bw1(x):
    torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record(); v = x.sum(dtype=torch.float32); e.record(); torch.cuda.synchronize()
    return round(x.numel() * 2 / (s.elapsed_time(e) / 1000) / 1e9, 1)
def curve(x, n=30):
    return [bw1(x) for _ in range(n)]
res = {}
(numa,) = chk(rt.cudaDeviceGetAttribute(rt.cudaDeviceAttr.cudaDevAttrHostNumaId, dev), "numa")
res["numa_id"] = int(numa)
(cc_major,) = chk(rt.cudaDeviceGetAttribute(rt.cudaDeviceAttr.cudaDevAttrConcurrentManagedAccess, dev), "cma")
(pma,) = chk(rt.cudaDeviceGetAttribute(rt.cudaDeviceAttr.cudaDevAttrPageableMemoryAccess, dev), "pma")
(pmah,) = chk(rt.cudaDeviceGetAttribute(rt.cudaDeviceAttr.cudaDevAttrPageableMemoryAccessUsesHostPageTables, dev), "pmah")
res["attrs"] = {"concurrentManagedAccess": int(cc_major), "pageableMemoryAccess": int(pma), "pageableUsesHostPageTables(ATS)": int(pmah)}
print(res, flush=True)

# P control
p = torch.empty(N // 2, dtype=torch.float16, pin_memory=True); p.fill_(1.0)
tp = torch.as_tensor(CAI(p.data_ptr(), N), device="cuda")
res["P_pinned"] = curve(tp, 5); print("P", res["P_pinned"], flush=True)

def managed(advise):
    (ptr,) = chk(rt.cudaMallocManaged(N, rt.cudaMemAttachGlobal), "mallocManaged")
    ctypes.memset(ctypes.c_void_p(int(ptr)), 0x3c, N)  # CPU first touch (0x3c3c = fp16 1.0-ish)
    loc_cpu = rt.cudaMemLocation(); loc_cpu.type = rt.cudaMemLocationType.cudaMemLocationTypeHostNuma; loc_cpu.id = int(numa)
    loc_gpu = rt.cudaMemLocation(); loc_gpu.type = rt.cudaMemLocationType.cudaMemLocationTypeDevice; loc_gpu.id = dev
    if "pref_cpu" in advise: chk(rt.cudaMemAdvise(ptr, N, rt.cudaMemoryAdvise.cudaMemAdviseSetPreferredLocation, loc_cpu), "advise pref cpu")
    if "accessed_by_gpu" in advise: chk(rt.cudaMemAdvise(ptr, N, rt.cudaMemoryAdvise.cudaMemAdviseSetAccessedBy, loc_gpu), "advise accessedby")
    if "prefetch_gpu" in advise:
        chk(rt.cudaMemPrefetchAsync(ptr, N, loc_gpu, 0, 0), "prefetch"); chk(rt.cudaDeviceSynchronize(), "sync")
    return int(ptr)

for name, adv in (("M0_managed_plain", ()), ("M1_managed_prefCPU_accessedByGPU", ("pref_cpu", "accessed_by_gpu")),
                  ("M2_managed_accessedByGPU", ("accessed_by_gpu",)), ("M3_managed_prefetched_to_GPU", ("prefetch_gpu",))):
    try:
        ptr = managed(adv); t = torch.as_tensor(CAI(ptr, N), device="cuda")
        c = curve(t, 30 if name != "M3_managed_prefetched_to_GPU" else 5)
        res[name] = c; print(name, c, flush=True)
        # residency after the GPU passes: full CPU read bandwidth (HBM-resident pages read slowly over C2C from the CPU / fault back)
        buf = np.frombuffer((ctypes.c_char * N).from_address(ptr), dtype=np.uint8)
        t0 = time.time(); _ = int(buf[::64].sum()); dt = time.time() - t0
        res[name + "_cpu_read_GBps_after"] = round(N / dt / 1e9, 2)
        # and the GPU curve again after the CPU touched it (did pages migrate back?)
        res[name + "_gpu_after_cpu_touch"] = curve(t, 5)
        print(name, "cpu_read_after", res[name + "_cpu_read_GBps_after"], "gpu_after_cpu", res[name + "_gpu_after_cpu_touch"], flush=True)
    except Exception as e:
        res[name] = f"ERR {type(e).__name__}: {e}"; print(name, res[name], flush=True)

# B: 8 GiB managed plain — does it all migrate (HBM has ~5 GiB free next to the REF)? expect partial/thrash
try:
    NB = 8 << 30
    (pb,) = chk(rt.cudaMallocManaged(NB, rt.cudaMemAttachGlobal), "mallocManaged big")
    ctypes.memset(ctypes.c_void_p(int(pb)), 0x3c, NB)
    tb = torch.as_tensor(CAI(int(pb), NB), device="cuda")
    res["B_managed_8GiB_plain"] = curve(tb, 6); print("B", res["B_managed_8GiB_plain"], flush=True)
    (free, total) = chk(rt.cudaMemGetInfo(), "meminfo"); res["hbm_free_GiB_after_B"] = round(free / 2**30, 2)
    chk(rt.cudaFree(pb), "free big")
except Exception as e:
    res["B_managed_8GiB_plain"] = f"ERR {type(e).__name__}: {e}"; print("B", res["B_managed_8GiB_plain"], flush=True)

# S: plain numpy (system-allocated, ATS)
try:
    a = np.full(N // 2, 1.0, dtype=np.float16)
    chk(rt.cudaHostRegister(a.ctypes.data, N, rt.cudaHostRegisterPortable), "hostRegister")
    (dptr,) = chk(rt.cudaHostGetDevicePointer(a.ctypes.data, 0), "getDevPtr")
    ts = torch.as_tensor(CAI(int(dptr), N), device="cuda")
    res["S_system_malloc"] = curve(ts, 30); print("S", res["S_system_malloc"], flush=True)
except Exception as e:
    res["S_system_malloc"] = f"ERR {type(e).__name__}: {e}"; print("S", res["S_system_malloc"], flush=True)

json.dump(res, open("/w/results/spike-managed.json", "w"), indent=1)
print(json.dumps({k: (v if not isinstance(v, list) else (v[:3] + ["..."] + v[-3:] if len(v) > 6 else v)) for k, v in res.items()}, indent=1))
