#!/usr/bin/env python3
"""Step-5b: bandwidth with a working set far larger than L2 (1 GiB each), single pass and repeated.
   host-NUMA VMM  vs  cudaHostAlloc pinned (UVA today)  vs  device VMM  vs  cudaMalloc."""
import json, torch
from cuda.bindings import driver as cu
def chk(res, what):
    err = res[0] if isinstance(res, tuple) else res
    if err != cu.CUresult.CUDA_SUCCESS: raise RuntimeError(f"{what}: {cu.cuGetErrorName(err)[1]}")
    return res[1:] if isinstance(res, tuple) and len(res) > 1 else None
chk(cu.cuInit(0), "init"); (dev,) = chk(cu.cuDeviceGet(0), "dev")
torch.cuda.init(); _ = torch.empty(1, device="cuda")
(ctx,) = chk(cu.cuDevicePrimaryCtxRetain(dev), "ctx"); chk(cu.cuCtxSetCurrent(ctx), "cur")
(numa,) = chk(cu.cuDeviceGetAttribute(cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, dev), "numa")
GRAN = 2 << 20
def prop(kind):
    p = cu.CUmemAllocationProp(); p.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    p.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA if kind == "host" else cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    p.location.id = int(numa) if kind == "host" else int(dev); return p
def vmm(nbytes, kind):
    (va,) = chk(cu.cuMemAddressReserve(nbytes, GRAN, 0, 0), "reserve"); (h,) = chk(cu.cuMemCreate(nbytes, prop(kind), 0), "create")
    chk(cu.cuMemMap(int(va), nbytes, 0, h, 0), "map")
    a = cu.CUmemAccessDesc(); a.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE; a.location.id = int(dev); a.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    chk(cu.cuMemSetAccess(int(va), nbytes, [a], 1), "access"); return int(va)
class CAI:
    def __init__(self, ptr, nbytes): self.__cuda_array_interface__ = {"shape": (nbytes // 2,), "typestr": "<f2", "data": (int(ptr), False), "version": 3, "strides": None}
def bw(x, iters):
    torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    a = torch.zeros((), device="cuda"); s.record()
    for _ in range(iters): a += x.sum(dtype=torch.float32)
    e.record(); torch.cuda.synchronize(); return round(x.numel() * 2 / (s.elapsed_time(e) / iters / 1000) / 1e9, 1)
def copy_bw(x, iters):
    dst = torch.empty_like(x, device="cuda") if x.device.type == "cuda" else None
    torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True); s.record()
    for _ in range(iters): dst.copy_(x)
    e.record(); torch.cuda.synchronize(); return round(x.numel() * 2 / (s.elapsed_time(e) / iters / 1000) / 1e9, 1)
N = 256 << 20  # 256 MiB: > B300 L2 (126 MiB), fits beside the live REF
res = {}
def dump(): json.dump(res, open("/w/results/spike-vmm-egm-step5b.json", "w"), indent=1)
va = vmm(N, "host"); th = torch.as_tensor(CAI(va, N), device="cuda"); th.fill_(1.0); torch.cuda.synchronize()
res["host_vmm_256MiB_reduce_first"] = bw(th, 1); res["host_vmm_256MiB_reduce_x5"] = bw(th, 5); res["host_vmm_256MiB_copy_x5"] = copy_bw(th, 5); dump()
p = torch.empty(N // 2, dtype=torch.float16, pin_memory=True); p.fill_(1.0)
tp = torch.as_tensor(CAI(p.data_ptr(), N), device="cuda")
res["pinned_uva_256MiB_reduce_first"] = bw(tp, 1); res["pinned_uva_256MiB_reduce_x5"] = bw(tp, 5); res["pinned_uva_256MiB_copy_x5"] = copy_bw(tp, 5); dump()
vd = vmm(N, "dev"); td = torch.as_tensor(CAI(vd, N), device="cuda"); td.fill_(1.0)
res["dev_vmm_256MiB_reduce_x5"] = bw(td, 5)
tm = torch.ones(N // 2, dtype=torch.float16, device="cuda"); res["cudaMalloc_256MiB_reduce_x5"] = bw(tm, 5)
print(json.dumps(res, indent=1)); json.dump(res, open("/w/results/spike-vmm-egm-step5b.json", "w"), indent=1)
