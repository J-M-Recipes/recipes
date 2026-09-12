#!/usr/bin/env python3
"""Step-5 follow-up: why does host-NUMA VMM read at 1.5 GB/s vs 276 GB/s pinned?
Variants on ONE 18 MiB row each, measured with a torch reduction (10 iters):
  A  host-NUMA VMM, access = DEVICE RW                      (what the spike did)
  B  host-NUMA VMM, access = DEVICE RW + HOST_NUMA RW        (EGM docs give both sockets access)
  C  host-NUMA VMM, first-touch from CPU (memset via host ptr after mapping HOST access), then read from GPU
  D  cudaHostAlloc pinned (UVA today)                        (control, expected ~275)
  E  host-NUMA VMM 64 MiB contiguous (larger region — rule out per-2MiB-page TLB effects)
  F  same as A but read twice (second pass: does anything cache?)
"""
import time, json, torch
from cuda.bindings import driver as cu

def chk(res, what):
    err = res[0] if isinstance(res, tuple) else res
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{what}: {cu.cuGetErrorName(err)[1]}")
    return res[1:] if isinstance(res, tuple) and len(res) > 1 else None

chk(cu.cuInit(0), "init"); (dev,) = chk(cu.cuDeviceGet(0), "dev")
torch.cuda.init(); _ = torch.empty(1, device="cuda")
(ctx,) = chk(cu.cuDevicePrimaryCtxRetain(dev), "ctx"); chk(cu.cuCtxSetCurrent(ctx), "cur")
(numa,) = chk(cu.cuDeviceGetAttribute(cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, dev), "numa")
GRAN = 2 << 20

def prop(kind):
    p = cu.CUmemAllocationProp(); p.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    if kind == "host":
        p.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA; p.location.id = int(numa)
    else:
        p.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE; p.location.id = int(dev)
    return p

def acc(kind):
    a = cu.CUmemAccessDesc(); a.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    if kind == "host":
        a.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA; a.location.id = int(numa)
    else:
        a.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE; a.location.id = int(dev)
    return a

class CAI:
    def __init__(self, ptr, nbytes): self.__cuda_array_interface__ = {"shape": (nbytes // 2,), "typestr": "<f2", "data": (int(ptr), False), "version": 3, "strides": None}

def vmm(nbytes, kind, access):
    (va,) = chk(cu.cuMemAddressReserve(nbytes, GRAN, 0, 0), "reserve")
    (h,) = chk(cu.cuMemCreate(nbytes, prop(kind), 0), "create")
    chk(cu.cuMemMap(int(va), nbytes, 0, h, 0), "map")
    descs = [acc(k) for k in access]
    chk(cu.cuMemSetAccess(int(va), nbytes, descs, len(descs)), "access")
    return int(va), h

def bw(x, iters=10):
    torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    a = torch.zeros((), device="cuda"); s.record()
    for _ in range(iters): a += x.float().sum()
    e.record(); torch.cuda.synchronize(); ms = s.elapsed_time(e) / iters
    return round(x.numel() * 2 / (ms / 1000) / 1e9, 1)

R = 18 << 20
res = {}
# A
va, _ = vmm(R, "host", ["dev"]); t = torch.as_tensor(CAI(va, R), device="cuda"); t.fill_(1.0); res["A_host_vmm_devaccess"] = bw(t)
res["F_second_pass"] = bw(t)
# B
try:
    va, _ = vmm(R, "host", ["dev", "host"]); t = torch.as_tensor(CAI(va, R), device="cuda"); t.fill_(1.0); res["B_host_vmm_dev+host_access"] = bw(t)
except RuntimeError as e: res["B_host_vmm_dev+host_access"] = f"ERR {e}"
# C: CPU first-touch via host access then GPU read
try:
    import ctypes
    va, _ = vmm(R, "host", ["dev", "host"])
    ctypes.memset(ctypes.c_void_p(va), 0, R)  # CPU touch through the mapped VA
    t = torch.as_tensor(CAI(va, R), device="cuda"); res["C_cpu_firsttouch_then_gpu"] = bw(t)
except Exception as e: res["C_cpu_firsttouch_then_gpu"] = f"ERR {type(e).__name__}: {e}"
# D control
p = torch.empty(R // 2, dtype=torch.float16, pin_memory=True); p.fill_(1.0)
res["D_pinned_uva_control"] = bw(torch.as_tensor(CAI(p.data_ptr(), R), device="cuda"))
# E larger
va, _ = vmm(64 << 20, "host", ["dev"]); t = torch.as_tensor(CAI(va, 64 << 20), device="cuda"); t.fill_(1.0); res["E_host_vmm_64MiB"] = bw(t)
# G: device-side control on VMM (rules out the VMM wrapper itself)
va, _ = vmm(R, "dev", ["dev"]); t = torch.as_tensor(CAI(va, R), device="cuda"); t.fill_(1.0); res["G_dev_vmm_control"] = bw(t)
# H: cudaMallocManaged-style? torch has none; skip. I: torch pinned but allocated with cudaHostAlloc portable? same as D.
print(json.dumps(res, indent=1))
json.dump(res, open("/w/results/spike-vmm-egm-step5.json", "w"), indent=1)
