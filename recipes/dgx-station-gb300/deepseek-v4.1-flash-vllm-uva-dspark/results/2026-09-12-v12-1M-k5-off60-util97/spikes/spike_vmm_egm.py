#!/usr/bin/env python3
"""VMM/EGM spike for the dsv41 static cold-set design (NEXT-STEPS step 3).

Question: can ONE virtual range hold expert rows whose physical backing is a mix of HBM (device) and
Grace memory (host-NUMA via cuMemCreate/EGM), be wrapped as a torch tensor, and be read by a kernel at
UVA-like bandwidth? If yes, the MoE kernel can keep full [num_experts, ...] tensors and one pointer per
layer while residency is decided row by row — no shape change, no autotune, no slot indices.

Steps (each prints PASS/FAIL):
  1. numaId from CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID; granularity for DEVICE and HOST_NUMA props
  2. reserve VA for N rows; cuMemCreate per row alternating DEVICE / HOST_NUMA; cuMemMap; cuMemSetAccess
  3. wrap as torch tensor (from_blob via ctypes-free path: torch.cuda caching allocator bypass using
     torch.frombuffer is host-only, so we use torch.empty(0).set_ via UntypedStorage from data_ptr) — we
     use the documented cudaHostRegister-free route: torch.cuda.memory.caching_allocator? no —
     simplest robust route: torch.as_tensor via __cuda_array_interface__ on a tiny shim object.
  4. write known pattern from GPU, read back per row; verify device rows and host rows both round-trip
  5. bandwidth: read every row with a reduction kernel (torch.sum over rows) ×10; compare host-NUMA-mapped
     rows vs cudaHostAlloc'd (pinned, what UVA offload uses today) rows of the same size
  6. remap: unmap one host row, map a fresh device allocation at the same VA, verify contents can be
     rewritten and the tensor pointer did not change

Sizes: ROW_BYTES = 17.93 MiB rounded up to granularity (expert row w13+scales), N_ROWS = 32 → ~576 MiB.
No serving lane is touched. Run inside vllm/vllm-openai:deepseekv41-flash-0909 with --gpus all.
"""
import ctypes, sys, time, json, os
import torch
from cuda.bindings import driver as cu

def chk(res, what):
    err = res[0] if isinstance(res, tuple) else res
    if err != cu.CUresult.CUDA_SUCCESS:
        _, name = cu.cuGetErrorName(err)
        raise RuntimeError(f"{what}: {name}")
    return res[1:] if isinstance(res, tuple) and len(res) > 1 else None

out = {"steps": {}}
def step(name, ok, **kw):
    out["steps"][name] = {"pass": bool(ok), **kw}
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {kw}", flush=True)

# ---- init ----
chk(cu.cuInit(0), "cuInit")
(dev,) = chk(cu.cuDeviceGet(0), "cuDeviceGet")
torch.cuda.init(); _ = torch.empty(1, device="cuda")  # primary ctx via torch
(ctx,) = chk(cu.cuDevicePrimaryCtxRetain(dev), "primaryCtxRetain")
chk(cu.cuCtxSetCurrent(ctx), "ctxSetCurrent")

# ---- step 1: numa id + granularity ----
(numa_id,) = chk(cu.cuDeviceGetAttribute(cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, dev), "attr HOST_NUMA_ID")
prop_dev = cu.CUmemAllocationProp()
prop_dev.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
prop_dev.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
prop_dev.location.id = int(dev)
prop_host = cu.CUmemAllocationProp()
prop_host.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
prop_host.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
prop_host.location.id = int(numa_id)
(gran_dev,) = chk(cu.cuMemGetAllocationGranularity(prop_dev, cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM), "gran dev")
try:
    (gran_host,) = chk(cu.cuMemGetAllocationGranularity(prop_host, cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM), "gran host")
    host_ok = True
except RuntimeError as e:
    gran_host = None; host_ok = False; print("host granularity error:", e)
step("1_numa_and_granularity", host_ok and numa_id >= 0, numa_id=int(numa_id), gran_dev=int(gran_dev), gran_host=(int(gran_host) if gran_host else None))
if not host_ok:
    json.dump(out, open("/w/results/spike-vmm-egm.json", "w"), indent=1); sys.exit(1)

GRAN = max(int(gran_dev), int(gran_host))
ROW_RAW = int(17.93 * 1024 * 1024)
ROW = (ROW_RAW + GRAN - 1) // GRAN * GRAN
N = 32
TOTAL = ROW * N
print(f"row={ROW/2**20:.2f} MiB (raw 17.93, pad {100*(ROW-ROW_RAW)/ROW_RAW:.2f}%) rows={N} total={TOTAL/2**20:.0f} MiB", flush=True)

# ---- step 2: reserve + per-row create/map with mixed backing ----
(va,) = chk(cu.cuMemAddressReserve(TOTAL, GRAN, 0, 0), "addressReserve")
handles = []
kinds = []
t0 = time.time()
for i in range(N):
    prop = prop_host if (i % 2 == 1) else prop_dev
    (h,) = chk(cu.cuMemCreate(ROW, prop, 0), f"memCreate row{i} {'host' if i%2 else 'dev'}")
    chk(cu.cuMemMap(int(va) + i * ROW, ROW, 0, h, 0), f"memMap row{i}")
    handles.append(h); kinds.append("host" if i % 2 else "dev")
acc = cu.CUmemAccessDesc()
acc.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
acc.location.id = int(dev)
acc.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
chk(cu.cuMemSetAccess(int(va), TOTAL, [acc], 1), "setAccess")
step("2_mixed_backing_one_va", True, map_ms=round((time.time() - t0) * 1000, 1), va=hex(int(va)))

# ---- step 3: wrap as torch tensor via __cuda_array_interface__ ----
class _CAI:
    def __init__(self, ptr, nbytes):
        self.__cuda_array_interface__ = {"shape": (nbytes // 2,), "typestr": "<f2", "data": (int(ptr), False), "version": 3, "strides": None}
t = torch.as_tensor(_CAI(int(va), TOTAL), device="cuda")
rows = t.view(N, ROW // 2)
step("3_torch_wrap", rows.data_ptr() == int(va) and rows.shape == (N, ROW // 2), dtype=str(rows.dtype), shape=list(rows.shape))

# ---- step 4: pattern round-trip on both kinds ----
for i in range(N):
    rows[i].fill_(float(i % 7 + 1))
torch.cuda.synchronize()
ok4 = True; bad = []
for i in range(N):
    v = rows[i][::4096].float()
    if not torch.all(v == float(i % 7 + 1)):
        ok4 = False; bad.append(i)
step("4_roundtrip_dev_and_host_rows", ok4, bad_rows=bad)

# ---- step 5: bandwidth vs cudaHostAlloc pinned (today's UVA path) ----
def bw(x, iters=10):
    torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    acc_ = torch.zeros((), device="cuda", dtype=torch.float32)
    s.record()
    for _ in range(iters):
        acc_ += x.float().sum()
    e.record(); torch.cuda.synchronize()
    ms = s.elapsed_time(e) / iters
    return x.numel() * 2 / (ms / 1000) / 1e9, ms
host_rows = rows[1::2]; dev_rows = rows[0::2]
gb_host, ms_host = bw(host_rows); gb_dev, ms_dev = bw(dev_rows)
pinned = torch.empty((N // 2, ROW // 2), dtype=torch.float16, pin_memory=True)
pinned.fill_(3.0)
pinned_dev_view = torch.as_tensor(_CAI(pinned.data_ptr(), pinned.numel() * 2), device="cuda").view(N // 2, ROW // 2)  # UVA read of pinned host memory
gb_pin, ms_pin = bw(pinned_dev_view)
rel = gb_host / gb_pin if gb_pin else 0
step("5_bandwidth", rel > 0.9, host_numa_vmm_GBps=round(gb_host, 1), pinned_uva_GBps=round(gb_pin, 1), hbm_rows_GBps=round(gb_dev, 1), ratio_vmm_over_pinned=round(rel, 3))

# ---- step 6: remap one host row to device at the same VA; pointer unchanged ----
i = 1
chk(cu.cuMemUnmap(int(va) + i * ROW, ROW), "unmap row1")
(h_new,) = chk(cu.cuMemCreate(ROW, prop_dev, 0), "memCreate replacement dev")
chk(cu.cuMemMap(int(va) + i * ROW, ROW, 0, h_new, 0), "memMap replacement")
chk(cu.cuMemSetAccess(int(va) + i * ROW, ROW, [acc], 1), "setAccess replacement")
rows[i].fill_(9.0); torch.cuda.synchronize()
ok6 = bool(torch.all(rows[i][::4096].float() == 9.0)) and rows.data_ptr() == int(va)
gb_after, _ = bw(rows[i:i+1])
step("6_remap_same_va", ok6, row1_now_GBps=round(gb_after, 1))
chk(cu.cuMemRelease(handles[i]), "release old host handle")

out["summary"] = {"row_MiB": ROW / 2**20, "granularity": GRAN, "numa_id": int(numa_id),
                  "all_pass": all(v["pass"] for v in out["steps"].values()),
                  "torch": torch.__version__, "driver": None}
try:
    (drv,) = chk(cu.cuDriverGetVersion(), "driverVersion"); out["summary"]["driver"] = int(drv)
except Exception: pass
json.dump(out, open("/w/results/spike-vmm-egm.json", "w"), indent=1)
print(json.dumps(out["summary"]))
