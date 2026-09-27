"""T4 Step 0: does GPU read bandwidth of pinned host memory (cudaHostAlloc, the UVA
path) change with transparent huge pages on the shmem backing?

Two access patterns that mirror the lane:
  gather : Engram-style, random 256-byte rows from a large table (translation-bound)
  stream : expert-style, 17.9 MiB contiguous reads at random 2 MiB-aligned offsets

Prints JSON. Run once per shmem_enabled setting. Reports whether the region
actually got huge pages (ShmemPmdMapped / mTHP counters from /proc/self/smaps).
"""
import json, os, sys, time
import torch, triton, triton.language as tl

GIB = 1 << 30
SIZE_GIB = float(os.environ.get("SIZE_GIB", "16"))
ROW = 256
ROWS_PER_STEP = int(os.environ.get("ROWS", "65536"))     # ~ Engram-ish batch of rows
EXPERT_BYTES = 17_930_000
STREAM_READS = int(os.environ.get("STREAMS", "48"))       # ~ 8 experts x 6 layers
ITERS = int(os.environ.get("ITERS", "20"))

def smaps_for(ptr, nbytes):
    out = {"Rss": 0, "ShmemPmdMapped": 0, "AnonHugePages": 0, "FilePmdMapped": 0}
    lo, hi = ptr, ptr + nbytes
    cur = None
    with open("/proc/self/smaps") as f:
        for line in f:
            if "-" in line.split()[0] and not line.startswith(("VmFlags", "Size")):
                a, b = line.split()[0].split("-")
                a, b = int(a, 16), int(b, 16)
                cur = (a < hi and b > lo)
                continue
            if cur:
                k = line.split(":")[0]
                if k in out:
                    out[k] += int(line.split()[1])   # kB
    return {k: v / 1048576 for k, v in out.items()}  # GiB

@triton.jit
def gather_kernel(base, idx, out, nrows, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < nrows
    r = tl.load(idx + offs, mask=m, other=0).to(tl.int64)
    acc = tl.zeros([BLOCK], dtype=tl.int64)
    for j in tl.static_range(0, 256, 8):
        acc += tl.load(base + r * 256 + j, mask=m, other=0).to(tl.int64)
    tl.store(out + offs, acc, mask=m)

MODE = os.environ.get("MODE", "hostalloc")   # hostalloc | reg_thp | reg_4k | reg_hugetlb
_keep = []

def alloc_host(n):
    """Return (cpu_tensor, extra) for the requested MODE."""
    import ctypes, mmap
    if MODE == "hostalloc":
        t = torch.empty(n, dtype=torch.uint8, pin_memory=True)   # cudaHostAlloc
        return t, {}
    flags = mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS
    if MODE == "reg_hugetlb":
        flags |= getattr(mmap, "MAP_HUGETLB", 0x40000)
    mm = mmap.mmap(-1, n, flags=flags, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    if MODE == "reg_thp":
        mm.madvise(mmap.MADV_HUGEPAGE)
    _keep.append(mm)
    t = torch.frombuffer(mm, dtype=torch.uint8)
    t.view(torch.int32).fill_(0x01010101)     # fault pages in BEFORE register
    cudart = ctypes.CDLL("libcudart.so.13")
    cudaHostRegisterPortable, cudaHostRegisterMapped = 1, 2
    rc = cudart.cudaHostRegister(ctypes.c_void_p(t.data_ptr()), ctypes.c_size_t(n),
                                 ctypes.c_uint(cudaHostRegisterPortable | cudaHostRegisterMapped))
    return t, {"cudaHostRegister_rc": int(rc)}

def main():
    dev = torch.device("cuda")
    n = int(SIZE_GIB * GIB)
    t0 = time.time()
    if MODE == "hbm":
        host = torch.full((n,), 1, dtype=torch.uint8, device=dev); extra = {}
    else:
        host, extra = alloc_host(n)
    if MODE == "hostalloc":
        host.view(torch.int32).fill_(0x01010101)                  # touch every page
    alloc_s = time.time() - t0
    ptr = host.data_ptr()
    pages = smaps_for(ptr, n) if MODE != "hbm" else {}
    if extra.get("cudaHostRegister_rc", 0) != 0:
        print(json.dumps({"mode": MODE, "error": "cudaHostRegister failed", **extra})); return
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor as viewfn  # noqa
    dview = host if MODE == "hbm" else viewfn(host)

    nrows = n // ROW
    res = {"mode": MODE, "size_gib": SIZE_GIB, "alloc_touch_s": round(alloc_s, 2),
           "is_pinned": bool(host.is_pinned()), "pages_gib": pages, **extra}

    # --- gather
    out = torch.empty(ROWS_PER_STEP, dtype=torch.int64, device=dev)
    idxs = [torch.randint(0, nrows, (ROWS_PER_STEP,), device=dev, dtype=torch.int32) for _ in range(ITERS)]
    grid = (triton.cdiv(ROWS_PER_STEP, 256),)
    for i in range(3):
        gather_kernel[grid](dview, idxs[i], out, ROWS_PER_STEP, BLOCK=256)
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
    st.record()
    for i in range(ITERS):
        gather_kernel[grid](dview, idxs[i], out, ROWS_PER_STEP, BLOCK=256)
    en.record(); torch.cuda.synchronize()
    ms = st.elapsed_time(en) / ITERS
    res["gather"] = {"rows": ROWS_PER_STEP, "ms": round(ms, 3),
                     "gbps_useful": round(ROWS_PER_STEP * ROW / ms / 1e6, 1),
                     "ns_per_row": round(ms * 1e6 / ROWS_PER_STEP, 1)}

    # --- stream (contiguous expert-sized reads at random offsets)
    dst = torch.empty(EXPERT_BYTES, dtype=torch.uint8, device=dev)
    g = torch.Generator().manual_seed(0)
    offs = [int(x) for x in torch.randint(0, (n - EXPERT_BYTES) // (2 << 20), (STREAM_READS * ITERS,), generator=g)]
    def one(k):
        o = offs[k] * (2 << 20)
        dst.copy_(dview[o:o + EXPERT_BYTES], non_blocking=True)
    for k in range(4): one(k)
    torch.cuda.synchronize()
    st.record()
    for k in range(STREAM_READS * ITERS): one(k)
    en.record(); torch.cuda.synchronize()
    ms = st.elapsed_time(en) / ITERS
    res["stream"] = {"reads_per_iter": STREAM_READS, "ms": round(ms, 3),
                     "gbps": round(STREAM_READS * EXPERT_BYTES / ms / 1e6, 1)}

    with open("/sys/kernel/mm/transparent_hugepage/shmem_enabled") as f:
        res["shmem_enabled"] = f.read().strip()
    try:
        with open("/sys/kernel/mm/transparent_hugepage/hugepages-2048kB/shmem_enabled") as f:
            res["mthp_2m_shmem"] = f.read().strip()
    except OSError:
        res["mthp_2m_shmem"] = None
    print(json.dumps(res))

if __name__ == "__main__":
    main()
