"""vllm#58178 follow-up: does PYTORCH_CUDA_ALLOC_CONF=pinned_max_cached_size_mb:N stop the
CachingHostAllocator's power-of-two rounding on Grace (GB300 Station, 494 GiB LPDDR5X),
and how does it compare with an exact-size cudaHostAlloc (what vllm#58185 does by default)?

One arm per process (the allocator setting is process-wide and read at first use).
  ARM=pin    : torch.Tensor.pin_memory()           (CachingHostAllocator; env knob applies)
  ARM=exact  : ctypes cudaHostAlloc(n, Mapped)      (exact size; #58185 default path)
Prints one JSON line. Host cost is measured three ways: process VmRSS delta, system
MemAvailable delta, and cgroup/driver-visible pinned via /proc/meminfo Shmem delta.
Optional GPU read check (READ=1): contiguous 17.9 MiB reads through the UVA view.
"""
import ctypes, json, os, time
import torch

GIB = 1 << 30
ARM = os.environ.get("ARM", "pin")
SIZE_GIB = float(os.environ.get("SIZE_GIB", "2.25"))
REALLOC_GIB = float(os.environ.get("REALLOC_GIB", "2"))
REALLOC_N = int(os.environ.get("REALLOC_N", "3"))
READ = os.environ.get("READ", "0") == "1"
STREAMS, ITERS, EXPERT_BYTES = 48, 10, 17_930_000


def meminfo():
    d = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":")
            d[k] = int(v.split()[0]) / 1048576  # GiB
    return d


def vmrss():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1048576


def pin_torch(src):
    return src.pin_memory()


def pin_exact(src):
    cudart = ctypes.CDLL("libcudart.so.13")
    p = ctypes.c_void_p()
    rc = cudart.cudaHostAlloc(ctypes.byref(p), ctypes.c_size_t(src.numel()), ctypes.c_uint(2))  # Mapped
    assert rc == 0, rc
    buf = (ctypes.c_uint8 * src.numel()).from_address(p.value)
    t = torch.frombuffer(buf, dtype=torch.uint8)
    t.copy_(src)
    return t


def main():
    torch.cuda.init()  # context up before baselines so it is not counted
    _ = torch.empty(1, device="cuda")
    torch.cuda.synchronize()
    n = int(SIZE_GIB * GIB)
    src = torch.empty(n, dtype=torch.uint8)
    src.fill_(1)
    m0, r0 = meminfo(), vmrss()
    t0 = time.time()
    t = pin_torch(src) if ARM == "pin" else pin_exact(src)
    torch.cuda.synchronize()
    alloc_s = time.time() - t0
    m1, r1 = meminfo(), vmrss()
    res = {
        "arm": ARM,
        "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        "torch": torch.__version__,
        "size_gib": SIZE_GIB,
        "alloc_s": round(alloc_s, 2),
        "is_pinned": bool(t.is_pinned()),
        "vmrss_delta_gib": round(r1 - r0, 3),
        "memavail_delta_gib": round(m0["MemAvailable"] - m1["MemAvailable"], 3),
        "shmem_delta_gib": round(m1["Shmem"] - m0["Shmem"], 3),
        "host_total_gib": round(m0["MemTotal"], 1),
    }
    del src

    # re-allocation tax: pin / free / pin of a REALLOC_GIB buffer, same process
    if ARM == "pin":
        times = []
        for _ in range(REALLOC_N):
            s = torch.empty(int(REALLOC_GIB * GIB), dtype=torch.uint8)
            t1 = time.time()
            p = s.pin_memory()
            torch.cuda.synchronize()
            times.append(round(time.time() - t1, 3))
            del p, s
        res["realloc"] = {"gib": REALLOC_GIB, "pin_s": times,
                          "memavail_after_gib": round(m0["MemAvailable"] - meminfo()["MemAvailable"], 3)}

    if READ:
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor as viewfn
        dview = viewfn(t)
        dst = torch.empty(EXPERT_BYTES, dtype=torch.uint8, device="cuda")
        g = torch.Generator().manual_seed(0)
        offs = [int(x) * (2 << 20) for x in torch.randint(0, (n - EXPERT_BYTES) // (2 << 20), (STREAMS * ITERS,), generator=g)]
        def one(k):
            dst.copy_(dview[offs[k]:offs[k] + EXPERT_BYTES], non_blocking=True)
        for k in range(4):
            one(k)
        torch.cuda.synchronize()
        st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
        st.record()
        for k in range(STREAMS * ITERS):
            one(k)
        en.record(); torch.cuda.synchronize()
        ms = st.elapsed_time(en) / ITERS
        res["stream_read"] = {"gpu": torch.cuda.get_device_name(), "ms_per_iter": round(ms, 3),
                              "gbps": round(STREAMS * EXPERT_BYTES / ms / 1e6, 1)}
    print(json.dumps(res))


if __name__ == "__main__":
    main()
