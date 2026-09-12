#!/usr/bin/env python3
"""Decode-step breakdown from vLLM torch-profiler traces (rank0 *.pt.trace.json.gz).
Groups GPU kernels by role, sums per-trace, prints per-step ms and share. Steps are counted from the CUDA-graph
replay / 'execute_model' markers if present; otherwise from the number of ProfilerStep ranges."""
import gzip, json, sys, re, collections
def role(name):
    n = name.lower()
    if "graph" in n and "launch" in n: return "cudagraph_launch"
    if any(k in n for k in ("moe", "fc1", "fc2", "expert", "grouped", "block_scale", "bmm_fp4", "routing", "topk", "gemm1", "gemm2")): return "moe_experts"
    if any(k in n for k in ("attention", "attn", "mla", "flash", "decode_kernel", "paged", "sparse_mla", "indexer")): return "attention"
    if any(k in n for k in ("gemm", "matmul", "cutlass", "nvjet", "sm100", "xmma", "wgmma")): return "dense_gemm"
    if any(k in n for k in ("memcpy", "copy", "dtoh", "htod", "dtod")): return "memcpy"
    if any(k in n for k in ("norm", "rmsnorm", "layernorm")): return "norm"
    if any(k in n for k in ("rotary", "rope", "embedding", "silu", "act", "elementwise", "vectorized", "fill", "index", "scatter", "gather", "cat", "reduce", "softmax", "argmax", "sampl", "cast", "convert", "quant")): return "elementwise_misc"
    if "dspark" in n or "draft" in n or "mtp" in n: return "draft"
    return "other"
for path in sys.argv[1:]:
    with gzip.open(path, "rt") as f: tr = json.load(f)
    ev = tr["traceEvents"] if isinstance(tr, dict) else tr
    kern = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e]
    steps = [e for e in ev if e.get("cat") in ("user_annotation", "cpu_op", "gpu_user_annotation") and re.search(r"ProfilerStep|execute_model|model_forward|EngineCore", e.get("name", ""))]
    nsteps = len({e["name"] for e in ev if e.get("name","").startswith("ProfilerStep")}) or 0
    if not nsteps:
        # fall back: count graph replays or distinct forward annotations
        nsteps = len([e for e in ev if e.get("name","").startswith("ProfilerStep")]) or len([e for e in steps if "execute_model" in e.get("name","")]) or 1
    t0 = min(e["ts"] for e in kern); t1 = max(e["ts"] + e["dur"] for e in kern)
    wall_ms = (t1 - t0) / 1000
    by = collections.Counter(); names = collections.defaultdict(float); cnt = collections.Counter()
    for e in kern:
        r = role(e["name"]); by[r] += e["dur"]; names[e["name"]] += e["dur"]; cnt[r] += 1
    tot = sum(by.values()) / 1000
    print(f"\n### {path.split('/')[-1][:40]}  steps~{nsteps}  gpu-busy {tot:.1f} ms  wall {wall_ms:.1f} ms  ({tot/wall_ms*100:.0f}% busy)  per-step wall {wall_ms/nsteps:.2f} ms  gpu {tot/nsteps:.2f} ms")
    print(f"{'role':18} {'ms/step':>8} {'share':>6} {'kernels/step':>12}")
    for r, us in by.most_common():
        print(f"{r:18} {us/1000/nsteps:8.2f} {us/1000/tot*100:5.0f}% {cnt[r]/nsteps:12.0f}")
    print("top kernels (ms/step):")
    for n, us in sorted(names.items(), key=lambda x: -x[1])[:14]:
        print(f"  {us/1000/nsteps:6.2f}  {role(n):16} {n[:100]}")
