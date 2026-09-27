#!/usr/bin/env python3
"""Per-decode-step breakdown from a vLLM torch trace. Finds decode steps by the CUDA-graph replay
markers (cudaGraphLaunch) or by clustering kernel timestamps into gaps > GAP_US, then reports the
median step's wall, gpu-busy, idle, and per-role ms. Also prints the top kernels of the median step.
usage: prof_step.py <trace.gz> [label]"""
import gzip, json, sys, re, collections, statistics
GAP_US = 400.0   # a decode step at C1 is ~6 ms; gaps between steps (CPU scheduling) are > 0.4 ms

def role(n):
    n = n.lower()
    if "bmm_mx" in n or "bmm_bfloat16_mx" in n: return "moe_expert_gemm"
    if any(k in n for k in ("moe", "routing", "topk", "expert", "finalize", "fma")): return "moe_other"
    if "mhc" in n: return "mhc"
    if "engram" in n or "ngram" in n or "hash" in n: return "engram"
    if any(k in n for k in ("mla", "flash_fwd", "attention", "attn", "indexer", "sparse", "get_mla_metadata", "combine")): return "attention"
    if any(k in n for k in ("gemm", "nvjet", "cutlass", "xmma", "matmul")): return "dense_gemm"
    if any(k in n for k in ("memcpy", "copy", "dtoh", "htod", "dtod")): return "memcpy"
    if any(k in n for k in ("norm",)): return "norm"
    if any(k in n for k in ("sampl", "argmax", "softmax", "rejection", "verify", "draft", "dspark", "spec")): return "sampling_spec"
    return "elementwise_other"

path = sys.argv[1]; label = sys.argv[2] if len(sys.argv) > 2 else path.split("/")[-2]
with gzip.open(path, "rt") as f: tr = json.load(f)
ev = tr["traceEvents"] if isinstance(tr, dict) else tr
kern = sorted((e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e), key=lambda e: e["ts"])
# cluster into steps by gap
steps = []; cur = [kern[0]]
for a, b in zip(kern, kern[1:]):
    if b["ts"] - (a["ts"] + a["dur"]) > GAP_US: steps.append(cur); cur = [b]
    else: cur.append(b)
steps.append(cur)
# keep steady-state decode steps: drop first/last 3, drop steps with > 3x median kernel count (prefill)
cnts = [len(s) for s in steps]; medc = statistics.median(cnts)
dec = [s for s in steps[3:-3] if 0.5 * medc <= len(s) <= 1.5 * medc]
def wall(s): return (s[-1]["ts"] + s[-1]["dur"] - s[0]["ts"]) / 1000
def busy(s): return sum(e["dur"] for e in s) / 1000
walls = sorted(wall(s) for s in dec)
gaps = []
for a, b in zip(steps, steps[1:]):
    gaps.append((b[0]["ts"] - (a[-1]["ts"] + a[-1]["dur"])) / 1000)
print(f"\n### {label}: {len(steps)} clusters, {len(dec)} steady decode steps, kernels/step median {medc:.0f}")
print(f"step wall ms: p10 {walls[int(.1*len(walls))]:.2f} p50 {walls[len(walls)//2]:.2f} p90 {walls[int(.9*len(walls))]:.2f} | inter-step gap p50 {statistics.median(gaps):.2f} ms | busy/wall p50 {statistics.median(busy(s)/wall(s) for s in dec)*100:.0f}%")
by = collections.defaultdict(list); names = collections.defaultdict(list)
for s in dec:
    r = collections.Counter(); nm = collections.Counter()
    for e in s: r[role(e["name"])] += e["dur"]; nm[e["name"]] += e["dur"]
    for k, v in r.items(): by[k].append(v / 1000)
    for k, v in nm.items(): names[k].append(v / 1000)
tot = sum(statistics.median(v) for v in by.values())
print(f"{'role':18} {'ms/step':>8} {'share':>6}")
for k, v in sorted(by.items(), key=lambda x: -statistics.median(x[1])):
    m = statistics.median(v) if len(v) == len(dec) else sum(v) / len(dec)
    print(f"{k:18} {m:8.2f} {m/tot*100:5.0f}%")
print("top kernels (median ms/step, present-in-steps):")
for k, v in sorted(names.items(), key=lambda x: -sum(x[1]))[:16]:
    print(f"  {sum(v)/len(dec):6.2f}  {len(v):4d}/{len(dec)}  {role(k):16} {k[:110]}")
