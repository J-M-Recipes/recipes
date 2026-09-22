"""Measure host memory consumed by UVA offload under the two pin paths. In-process vLLM, 16-layer MiMo-Pro,
offload ~5 MoE layers (OFFGB=40). Reads /proc/meminfo before construction and after load; reports delta of
(MemTotal - MemAvailable) and the offloader's own reported bytes.
"""
import os, sys, json, torch, time
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"; os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
def mi():
    d = {}
    for l in open("/proc/meminfo"):
        k, v = l.split(":"); d[k] = int(v.split()[0]) * 1024
    return d
def used(d): return d["MemTotal"] - d["MemAvailable"]
b = mi()
from vllm import LLM
llm = LLM(model=sys.argv[1], tensor_parallel_size=1, max_model_len=4096, max_num_seqs=1, gpu_memory_utilization=0.6,
          hf_overrides={"num_hidden_layers": 16}, enforce_eager=True, trust_remote_code=True, moe_backend="marlin",
          offload_backend="uva", cpu_offload_gb=float(os.environ.get("OFFGB", "40")),
          cpu_offload_params=["routed_experts.w13_weight", "routed_experts.w2_weight"])
a = mi()
tag = sys.argv[2]
print(f"[{tag}] host used delta = {(used(a) - used(b)) / 2**30:.2f} GiB ; Mlocked {a['Mlocked'] / 2**30:.2f} GiB ; Shmem {a['Shmem'] / 2**30:.2f} GiB ; Unevictable {a['Unevictable'] / 2**30:.2f} GiB", flush=True)
