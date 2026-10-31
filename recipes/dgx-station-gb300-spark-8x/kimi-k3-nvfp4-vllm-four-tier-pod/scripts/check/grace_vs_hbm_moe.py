#!/usr/bin/env python3
"""GB300: stock NVFP4 routed-MoE kernel at M=1, weights in HBM vs pinned Grace memory (UVA, read over C2C).
Times k = 1..16 local experts hit (rest -1), CUDA-graph captured so launch overhead is excluded."""
import json, torch
from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

H, I, E, TOPK = 3584, 3072, 64, 16
dev = "cuda"
torch.manual_seed(0)


def u8(*s):
    return torch.randint(0, 256, s, dtype=torch.uint8)


def e4(*s, b=0x18):
    return torch.full(s, b, dtype=torch.uint8)


def weights(where):
    w1, w2 = u8(E, 2 * I, H // 2), u8(E, H, I // 2)
    for w in (w1, w2):
        w.bitwise_and_(0x77)
    if where == "hbm":
        return w1.to(dev), w2.to(dev)
    return (get_accelerator_view_from_cpu_tensor(w1.pin_memory()),
            get_accelerator_view_from_cpu_tensor(w2.pin_memory()))


w1s = e4(E, 2 * I, H // 16).to(dev).view(torch.float8_e4m3fn)
w2s = e4(E, H, I // 16).to(dev).view(torch.float8_e4m3fn)
ones = torch.ones(E, dtype=torch.float32, device=dev)
x = u8(1, H // 2).to(dev).bitwise_and_(0x33)
xs = e4(1, H // 16, b=0x38).to(dev).view(torch.float8_e4m3fn)
out = torch.zeros(1, H, dtype=torch.bfloat16, device=dev)
wts = torch.full((1, TOPK), 1.0 / TOPK, dtype=torch.bfloat16, device=dev)
res = {}
for where in ("hbm", "grace"):
    w1, w2 = weights(where)
    res[where] = {}
    for k in (1, 2, 4, 6, 8, 12, 16):
        idss = [torch.tensor([torch.randperm(E)[:k].tolist() + [-1] * (TOPK - k)], dtype=torch.int32, device=dev)
                for _ in range(20)]          # a different expert set every call: L2 can't hold the working set
        cur = {"i": 0}

        def call():
            ids = idss[cur["i"] % 20]; cur["i"] += 1
            trtllm_fp4_block_scale_routed_moe(
                topk_ids=(ids, wts), routing_bias=None, hidden_states=x, hidden_states_scale=xs,
                gemm1_weights=w1, gemm1_weights_scale=w1s, gemm1_bias=None, gemm1_alpha=None, gemm1_beta=None,
                gemm1_clamp_limit=None, gemm2_weights=w2, gemm2_weights_scale=w2s, gemm2_bias=None,
                output1_scale_scalar=ones, output1_scale_gate_scalar=ones, output2_scale_scalar=ones,
                num_experts=E, top_k=TOPK, n_group=0, topk_group=0, intermediate_size=I,
                local_expert_offset=0, local_num_experts=E, routed_scaling_factor=None, routing_method_type=1,
                do_finalize=True, output=out, tune_max_num_tokens=8)
        for _ in range(5):
            call()
        torch.cuda.synchronize()
        gph = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            call()
            torch.cuda.synchronize()
            with torch.cuda.graph(gph, stream=s):
                for _ in range(20):
                    call()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); gph.replay(); e1.record(); e1.synchronize()
        us = e0.elapsed_time(e1) * 1e3 / 20
        res[where][k] = round(us, 1)
        print(where, k, round(us, 1), "us", flush=True)
    del w1, w2
print(json.dumps(res))
