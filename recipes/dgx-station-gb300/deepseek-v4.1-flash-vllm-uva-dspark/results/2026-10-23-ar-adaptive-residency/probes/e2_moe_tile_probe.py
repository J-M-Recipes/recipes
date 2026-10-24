#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""E2 probe — does the MXFP4 TRT-LLM MoE re-read a cold expert per M-tile, and
what does a UVA demand read cost versus a pool copy?

No model load, no serving-lane changes. It builds `--experts` experts at
DeepSeek-V4.1-Flash's real per-expert geometry (hidden 5120, intermediate
2304, MXFP4 -> 17.92 MiB per row) and sends M tokens to ONE of them per call,
cycling over `--rotate` distinct experts so consecutive calls never re-read the
same row. That matters: one 17.9 MiB row would otherwise sit in L2 across
iterations and the demand-read cost would vanish.

  hbm   weights in HBM (baseline)
  uva   the same weights in pinned host, read in place over C2C
  pool  host weights copied into HBM first, then the same call

Delta(uva - hbm) is the demand-read cost for one row. Divided by the
calibrated single-row read it gives the number of times the kernel pulled that
row during the call — flat in M means once (or L2 absorbed the later tiles),
stepping with ceil(M / tile) means once per M-tile. That is the fact that
decides whether a copy-on-miss expert pool can ever beat a demand read on a
coherent link, and at which M.

`--dry-run` prints the plan with the standard library only (no CUDA, no torch).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys

MIB = 1024 * 1024


def row_bytes(hidden: int, inter: int) -> int:
    """One MXFP4 expert row: w13 + w2 + their ue8m0 block scales."""
    k, n = hidden, inter
    return 2 * n * (k // 2) + k * (n // 2) + 2 * n * (k // 32) + k * (n // 32)


def plan(args) -> dict:
    rb = row_bytes(args.hidden, args.inter)
    rot = min(args.rotate, args.experts)
    return {
        "device": args.device,
        "experts": args.experts,
        "rotate": rot,
        "hidden": args.hidden,
        "intermediate": args.inter,
        "top_k": args.top_k,
        "row_mib": round(rb / MIB, 3),
        "weights_mib": round(rb * args.experts / MIB, 1),
        "pinned_host_mib": round(rb * args.experts / MIB, 1),
        "rotating_working_set_mib": round(rb * rot / MIB, 1),
        "m_values": args.m,
        "routes_to_routed_expert": args.m,
        "tiles_lower_bound": {m: math.ceil(m / 8) for m in args.m},
        "arms": ["hbm", "uva", "pool"],
        "note": "each call routes all lanes to one expert; consecutive calls "
        "use a different expert, so the working set (rotate x 17.93 MiB) "
        "exceeds a 126 MB L2 and rows are not re-read from cache.",
    }


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=0,
                   help="index among VISIBLE CUDA devices (run_probe.sh picks "
                        "one device, so this is 0)")
    p.add_argument("--experts", type=int, default=16)
    p.add_argument("--rotate", type=int, default=16,
                   help="distinct experts cycled across timed iterations; "
                        "keep experts*17.9 MiB above the GPU's L2")
    p.add_argument("--hidden", type=int, default=5120)
    p.add_argument("--inter", type=int, default=2304)
    p.add_argument("--top-k", type=int, default=1,
                   help="1 keeps one route per token, so the routed expert "
                        "owns exactly M rows and the GEMM stays read-bound")
    p.add_argument("--m", type=int, nargs="+",
                   default=[6, 8, 16, 24, 32, 48, 64, 96, 128])
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--min-free-gb", type=float, default=2.0)
    p.add_argument("--json", default="")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse()
    if args.dry_run:
        print(json.dumps(plan(args), indent=2))
        return 0

    import torch

    if not torch.cuda.is_available():
        print("no CUDA device", file=sys.stderr)
        return 2
    free, total = torch.cuda.mem_get_info(args.device)
    free_gb = free / 1024**3
    print(f"[e2] device {args.device}: {free_gb:.2f} GiB free of "
          f"{total/1024**3:.1f}")
    if free_gb < args.min_free_gb:
        print(f"[e2] REFUSING: free {free_gb:.2f} GiB < --min-free-gb "
              f"{args.min_free_gb}.", file=sys.stderr)
        return 3

    from flashinfer import trtllm_fp4_block_scale_routed_moe
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        activation_to_flashinfer_int,
    )
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    import triton
    import triton.language as tl

    torch.cuda.set_device(args.device)
    dev = torch.device("cuda", args.device)
    cap = torch.cuda.get_device_capability(args.device)
    E, K, N, TK = args.experts, args.hidden, args.inter, args.top_k
    ROT = min(args.rotate, E)
    rb = row_bytes(K, N)
    rb_mib = rb / MIB
    l2_mib = torch.cuda.get_device_properties(args.device).L2_cache_size / MIB
    print(f"[e2] {torch.cuda.get_device_name(args.device)} sm_{cap[0]}{cap[1]} | "
          f"L2 {l2_mib:.0f} MiB | row {rb_mib:.2f} MiB | {E} experts, {ROT} "
          f"rotating ({rb_mib*ROT:.0f} MiB working set)")

    # Shapes from TrtLlmMxfp4ExpertsBase's assertions: w13 [E, 2N, K/2],
    # w13_scale [E, 2N, K/32], w2 [E, K, N/2], w2_scale [E, K, N/32]; uint8
    # packed fp4 and ue8m0 scales. Values are timing-only; scales are 127
    # (=1.0) so nothing goes non-finite and perturbs the instruction mix.
    shapes = {
        "w13": (E, 2 * N, K // 2),
        "w13s": (E, 2 * N, K // 32),
        "w2": (E, K, N // 2),
        "w2s": (E, K, N // 32),
    }
    fill = {"w13": 0x21, "w13s": 127, "w2": 0x21, "w2s": 127}

    hbm_u8, host_u8, uva_u8 = {}, {}, {}
    bank_hbm, bank_uva = {}, {}
    for name, shape in shapes.items():
        t = torch.empty(shape, dtype=torch.uint8, device=dev)
        t.fill_(fill[name])
        hbm_u8[name] = t
        h = torch.empty(shape, dtype=torch.uint8, pin_memory=True)
        h.fill_(fill[name])
        host_u8[name] = h
        u = get_accelerator_view_from_cpu_tensor(h)
        uva_u8[name] = u
        # The kernel checks gemm*_weights_scale.dtype() == float8_e4m3fn even
        # though the bytes are ue8m0 (E8M0) block exponents; vLLM passes the
        # same uint8 storage under that dtype tag. Weights stay uint8-packed.
        if name.endswith("s"):
            bank_hbm[name] = t.view(torch.float8_e4m3fn)
            bank_uva[name] = u.view(torch.float8_e4m3fn)
        else:
            bank_hbm[name] = t
            bank_uva[name] = u

    alpha = torch.full((E,), 1.702, dtype=torch.float32, device=dev)
    beta = torch.full((E,), 1.0, dtype=torch.float32, device=dev)
    clamp = torch.full((E,), 7.0, dtype=torch.float32, device=dev)
    out = torch.empty((max(args.m), K), dtype=torch.bfloat16, device=dev)
    tune_max = max(args.m)

    def call(ids, w, x, bank):
        return trtllm_fp4_block_scale_routed_moe(
            topk_ids=(ids, w),
            routing_bias=None,
            hidden_states=x,
            hidden_states_scale=None,
            gemm1_weights=bank["w13"],
            gemm1_weights_scale=bank["w13s"],
            gemm1_bias=None,
            gemm1_alpha=alpha,
            gemm1_beta=beta,
            gemm1_clamp_limit=clamp,
            gemm2_weights=bank["w2"],
            gemm2_weights_scale=bank["w2s"],
            gemm2_bias=None,
            output1_scale_scalar=None,
            output1_scale_gate_scalar=None,
            output2_scale_scalar=None,
            num_experts=E,
            top_k=TK,
            n_group=None,
            topk_group=None,
            intermediate_size=N,
            local_expert_offset=0,
            local_num_experts=E,
            routed_scaling_factor=None,
            routing_method_type=int(RoutingMethodType.Renormalize),
            do_finalize=True,
            enable_pdl=True,
            activation_type=activation_to_flashinfer_int(MoEActivation.SILU),
            output=out[: ids.shape[0]],
            tune_max_num_tokens=tune_max,
        )

    def time_rot(fn_of_row, iters: int) -> float:
        """Median ms per call, cycling the routed expert each iteration."""
        for k in range(ROT):
            fn_of_row(k)
        torch.cuda.synchronize()
        ts = []
        for i in range(iters):
            k = i % ROT
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record()
            fn_of_row(k)
            b.record()
            torch.cuda.synchronize()
            ts.append(a.elapsed_time(b))
        return statistics.median(ts)

    def copy_row(k):
        for name in hbm_u8:
            hbm_u8[name][k].copy_(host_u8[name][k], non_blocking=True)

    # --- calibration: one row, copy engine and SM --------------------------
    @triton.jit
    def _read_row(src, out_ptr, n_words, PROGRAMS: tl.constexpr,
                  BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        stride = tl.num_programs(0) * BLOCK
        acc = tl.zeros((BLOCK,), tl.int32)
        for start in range(pid * BLOCK, n_words, stride):
            offs = start + tl.arange(0, BLOCK)
            acc += tl.load(src + offs, mask=offs < n_words, other=0)
        tl.store(out_ptr + pid, tl.sum(acc, 0))

    scratch = torch.zeros(128, dtype=torch.int32, device=dev)
    uva_words = {n: t.view(torch.int32).reshape(E, -1) for n, t in uva_u8.items()}

    def sm_read(k):
        for n, w in uva_words.items():
            _read_row[(128,)](w[k], scratch, w.shape[1], PROGRAMS=128, BLOCK=1024)

    ce_us = time_rot(copy_row, 30) * 1000
    sm_us = time_rot(sm_read, 30) * 1000
    print(f"[e2] one row: copy-engine H2D {ce_us:.1f} us "
          f"({rb_mib/1024/(ce_us/1e6):.0f} GB/s) | SM stream read {sm_us:.1f} us "
          f"({rb_mib/1024/(sm_us/1e6):.0f} GB/s)")

    rows = []
    for m in args.m:
        ids = [torch.full((m, TK), k, dtype=torch.int32, device=dev)
               for k in range(ROT)]
        w = torch.full((m, TK), 1.0 / TK, dtype=torch.float32, device=dev)
        x = torch.randn((m, K), dtype=torch.bfloat16, device=dev) * 0.1

        t_hbm = time_rot(lambda k: call(ids[k], w, x, bank_hbm), args.iters)
        t_uva = time_rot(lambda k: call(ids[k], w, x, bank_uva), args.iters)
        t_copy = time_rot(copy_row, max(20, args.iters // 2))
        t_pool = time_rot(lambda k: (copy_row(k), call(ids[k], w, x, bank_hbm)),
                          args.iters)

        delta_us = (t_uva - t_hbm) * 1000
        implied_sm = delta_us / sm_us if sm_us > 0 else float("nan")
        implied_ce = delta_us / ce_us if ce_us > 0 else float("nan")
        rows.append({
            "m": m,
            "tile_lower_bound": math.ceil(m / 8),
            "hbm_ms": round(t_hbm, 4),
            "uva_ms": round(t_uva, 4),
            "delta_us": round(delta_us, 1),
            "implied_row_reads_vs_sm": round(implied_sm, 2),
            "implied_row_reads_vs_ce": round(implied_ce, 2),
            "copy_us": round(t_copy * 1000, 1),
            "pool_ms": round(t_pool, 4),
            "pool_minus_hbm_us": round((t_pool - t_hbm) * 1000, 1),
        })
        r = rows[-1]
        print(f"[e2] M={m:4d} tiles>={r['tile_lower_bound']:2d} "
              f"hbm={r['hbm_ms']*1000:7.1f}us uva={r['uva_ms']*1000:7.1f}us "
              f"delta={r['delta_us']:7.1f}us "
              f"reads(sm)~{r['implied_row_reads_vs_sm']:5.2f} "
              f"reads(ce)~{r['implied_row_reads_vs_ce']:5.2f} "
              f"copy={r['copy_us']:6.1f}us pool={r['pool_ms']*1000:7.1f}us "
              f"(pool-hbm={r['pool_minus_hbm_us']:+7.1f}us)")

    result = {
        "probe": "e2_moe_tile",
        "device": args.device,
        "device_name": torch.cuda.get_device_name(args.device),
        "capability": f"sm_{cap[0]}{cap[1]}",
        "l2_mib": round(l2_mib, 1),
        "row_mib": round(rb_mib, 3),
        "experts": E,
        "rotate": ROT,
        "top_k": TK,
        "calibration": {
            "copy_engine_us": round(ce_us, 1),
            "sm_read_us": round(sm_us, 1),
            "copy_engine_gbs": round(rb_mib / 1024 / (ce_us / 1e6), 1),
            "sm_read_gbs": round(rb_mib / 1024 / (sm_us / 1e6), 1),
        },
        "rows": rows,
    }
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[e2] wrote {args.json}")
    else:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
