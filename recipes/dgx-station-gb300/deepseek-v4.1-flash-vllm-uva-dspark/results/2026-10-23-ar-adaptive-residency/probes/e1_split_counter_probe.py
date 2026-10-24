#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""E1 probe — what does the adaptive-residency counter cost inside the split
kernel we already run every MoE layer?

Reproduces `split_ids` (v19 hook: ids -> hot rows / cold rows through a
[E] int32 row map, one Triton launch per layer) and compares it with the same
kernel plus one `atomic_add` per routed lane into a fixed-address [L, E] int32
counter table — the cheapest observation primitive found so far for dynamic
expert residency (see PORT-56177-MXFP4.md, shape B).

Measured the way the lane runs it: 40 layer-launches captured in one CUDA
graph, replayed, so launch overhead is amortised exactly as in serving.

Outputs per-shape overhead in microseconds per decode step and as a percentage
of representative step times (14.6 ms = v18 C1 profile; 7 ms; 4 ms).

`--dry-run` prints the plan with the standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=0,
                   help="index among VISIBLE CUDA devices (the runner sets "
                        "CUDA_VISIBLE_DEVICES, so 0 is the device it picked)")
    p.add_argument("--layers", type=int, default=40)
    p.add_argument("--experts", type=int, default=384)
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--tokens", type=int, nargs="+", default=[6, 16, 32],
                   help="token counts per layer-step (6 ~ C1 k=5, 32 ~ C6)")
    p.add_argument("--replays", type=int, default=100)
    p.add_argument("--block", type=int, default=1024)
    p.add_argument("--min-free-gb", type=float, default=1.0)
    p.add_argument("--json", default="")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse()
    plan = {
        "probe": "e1_split_counter",
        "layers": args.layers,
        "experts": args.experts,
        "top_k": args.top_k,
        "lanes_per_layer": {t: t * args.top_k for t in args.tokens},
        "counter_bytes": args.layers * args.experts * 4,
        "graph": f"{args.layers} launches per replay x {args.replays} replays",
        "arms": ["base", "count"],
        "steps_ms": [14.6, 7.0, 4.0],
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    import torch

    if not torch.cuda.is_available():
        print("no CUDA device", file=sys.stderr)
        return 2
    free, total = torch.cuda.mem_get_info(args.device)
    free_gb = free / 1024**3
    print(f"[e1] device {args.device} ({torch.cuda.get_device_name(args.device)}): "
          f"{free_gb:.2f} GiB free of {total/1024**3:.1f}")
    if free_gb < args.min_free_gb:
        print(f"[e1] REFUSING: free {free_gb:.2f} GiB < {args.min_free_gb}",
              file=sys.stderr)
        return 3

    torch.cuda.set_device(args.device)
    dev = torch.device("cuda", args.device)
    import triton
    import triton.language as tl

    # --- kernels (v19 hook's _sk, verbatim, plus a counting variant) --------
    @triton.jit
    def sk_base(ids_ptr, rm_ptr, hot_ptr, cold_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < N
        i = tl.load(ids_ptr + offs, mask=m, other=-1)
        valid = i >= 0
        ic = tl.where(valid, i, 0)
        rm = tl.load(rm_ptr + ic, mask=m, other=0)
        is_hot = valid & (rm >= 0)
        is_cold = valid & (rm < 0)
        neg1 = tl.full([BLOCK], -1, tl.int32)
        tl.store(hot_ptr + offs, tl.where(is_hot, rm, neg1), mask=m)
        tl.store(cold_ptr + offs, tl.where(is_cold, -rm - 1, neg1), mask=m)

    @triton.jit
    def sk_count(ids_ptr, rm_ptr, hot_ptr, cold_ptr, cnt_ptr, N,
                 BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < N
        i = tl.load(ids_ptr + offs, mask=m, other=-1)
        valid = i >= 0
        ic = tl.where(valid, i, 0)
        rm = tl.load(rm_ptr + ic, mask=m, other=0)
        is_hot = valid & (rm >= 0)
        is_cold = valid & (rm < 0)
        neg1 = tl.full([BLOCK], -1, tl.int32)
        tl.store(hot_ptr + offs, tl.where(is_hot, rm, neg1), mask=m)
        tl.store(cold_ptr + offs, tl.where(is_cold, -rm - 1, neg1), mask=m)
        # The whole observation primitive: one RMW per routed lane.
        tl.atomic_add(cnt_ptr + ic, 1, mask=m & valid)

    @triton.jit
    def sk_count_relaxed(ids_ptr, rm_ptr, hot_ptr, cold_ptr, cnt_ptr, flag_ptr, N,
                         BLOCK: tl.constexpr):
        # v26ar.1 shape: device flag + relaxed (fence-free) reduction.
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < N
        i = tl.load(ids_ptr + offs, mask=m, other=-1)
        valid = i >= 0
        ic = tl.where(valid, i, 0)
        rm = tl.load(rm_ptr + ic, mask=m, other=0)
        is_hot = valid & (rm >= 0)
        is_cold = valid & (rm < 0)
        neg1 = tl.full([BLOCK], -1, tl.int32)
        tl.store(hot_ptr + offs, tl.where(is_hot, rm, neg1), mask=m)
        tl.store(cold_ptr + offs, tl.where(is_cold, -rm - 1, neg1), mask=m)
        on = tl.load(flag_ptr)
        tl.atomic_add(cnt_ptr + ic, tl.full([BLOCK], 1, tl.int32),
                      mask=m & valid & (on != 0), sem="relaxed")

    E, L = args.experts, args.layers
    rm = torch.empty(E, dtype=torch.int32, device=dev)
    # A plausible static map: 295 hot rows, 89 cold (v15 geometry).
    hot_eids = torch.randperm(E, device=dev)[:295]
    cold_eids = torch.randperm(E, device=dev)[: E - 295]
    rm[hot_eids] = torch.arange(295, device=dev, dtype=torch.int32)
    rm[cold_eids] = -(torch.arange(1, E - 295 + 1, device=dev, dtype=torch.int32))

    cnt = torch.zeros((L, E), dtype=torch.int32, device=dev)
    flag = torch.ones(1, dtype=torch.int32, device=dev)
    results = []
    for tokens in args.tokens:
        n = tokens * args.top_k
        ids = torch.randint(0, E, (L, n), dtype=torch.int32, device=dev)
        hot = torch.empty((L, n), dtype=torch.int32, device=dev)
        cold = torch.empty((L, n), dtype=torch.int32, device=dev)
        grid = (triton.cdiv(n, args.block),)

        def launch_base(layer):
            sk_base[grid](ids[layer], rm, hot[layer], cold[layer], n,
                          BLOCK=args.block)

        def launch_count(layer):
            sk_count[grid](ids[layer], rm, hot[layer], cold[layer], cnt[layer],
                           n, BLOCK=args.block)

        def launch_relaxed(layer):
            sk_count_relaxed[grid](ids[layer], rm, hot[layer], cold[layer],
                                   cnt[layer], flag, n, BLOCK=args.block)

        def capture(fn):
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    for layer in range(L):
                        fn(layer)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for layer in range(L):
                    fn(layer)
            return g

        def replay_ms(g, replays):
            for _ in range(5):
                g.replay()
            torch.cuda.synchronize()
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(replays):
                g.replay()
            b.record()
            torch.cuda.synchronize()
            return a.elapsed_time(b) / replays

        g0 = capture(launch_base)
        g1 = capture(launch_count)
        g2 = capture(launch_relaxed)
        # interleave arms to cancel drift: 3 rounds
        t0s, t1s, t2s = [], [], []
        for _ in range(3):
            t0s.append(replay_ms(g0, args.replays))
            t1s.append(replay_ms(g1, args.replays))
            t2s.append(replay_ms(g2, args.replays))
        t0, t1, t2 = min(t0s), min(t1s), min(t2s)
        delta_us = (t1 - t0) * 1000
        delta_rx_us = (t2 - t0) * 1000
        row = {
            "tokens": tokens,
            "lanes_per_layer": n,
            "base_ms_per_step": round(t0, 4),
            "count_ms_per_step": round(t1, 4),
            "relaxed_ms_per_step": round(t2, 4),
            "delta_relaxed_us_per_step": round(delta_rx_us, 2),
            "pct_relaxed_of_14p6ms": round(100 * delta_rx_us / 14600, 3),
            "delta_us_per_step": round(delta_us, 2),
            "pct_of_14p6ms": round(100 * delta_us / 14600, 3),
            "pct_of_7ms": round(100 * delta_us / 7000, 3),
            "pct_of_4ms": round(100 * delta_us / 4000, 3),
        }
        results.append(row)
        print(f"[e1] tokens={tokens:3d} lanes={n:4d} "
              f"base={t0*1000:7.1f}us acq_rel={t1*1000:7.1f}us relaxed={t2*1000:7.1f}us "
              f"delta={delta_us:6.2f}us delta_relaxed={delta_rx_us:6.2f}us  "
              f"({row['pct_of_14p6ms']:.3f}% of 14.6ms, "
              f"{row['pct_of_7ms']:.3f}% of 7ms)")

    out = {
        "probe": "e1_split_counter",
        "device_name": torch.cuda.get_device_name(args.device),
        "layers": L,
        "experts": E,
        "results": results,
    }
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[e1] wrote {args.json}")
    else:
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
