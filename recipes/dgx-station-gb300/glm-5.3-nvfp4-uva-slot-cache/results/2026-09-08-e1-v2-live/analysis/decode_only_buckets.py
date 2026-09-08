#!/usr/bin/env python3
"""E1 v2 decode-only attribution from nsys cuda_gpu_kern_sum.csv.

The original E1 v2 bucket cut divided the whole captured GPU aggregate by the
140 verification steps. This helper separates the prompt/prefill rows by
instance count before printing the decode-only table used in OUTCOME.md.
"""
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

REQUESTS = 4
CACHED_LAYERS = 75
STEPS = 140
TOKENS = 256
DECODE_TOKENS_PER_S = 45.648
PROFILE_WALL_MS = 8217.965

BUCKET_ORDER = (
    "masked_row_copy",
    "dense_gemm",
    "other",
    "routed_moe",
    "fused_bookkeeping",
)


@dataclass
class Accum:
    total_ms: float = 0.0
    instances: int = 0
    rows: int = 0


def number(text: str) -> float:
    return float((text or "0").replace(",", ""))


def integer(text: str) -> int:
    return int((text or "0").replace(",", ""))


def classify(name: str) -> str:
    lowered = name.lower()
    if "masked_row_copy" in lowered:
        return "masked_row_copy"
    if "fused_bookkeeping" in lowered:
        return "fused_bookkeeping"
    if re.search(r"(^|[^a-z0-9_])bmm_e2m1", lowered) or re.search(r"(^|[^a-z0-9_])bmm_bfloat16_e2m1", lowered):
        return "routed_moe"
    if re.search(r"(^|[^a-z0-9_])nvjet", lowered) or "cublaslt" in lowered:
        return "dense_gemm"
    return "other"


def is_prefill_row(name: str, instances: int) -> bool:
    lowered = name.lower()
    prompt_moe = (
        (instances == REQUESTS * CACHED_LAYERS)
        and (
            re.search(r"(^|[^a-z0-9_])bmm_e2m1", lowered)
            or re.search(r"(^|[^a-z0-9_])bmm_bfloat16_e2m1", lowered)
        )
        and "t128x32x512" in lowered
    )
    prompt_dense = instances == REQUESTS and "bmm_bfloat16_bfloat16bfloat16" in lowered and "t128x32x128" in lowered
    return bool(prompt_moe or prompt_dense)


def default_report() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / "live-receipts" / "e1_cuda_gpu_kern_sum.csv",
        here / "e1_cuda_gpu_kern_sum.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("could not find live-receipts/e1_cuda_gpu_kern_sum.csv or analysis copy")


def summarize(path: Path) -> tuple[dict[str, Accum], Accum, float, int]:
    buckets = {bucket: Accum() for bucket in BUCKET_ORDER}
    prefill = Accum()
    full_total_ms = 0.0
    all_rows = 0

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            all_rows += 1
            name = row["Name"]
            instances = integer(row["Instances"])
            total_ms = number(row["Total Time (ns)"]) / 1_000_000.0
            full_total_ms += total_ms
            if is_prefill_row(name, instances):
                prefill.total_ms += total_ms
                prefill.instances += instances
                prefill.rows += 1
                continue
            bucket = classify(name)
            buckets[bucket].total_ms += total_ms
            buckets[bucket].instances += instances
            buckets[bucket].rows += 1

    # Keep scalar_gather/MLA/small residual kernels together as "other" in the
    # decode-only publication table; this is intentionally coarser than the
    # older corrected_nsys_bucket.py table.
    decode_total_ms = full_total_ms - prefill.total_ms
    assigned = sum(b.total_ms for name, b in buckets.items() if name != "other")
    buckets["other"].total_ms = decode_total_ms - assigned
    return buckets, prefill, full_total_ms, all_rows


def print_table(path: Path) -> None:
    buckets, prefill, full_total_ms, all_rows = summarize(path)
    decode_gpu_ms = full_total_ms - prefill.total_ms
    decode_wall_ms = TOKENS / DECODE_TOKENS_PER_S * 1000.0
    residual_ms = PROFILE_WALL_MS - decode_wall_ms

    print(f"source: {path}")
    print(f"rows: {all_rows}")
    decode_instances = sum(b.instances for b in buckets.values())
    print(f"prefill_rows: {prefill.rows} prefill_gpu_ms: {prefill.total_ms:.1f}")
    print(f"decode_kernel_instances_per_step: {decode_instances / STEPS:.1f}")
    print(f"decode_wall_ms_per_step: {decode_wall_ms / STEPS:.2f}")
    print(f"decode_gpu_ms_per_step: {decode_gpu_ms / STEPS:.2f}")
    print(f"residual_wall_ms: {residual_ms:.0f} ({residual_ms / REQUESTS:.0f} ms/request)")
    print()
    print("| bucket | ms/step | % decode GPU | instances | rows |")
    print("|---|---:|---:|---:|---:|")
    for name in BUCKET_ORDER:
        b = buckets[name]
        print(f"| {name} | {b.total_ms / STEPS:.2f} | {100.0 * b.total_ms / decode_gpu_ms:.1f}% | {b.instances:,} | {b.rows} |")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", nargs="?", type=Path, default=None)
    args = parser.parse_args()
    print_table(args.report or default_report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
