#!/usr/bin/env python3
"""Bucket an Nsight Systems CUDA kernel summary for the GLM-5.3 slot-cache lane.

This reports aggregate GPU work, not a critical-path decomposition. Different CUDA
streams may overlap; overlap and unaccounted wall time are therefore explicit.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Iterable

SCHEMA = "glm53-nsys-buckets-v1"
BUCKETS = (
    "fused_bookkeeping",
    "masked_row_copy",
    "routed_moe",
    "dense_gemm",
    "scalar_gather",
    "mla_attention",
    "mtp_verify",
    "other_gpu",
)


def classify_kernel(name: str) -> str:
    """Map a demangled CUDA kernel name to a stable experiment bucket."""
    lowered = name.lower()
    if "fused_bookkeeping" in lowered:
        return "fused_bookkeeping"
    if "masked_row_copy" in lowered:
        return "masked_row_copy"
    if re.search(r"(^|[^a-z0-9_])bmm_e2m1", lowered) or re.search(r"(^|[^a-z0-9_])bmm_bfloat16_e2m1", lowered):
        return "routed_moe"
    if re.search(r"(^|[^a-z0-9_])nvjet", lowered) or "cublaslt" in lowered:
        return "dense_gemm"
    if "routed_moe" in lowered or "moe_gemm" in lowered:
        return "routed_moe"
    if any(token in lowered for token in ("index_put", "indexselect", "index_select", "gather")):
        return "scalar_gather"
    if any(token in lowered for token in ("mtp", "speculat", "draft")):
        return "mtp_verify"
    if any(token in lowered for token in ("mla", "attention", "flash_fwd", "fmha")):
        return "mla_attention"
    return "other_gpu"


def _total_time_column(fieldnames: Iterable[str]) -> tuple[str, float]:
    for name in fieldnames:
        match = re.fullmatch(r"Total Time \((ns|us|µs|ms|s)\)", name.strip())
        if not match:
            continue
        unit = match.group(1)
        to_ms = {"ns": 1e-6, "us": 1e-3, "µs": 1e-3, "ms": 1.0, "s": 1e3}[unit]
        return name, to_ms
    raise ValueError("report has no supported 'Total Time (<unit>)' column")


def summarize_report(report: Path, *, steps: int, wall_ms: float) -> dict:
    """Parse cuda_gpu_kern_sum CSV and return bucketed aggregate work."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if wall_ms <= 0:
        raise ValueError("wall_ms must be positive")

    with report.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "Name" not in reader.fieldnames:
            raise ValueError("report has no Name column")
        total_column, to_ms = _total_time_column(reader.fieldnames)
        totals = {name: {"total_ms": 0.0, "instances": 0} for name in BUCKETS}
        kernel_rows = 0
        for row in reader:
            if not row.get("Name"):
                continue
            bucket = classify_kernel(row["Name"])
            totals[bucket]["total_ms"] += float(row[total_column].replace(",", "")) * to_ms
            totals[bucket]["instances"] += int(row.get("Instances", "0").replace(",", ""))
            kernel_rows += 1

    if kernel_rows == 0:
        raise ValueError("report contains no kernel rows")

    gpu_total_ms = sum(bucket["total_ms"] for bucket in totals.values())
    for bucket in totals.values():
        bucket["total_ms"] = round(bucket["total_ms"], 9)
        bucket["per_step_ms"] = round(bucket["total_ms"] / steps, 9)

    wall_residual_ms = max(0.0, wall_ms - gpu_total_ms)
    overlap_ms = max(0.0, gpu_total_ms - wall_ms)
    return {
        "schema": SCHEMA,
        "source_report": report.name,
        "methodology": "aggregate GPU kernel work; overlap and wall residual are not attributed to a kernel bucket",
        "steps": steps,
        "wall_ms": round(wall_ms, 9),
        "wall_per_step_ms": round(wall_ms / steps, 9),
        "gpu_total_ms": round(gpu_total_ms, 9),
        "gpu_per_step_ms": round(gpu_total_ms / steps, 9),
        "wall_residual_ms": round(wall_residual_ms, 9),
        "wall_residual_per_step_ms": round(wall_residual_ms / steps, 9),
        "overlap_ms": round(overlap_ms, 9),
        "accounted_wall_fraction": round(min(1.0, gpu_total_ms / wall_ms), 9),
        "kernel_rows": kernel_rows,
        "buckets": totals,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="nsys cuda_gpu_kern_sum CSV")
    parser.add_argument("--steps", type=int, required=True, help="profiled decode steps")
    parser.add_argument("--wall-ms", type=float, required=True, help="measured wall time for those steps")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        result = summarize_report(args.report, steps=args.steps, wall_ms=args.wall_ms)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        f"NSYS_BUCKETS gpu_ms={result['gpu_total_ms']:.3f} "
        f"wall_ms={result['wall_ms']:.3f} residual_ms={result['wall_residual_ms']:.3f} "
        f"overlap_ms={result['overlap_ms']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
