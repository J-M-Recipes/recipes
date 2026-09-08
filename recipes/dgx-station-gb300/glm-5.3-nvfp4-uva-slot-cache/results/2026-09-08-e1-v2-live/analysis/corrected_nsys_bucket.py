#!/usr/bin/env python3
"""Corrected Nsight CUDA kernel bucketing for E1 v2 GLM-5.3 NVFP4.

Buckets nsys `cuda_gpu_kern_sum` rows by demangled kernel name. This is
aggregate GPU work, not a critical-path decomposition; concurrent streams can
make aggregate GPU ms exceed wall ms.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

SCHEMA = "glm53-e1v2-corrected-nsys-buckets-v1"
BUCKETS = (
    "routed_moe",
    "dense_gemm",
    "masked_row_copy",
    "fused_bookkeeping",
    "scalar_gather",
    "mla_attention",
    "mtp_verify",
    "memcpy/memset",
    "other",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def classify_kernel(name: str) -> str:
    """Map a demangled CUDA kernel name to the corrected E1 v2 buckets."""
    lowered = name.lower()

    if "masked_row_copy" in lowered:
        return "masked_row_copy"
    if "fused_bookkeeping" in lowered:
        return "fused_bookkeeping"
    if any(token in lowered for token in ("scalar_gather", "index_put", "indexselect", "index_select", "gather")):
        return "scalar_gather"

    # NVFP4 MoE grouped GEMMs. These were previously missed because names are
    # template specializations beginning with bmm_E2m1/bmm_Bfloat16_E2m1 rather
    # than containing "routed_moe".
    if re.search(r"(^|[^a-z0-9_])bmm_e2m1", lowered) or re.search(r"(^|[^a-z0-9_])bmm_bfloat16_e2m1", lowered):
        return "routed_moe"

    # Dense GEMM implementations observed in this profile.
    if re.search(r"(^|[^a-z0-9_])nvjet", lowered) or "cublaslt" in lowered:
        return "dense_gemm"

    if any(token in lowered for token in ("mla", "attention", "flash_fwd", "fmha", "paged_attention")):
        return "mla_attention"
    if any(token in lowered for token in ("mtp", "speculat", "draft", "verify")):
        return "mtp_verify"
    if any(token in lowered for token in ("memcpy", "memset", "[cuda memcpy", "[cuda memset")):
        return "memcpy/memset"
    return "other"


def total_time_column(fieldnames: Iterable[str]) -> tuple[str, float]:
    for name in fieldnames:
        match = re.fullmatch(r"Total Time \((ns|us|µs|ms|s)\)", name.strip())
        if match:
            return name, {"ns": 1e-6, "us": 1e-3, "µs": 1e-3, "ms": 1.0, "s": 1e3}[match.group(1)]
    raise ValueError("report has no supported 'Total Time (<unit>)' column")


def as_number(text: str) -> float:
    return float((text or "0").replace(",", ""))


def as_int(text: str) -> int:
    return int((text or "0").replace(",", ""))


def summarize(report: Path, *, steps: int, wall_ms: float, previous_buckets: Path | None) -> dict:
    totals = {bucket: {"total_ms": 0.0, "instances": 0, "rows": 0, "top_kernels": []} for bucket in BUCKETS}
    kernel_rows = []

    with report.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "Name" not in reader.fieldnames:
            raise ValueError("report has no Name column")
        total_col, to_ms = total_time_column(reader.fieldnames)
        for csv_line, row in enumerate(reader, start=2):
            name = row.get("Name", "")
            if not name:
                continue
            bucket = classify_kernel(name)
            total_ms = as_number(row[total_col]) * to_ms
            instances = as_int(row.get("Instances", "0"))
            totals[bucket]["total_ms"] += total_ms
            totals[bucket]["instances"] += instances
            totals[bucket]["rows"] += 1
            kernel_rows.append({
                "csv_line": csv_line,
                "bucket": bucket,
                "name": name,
                "total_ms": total_ms,
                "instances": instances,
                "avg_ns": as_number(row.get("Avg (ns)", "0")),
                "min_ns": as_number(row.get("Min (ns)", "0")),
                "max_ns": as_number(row.get("Max (ns)", "0")),
            })

    gpu_total_ms = sum(v["total_ms"] for v in totals.values())
    for bucket, data in totals.items():
        data["total_ms"] = round(data["total_ms"], 9)
        data["per_step_ms"] = round(data["total_ms"] / steps, 9)
        data["pct_gpu_aggregate"] = round(100.0 * data["total_ms"] / gpu_total_ms, 9) if gpu_total_ms else 0.0
        top = sorted((r for r in kernel_rows if r["bucket"] == bucket), key=lambda r: r["total_ms"], reverse=True)[:8]
        data["top_kernels"] = [
            {
                "csv_line": r["csv_line"],
                "name": r["name"],
                "total_ms": round(r["total_ms"], 9),
                "per_step_ms": round(r["total_ms"] / steps, 9),
                "instances": r["instances"],
                "avg_ns": r["avg_ns"],
                "min_ns": r["min_ns"],
                "max_ns": r["max_ns"],
            }
            for r in top
        ]

    previous_other = None
    input_hashes = {str(report): sha256_file(report)}
    if previous_buckets:
        previous = json.loads(previous_buckets.read_text())
        input_hashes[str(previous_buckets)] = sha256_file(previous_buckets)
        prev_other_ms_step = previous["buckets"]["other_gpu"]["per_step_ms"]
        new_other_ms_step = totals["other"]["per_step_ms"]
        attributed_ms_step = prev_other_ms_step - new_other_ms_step
        previous_other = {
            "previous_bucket_file": str(previous_buckets),
            "previous_other_gpu_per_step_ms": prev_other_ms_step,
            "corrected_other_per_step_ms": new_other_ms_step,
            "newly_attributed_per_step_ms": round(attributed_ms_step, 9),
            "newly_attributed_fraction": round(attributed_ms_step / prev_other_ms_step, 9),
            "newly_attributed_percent": round(100.0 * attributed_ms_step / prev_other_ms_step, 6),
            "newly_attributed_buckets": ["routed_moe", "dense_gemm"],
        }

    return {
        "schema": SCHEMA,
        "source_report": str(report),
        "input_sha256": input_hashes,
        "methodology": "aggregate GPU kernel work from nsys cuda_gpu_kern_sum.csv; percentages are of aggregate GPU kernel time, not wall critical path",
        "steps": steps,
        "wall_ms": round(wall_ms, 9),
        "wall_per_step_ms": round(wall_ms / steps, 9),
        "gpu_total_ms": round(gpu_total_ms, 9),
        "gpu_per_step_ms": round(gpu_total_ms / steps, 9),
        "gpu_minus_wall_ms": round(gpu_total_ms - wall_ms, 9),
        "gpu_minus_wall_per_step_ms": round((gpu_total_ms - wall_ms) / steps, 9),
        "gpu_to_wall_ratio": round(gpu_total_ms / wall_ms, 9),
        "kernel_rows": len(kernel_rows),
        "bucket_order": list(BUCKETS),
        "buckets": totals,
        "previous_other_gpu_reattribution": previous_other,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--wall-ms", type=float, required=True)
    parser.add_argument("--previous-buckets", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = summarize(args.report, steps=args.steps, wall_ms=args.wall_ms, previous_buckets=args.previous_buckets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        "corrected_buckets "
        f"gpu_per_step_ms={result['gpu_per_step_ms']:.6f} "
        f"routed_moe_ms={result['buckets']['routed_moe']['per_step_ms']:.6f} "
        f"dense_gemm_ms={result['buckets']['dense_gemm']['per_step_ms']:.6f} "
        f"masked_row_copy_ms={result['buckets']['masked_row_copy']['per_step_ms']:.6f} "
        f"other_ms={result['buckets']['other']['per_step_ms']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
