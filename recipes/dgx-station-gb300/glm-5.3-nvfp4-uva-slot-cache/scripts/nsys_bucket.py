#!/usr/bin/env python3
"""Bucket Nsight Systems CUDA kernel reports for the GLM-5.3 slot-cache lane.

Legacy mode reports aggregate GPU work from cuda_gpu_kern_sum CSV, not a
critical-path decomposition. Opt-in phase mode reports aggregate GPU work from
cuda_gpu_trace CSV inside explicit timestamp intervals; it does not infer decode
membership from kernel instance counts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable

SCHEMA = "glm53-nsys-buckets-v1"
PHASE_SCHEMA = "glm53-nsys-phase-buckets-v1"
INTERVAL_SCHEMA = "glm53-nsys-phase-intervals-v1"
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


def _unit_to_ms(unit: str) -> float:
    return {"ns": 1e-6, "us": 1e-3, "µs": 1e-3, "ms": 1.0, "s": 1e3}[unit]


def _unit_to_ns_decimal(unit: str) -> Decimal:
    return {"ns": Decimal(1), "us": Decimal(1000), "µs": Decimal(1000), "ms": Decimal(1000000), "s": Decimal(1000000000)}[unit]


def _total_time_column(fieldnames: Iterable[str]) -> tuple[str, float]:
    for name in fieldnames:
        match = re.fullmatch(r"Total Time \((ns|us|µs|ms|s)\)", name.strip())
        if not match:
            continue
        return name, _unit_to_ms(match.group(1))
    raise ValueError("report has no supported 'Total Time (<unit>)' column")


def _time_column(fieldnames: Iterable[str], label: str) -> tuple[str, str, float]:
    for name in fieldnames:
        match = re.fullmatch(rf"{re.escape(label)} \((ns|us|µs|ms|s)\)", name.strip())
        if match:
            unit = match.group(1)
            return name, unit, _unit_to_ms(unit)
    raise ValueError(f"gpu trace has no supported '{label} (<unit>)' column")


def _new_bucket_totals() -> dict:
    return {name: {"total_ms": 0.0, "instances": 0} for name in BUCKETS}


def _finish_buckets(totals: dict, steps: int | None = None) -> dict:
    for bucket in totals.values():
        bucket["total_ms"] = round(bucket["total_ms"], 9)
        if steps is not None:
            bucket["per_step_ms"] = round(bucket["total_ms"] / steps, 9)
    return totals


def _parse_int(raw: str | None, *, field: str, row_number: int) -> int:
    if raw is None or raw == "":
        raise ValueError(f"row {row_number}: missing {field}")
    try:
        return int(raw.replace(",", ""))
    except ValueError as exc:
        raise ValueError(f"row {row_number}: malformed {field}: {raw!r}") from exc


def _parse_number(raw, *, field: str, row_number: int, non_negative: bool = True, positive: bool = False) -> float:
    if raw is None or raw == "":
        raise ValueError(f"row {row_number}: missing {field}")
    try:
        value = float(str(raw).replace(",", ""))
    except ValueError as exc:
        raise ValueError(f"row {row_number}: malformed {field}: {raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"row {row_number}: {field} must be finite")
    if positive and value <= 0:
        raise ValueError(f"row {row_number}: {field} must be positive")
    if non_negative and value < 0:
        raise ValueError(f"row {row_number}: {field} must be non-negative")
    return value


def _parse_decimal(raw, *, field: str, row_number: int, non_negative: bool = True, positive: bool = False) -> Decimal:
    if raw is None or raw == "":
        raise ValueError(f"row {row_number}: missing {field}")
    try:
        value = Decimal(str(raw).replace(",", ""))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"row {row_number}: malformed {field}: {raw!r}") from exc
    if not value.is_finite():
        raise ValueError(f"row {row_number}: {field} must be finite")
    if positive and value <= 0:
        raise ValueError(f"row {row_number}: {field} must be positive")
    if non_negative and value < 0:
        raise ValueError(f"row {row_number}: {field} must be non-negative")
    return value


def _decimal_to_ms_float(value: Decimal, to_ms: float, *, field: str, row_number: int) -> float:
    ms_value = float(value) * to_ms
    if not math.isfinite(ms_value):
        raise ValueError(f"row {row_number}: {field} is too large to emit as finite milliseconds")
    return ms_value


def _parse_integral_ns(raw, *, field: str, row_number: int, unit: str = "ns", positive: bool = False) -> int:
    value = _parse_decimal(raw, field=field, row_number=row_number, positive=positive)
    ns_value = value * _unit_to_ns_decimal(unit)
    if ns_value != ns_value.to_integral_value():
        raise ValueError(f"row {row_number}: {field} must normalize to integral ns")
    return int(ns_value)


def _validate_steps(steps: int | None) -> None:
    if steps is not None and (not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0):
        raise ValueError("steps must be a finite positive integer")


def _validate_wall_ms(wall_ms: float | None) -> None:
    if wall_ms is not None:
        try:
            value = float(wall_ms)
        except (TypeError, ValueError) as exc:
            raise ValueError("wall_ms must be finite and positive") from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError("wall_ms must be finite and positive")


def _reject_duplicate_json_keys(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate JSON key: {key}")
        obj[key] = value
    return obj


def _read_text_and_sha256(path: Path) -> tuple[str, str]:
    data = path.read_bytes()
    return data.decode(), hashlib.sha256(data).hexdigest()


def _dict_reader_from_text(text: str, *, kind: str) -> csv.DictReader:
    reader = csv.DictReader(io.StringIO(text), strict=True)
    if not reader.fieldnames:
        raise ValueError(f"{kind} has no header")
    duplicate_headers = sorted({name for name in reader.fieldnames if reader.fieldnames.count(name) > 1})
    if duplicate_headers:
        raise ValueError(f"{kind} has duplicate header keys: {', '.join(duplicate_headers)}")
    return reader


def _reject_extra_csv_cells(row: dict, *, kind: str, row_number: int) -> None:
    if None in row:
        raise ValueError(f"{kind} row {row_number}: extra CSV cells")


def _pick_alias(item: dict, primary: str, alias: str, *, row_number: int):
    has_primary = primary in item
    has_alias = alias in item
    if has_primary and has_alias and item[primary] != item[alias]:
        raise ValueError(f"interval {row_number}: conflicting aliases {primary} and {alias}")
    if has_primary:
        return item[primary]
    if has_alias:
        return item[alias]
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_report(report: Path, *, steps: int, wall_ms: float) -> dict:
    """Parse cuda_gpu_kern_sum CSV and return bucketed aggregate work."""
    _validate_steps(steps)
    _validate_wall_ms(wall_ms)

    report_text, _ = _read_text_and_sha256(report)
    reader = _dict_reader_from_text(report_text, kind="report")
    if not reader.fieldnames or "Name" not in reader.fieldnames:
        raise ValueError("report has no Name column")
    total_column, to_ms = _total_time_column(reader.fieldnames)
    totals = _new_bucket_totals()
    kernel_rows = 0
    for row_number, row in enumerate(reader, start=2):
        _reject_extra_csv_cells(row, kind="report", row_number=row_number)
        if not row.get("Name"):
            continue
        bucket = classify_kernel(row["Name"])
        total_time = _parse_decimal(row.get(total_column), field=total_column, row_number=row_number)
        totals[bucket]["total_ms"] += _decimal_to_ms_float(total_time, to_ms, field=total_column, row_number=row_number)
        totals[bucket]["instances"] += _parse_int(row.get("Instances", "0"), field="Instances", row_number=row_number)
        kernel_rows += 1

    if kernel_rows == 0:
        raise ValueError("report contains no kernel rows")

    gpu_total_ms = sum(bucket["total_ms"] for bucket in totals.values())
    _finish_buckets(totals, steps)

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


def _load_intervals(path: Path, *, trace_sha256: str) -> tuple[list[dict], str, dict, str]:
    try:
        text, intervals_sha256 = _read_text_and_sha256(path)
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_float=Decimal,
            parse_int=Decimal,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"intervals JSON is malformed: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("intervals JSON must be an object")
    if payload.get("schema") != INTERVAL_SCHEMA:
        raise ValueError(f"intervals schema must be {INTERVAL_SCHEMA}")
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("intervals run_id must be a non-empty string")
    clock_domain = payload.get("clock_domain")
    if not isinstance(clock_domain, str) or not clock_domain.strip():
        raise ValueError("intervals clock_domain must be a non-empty string")
    declared_trace_sha256 = payload.get("trace_sha256")
    if declared_trace_sha256 != trace_sha256:
        raise ValueError("intervals trace_sha256 must match actual gpu trace sha256")
    intervals_payload = payload.get("intervals")
    if not isinstance(intervals_payload, list) or not intervals_payload:
        raise ValueError("intervals JSON must contain a non-empty intervals array")
    unit = payload.get("time_unit", "ns")
    if unit not in {"ns", "us", "µs", "ms", "s"}:
        raise ValueError("intervals time_unit must be one of ns, us, µs, ms, s")
    intervals = []
    for index, item in enumerate(intervals_payload):
        row_number = index + 1
        if not isinstance(item, dict):
            raise ValueError(f"interval {row_number}: must be an object")
        phase = item.get("phase")
        if not isinstance(phase, str) or not phase.strip():
            raise ValueError(f"interval {row_number}: phase must be a non-empty string")
        if unit != "ns" and ("start_ns" in item or "end_ns" in item):
            raise ValueError(f"interval {row_number}: _ns aliases require intervals time_unit ns")
        start = _parse_integral_ns(_pick_alias(item, "start", "start_ns", row_number=row_number), field="start", row_number=row_number, unit=unit)
        end = _parse_integral_ns(_pick_alias(item, "end", "end_ns", row_number=row_number), field="end", row_number=row_number, unit=unit)
        if end <= start:
            raise ValueError(f"interval {row_number}: end must be greater than start")
        interval = {"phase": phase.strip(), "start": start, "end": end}
        if "steps" in item:
            interval_steps = item["steps"]
            if isinstance(interval_steps, Decimal) and interval_steps == interval_steps.to_integral_value():
                interval_steps = int(interval_steps)
            if not isinstance(interval_steps, int) or isinstance(interval_steps, bool) or interval_steps <= 0:
                raise ValueError(f"interval {row_number}: steps must be a finite positive integer")
            interval["steps"] = interval_steps
        intervals.append(interval)
    intervals.sort(key=lambda interval: (interval["start"], interval["end"], interval["phase"]))
    previous = None
    for interval in intervals:
        if previous is not None and interval["start"] < previous["end"]:
            raise ValueError(
                f"intervals overlap: {previous['phase']} [{previous['start']}, {previous['end']}) and "
                f"{interval['phase']} [{interval['start']}, {interval['end']})"
            )
        previous = interval
    contract = {
        "run_id": run_id.strip(),
        "clock_domain": clock_domain.strip(),
        "trace_sha256": trace_sha256,
        "attribution": "external-explicit-intervals",
    }
    return intervals, unit, contract, intervals_sha256


def _locate_interval(start: int, end: int, intervals: list[dict]) -> dict | None:
    containing = [interval for interval in intervals if start >= interval["start"] and end <= interval["end"]]
    if len(containing) > 1:
        raise ValueError(f"kernel [{start}, {end}) has ambiguous interval membership")
    if containing:
        return containing[0]
    for interval in intervals:
        overlaps = start < interval["end"] and end > interval["start"]
        if overlaps:
            raise ValueError(
                f"kernel [{start}, {end}) crosses interval boundary for phase {interval['phase']} "
                f"[{interval['start']}, {interval['end']})"
            )
    return None


def _phase_shell() -> dict:
    return {"kernel_rows": 0, "gpu_total_ms": 0.0, "buckets": _new_bucket_totals()}


def _finish_phase(phase: dict, steps: int | None, wall_ms: float | None = None) -> dict:
    phase["gpu_total_ms"] = round(phase["gpu_total_ms"], 9)
    if steps is not None:
        phase["gpu_per_step_ms"] = round(phase["gpu_total_ms"] / steps, 9)
        if wall_ms is not None:
            phase["wall_ms"] = round(wall_ms, 9)
            phase["wall_per_step_ms"] = round(wall_ms / steps, 9)
            phase["accounted_wall_fraction"] = round(min(1.0, phase["gpu_total_ms"] / wall_ms), 9)
            phase["wall_residual_ms"] = round(max(0.0, wall_ms - phase["gpu_total_ms"]), 9)
            phase["wall_residual_per_step_ms"] = round(max(0.0, wall_ms - phase["gpu_total_ms"]) / steps, 9)
            phase["overlap_ms"] = round(max(0.0, phase["gpu_total_ms"] - wall_ms), 9)
    _finish_buckets(phase["buckets"], steps)
    return phase


def summarize_phase_trace(gpu_trace: Path, intervals_json: Path, *, steps: int | None = None, wall_ms: float | None = None) -> dict:
    """Parse cuda_gpu_trace CSV using explicit half-open timestamp intervals.

    Contract: the trace must contain Start (<unit>), Duration (<unit>), and Name
    or Kernel Name columns. Intervals JSON must contain a non-empty ``intervals``
    array with ``phase``, ``start``, and ``end`` values in ``time_unit`` units
    (default ns). Intervals are half-open [start, end), non-overlapping, finite,
    and non-negative. Kernel executions that cross an interval boundary raise an
    error; executions outside all intervals are counted only in outside_interval.
    """
    _validate_steps(steps)
    _validate_wall_ms(wall_ms)

    trace_text, trace_sha256 = _read_text_and_sha256(gpu_trace)
    intervals, interval_unit, interval_contract, intervals_sha256 = _load_intervals(intervals_json, trace_sha256=trace_sha256)
    scaled_intervals = intervals
    decode_intervals = [item for item in intervals if item["phase"] == "decode"]
    decode_intervals_with_steps = [item for item in decode_intervals if "steps" in item]
    declared_decode_steps = None
    if decode_intervals_with_steps:
        if len(decode_intervals_with_steps) != len(decode_intervals):
            raise ValueError("all decode intervals must declare steps when any decode interval declares steps")
        declared_decode_steps = sum(item["steps"] for item in decode_intervals_with_steps)
        if steps is not None and steps != declared_decode_steps:
            raise ValueError("--steps must match declared decode interval steps total")
    elif steps is not None:
        declared_decode_steps = steps

    phases: dict[str, dict] = {}
    for interval in intervals:
        phases.setdefault(interval["phase"], _phase_shell())
    outside = {"kernel_rows": 0, "gpu_total_ms": 0.0}
    kernel_rows = 0

    reader = _dict_reader_from_text(trace_text, kind="gpu trace")
    start_column, start_unit, _ = _time_column(reader.fieldnames, "Start")
    duration_column, duration_unit, duration_to_ms = _time_column(reader.fieldnames, "Duration")
    name_column = "Name" if "Name" in reader.fieldnames else "Kernel Name" if "Kernel Name" in reader.fieldnames else None
    if name_column is None:
        raise ValueError("gpu trace has no Name or Kernel Name column")
    for row_number, row in enumerate(reader, start=2):
        _reject_extra_csv_cells(row, kind="gpu trace", row_number=row_number)
        if "Name" in reader.fieldnames and "Kernel Name" in reader.fieldnames:
            if row.get("Name") and row.get("Kernel Name") and row["Name"] != row["Kernel Name"]:
                raise ValueError(f"row {row_number}: conflicting Name and Kernel Name aliases")
        name = row.get(name_column)
        if not name:
            raise ValueError(f"row {row_number}: missing {name_column}")
        start = _parse_integral_ns(row.get(start_column), field=start_column, row_number=row_number, unit=start_unit)
        duration_raw_decimal = _parse_decimal(row.get(duration_column), field=duration_column, row_number=row_number, positive=True)
        duration_ns = _parse_integral_ns(row.get(duration_column), field=duration_column, row_number=row_number, unit=duration_unit, positive=True)
        end = start + duration_ns
        interval = _locate_interval(start, end, scaled_intervals)
        kernel_rows += 1
        duration_ms = _decimal_to_ms_float(duration_raw_decimal, duration_to_ms, field=duration_column, row_number=row_number)
        if interval is None:
            outside["kernel_rows"] += 1
            outside["gpu_total_ms"] += duration_ms
            continue
        phase = phases[interval["phase"]]
        bucket = classify_kernel(name)
        phase["kernel_rows"] += 1
        phase["gpu_total_ms"] += duration_ms
        phase["buckets"][bucket]["total_ms"] += duration_ms
        phase["buckets"][bucket]["instances"] += 1

    if kernel_rows == 0:
        raise ValueError("gpu trace contains no kernel rows")

    for phase_name, phase in phases.items():
        _finish_phase(phase, declared_decode_steps if phase_name == "decode" else None, wall_ms if phase_name == "decode" else None)
    outside["gpu_total_ms"] = round(outside["gpu_total_ms"], 9)
    included_gpu_total_ms = sum(phase["gpu_total_ms"] for phase in phases.values())
    result = {
        "schema": PHASE_SCHEMA,
        "methodology": "timestamp interval analysis; kernels are bucketed only when wholly contained in one explicit interval",
        "inputs": {
            "gpu_trace": {"path": gpu_trace.name, "sha256": trace_sha256},
            "intervals": {"path": intervals_json.name, "sha256": intervals_sha256},
        },
        "interval_contract": {
            **interval_contract,
            "time_unit": interval_unit,
            "semantics": "half-open [start, end); non-overlapping; kernel executions crossing any interval boundary are errors",
            "intervals": intervals,
        },
        "kernel_rows": kernel_rows,
        "outside_interval": outside,
        "phases": phases,
        "totals": {"gpu_total_ms": round(included_gpu_total_ms, 9)},
    }
    if steps is not None:
        result["steps"] = steps
    if wall_ms is not None:
        result["wall_ms"] = round(wall_ms, 9)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="nsys cuda_gpu_kern_sum CSV, or cuda_gpu_trace CSV when --gpu-trace is set")
    parser.add_argument("--steps", type=int, help="profiled decode steps")
    parser.add_argument("--wall-ms", type=float, help="measured wall time for those steps")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-trace", action="store_true", help="treat report as cuda_gpu_trace CSV and require explicit intervals")
    parser.add_argument("--decode-intervals", type=Path, help="explicit phase/decode intervals JSON for --gpu-trace mode")
    args = parser.parse_args()

    try:
        if args.gpu_trace:
            if args.decode_intervals is None:
                parser.error("--gpu-trace requires --decode-intervals")
            result = summarize_phase_trace(args.report, args.decode_intervals, steps=args.steps, wall_ms=args.wall_ms)
        else:
            if args.decode_intervals is not None:
                parser.error("--decode-intervals requires --gpu-trace")
            if args.steps is None or args.wall_ms is None:
                parser.error("legacy summary mode requires --steps and --wall-ms")
            result = summarize_report(args.report, steps=args.steps, wall_ms=args.wall_ms)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.gpu_trace:
        print(
            f"NSYS_PHASE_BUCKETS gpu_ms={result['totals']['gpu_total_ms']:.3f} "
            f"kernels={result['kernel_rows']} phases={','.join(result['phases'])}"
        )
    else:
        print(
            f"NSYS_BUCKETS gpu_ms={result['gpu_total_ms']:.3f} "
            f"wall_ms={result['wall_ms']:.3f} residual_ms={result['wall_residual_ms']:.3f} "
            f"overlap_ms={result['overlap_ms']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
