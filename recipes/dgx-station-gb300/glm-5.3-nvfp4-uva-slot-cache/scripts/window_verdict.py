#!/usr/bin/env python3
"""Compute frozen E0/E1/E5 acceptance, throughput, and quality verdicts."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

SCHEMA = "glm53-window-verdict-v1"


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _acceptance_value(path: Path) -> float:
    payload = json.loads(path.read_text())
    if payload.get("schema") != "glm53-dflash2-uva-acceptance-v1":
        raise ValueError(f"{path}: acceptance receipt schema mismatch")
    if payload.get("model") != "glm-5.3-big":
        raise ValueError(f"{path}: acceptance receipt model mismatch")
    if payload.get("max_tokens") != 512:
        raise ValueError(f"{path}: acceptance receipt max_tokens must equal 512")
    rows = payload.get("rows")
    summary = payload.get("summary")
    if not isinstance(rows, list) or len(rows) != 4 or not isinstance(summary, dict):
        raise ValueError(f"{path}: acceptance receipt must contain four rows and a summary")
    if summary.get("requests") != 4 or float(summary.get("verification_steps") or 0) <= 0:
        raise ValueError(f"{path}: acceptance receipt has incomplete requests or zero verification steps")
    completion_tokens = 0
    verification_steps = 0.0
    expected_kinds = ("prose", "prose", "code", "code")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or int(row.get("completion_tokens") or 0) <= 0:
            raise ValueError(f"{path}: row {index} has no completion tokens")
        if row.get("index") != index or row.get("kind") != expected_kinds[index]:
            raise ValueError(f"{path}: row {index} provenance mismatch")
        metrics = row.get("metric_delta")
        if not isinstance(metrics, dict) or any(
            not isinstance(metrics.get(name), (int, float)) or float(metrics[name]) <= 0
            for name in ("drafts", "draft_tokens", "accepted_tokens")
        ):
            raise ValueError(f"{path}: row {index} has incomplete speculative metrics")
        completion_tokens += int(row["completion_tokens"])
        verification_steps += float(metrics["drafts"])
    if summary.get("completion_tokens") != completion_tokens:
        raise ValueError(f"{path}: summary completion tokens do not match rows")
    if abs(float(summary["verification_steps"]) - verification_steps) > 1e-12:
        raise ValueError(f"{path}: summary verification steps do not match row counters")
    value = summary.get("acceptance_length_weighted")
    if not isinstance(value, (int, float)):
        raise ValueError(f"{path}: no weighted acceptance length")
    derived_value = completion_tokens / verification_steps
    if abs(float(value) - derived_value) > 1e-12:
        raise ValueError(f"{path}: weighted acceptance length does not match row counters")
    return derived_value


def acceptance_verdict(baseline_path: Path, candidate_path: Path) -> dict[str, Any]:
    baseline = _acceptance_value(baseline_path)
    candidate = _acceptance_value(candidate_path)
    gain = candidate - baseline
    return {
        "schema": SCHEMA,
        "gate": "acceptance",
        "baseline_weighted_acceptance_length": baseline,
        "candidate_weighted_acceptance_length": candidate,
        "required_gain": 0.15,
        "gain": round(gain, 9),
        "pass": gain >= 0.15 - 1e-12,
    }


def _bench_rows(path: Path) -> dict[int, float]:
    rows: dict[int, float] = {}
    for line in path.read_text().splitlines():
        if not line.startswith("BENCH "):
            continue
        row = json.loads(line[6:])
        concurrency = int(row["C"])
        if concurrency in rows:
            raise ValueError(f"{path}: duplicate C{concurrency} row")
        rows[concurrency] = float(row["agg_tok_s"])
    if set(rows) != {1, 4, 8}:
        raise ValueError(f"{path}: expected exactly C1/C4/C8 BENCH rows")
    return rows


def _bench_summary(directory: Path) -> dict[str, Any]:
    prose_paths = sorted(directory.glob("bench-prose-rep*.txt"))
    code_paths = sorted(directory.glob("bench-code-rep*.txt"))
    if len(prose_paths) != 3 or len(code_paths) != 1:
        raise ValueError(f"{directory}: expected three prose receipts and one code receipt")
    prose_rows = [_bench_rows(path) for path in prose_paths]
    code_rows = [_bench_rows(path) for path in code_paths]

    def summarize(rows: list[dict[int, float]]) -> dict[str, Any]:
        return {
            str(c): {
                "samples": len(rows),
                "values": [row[c] for row in rows],
                "mean_agg_tok_s": round(statistics.fmean(row[c] for row in rows), 9),
            }
            for c in (1, 4, 8)
        }

    return {"prose": summarize(prose_rows), "code": summarize(code_rows)}


def throughput_verdict(baseline_dir: Path, candidate_dir: Path) -> dict[str, Any]:
    baseline = _bench_summary(baseline_dir)
    candidate = _bench_summary(candidate_dir)
    baseline_c1 = baseline["prose"]["1"]["mean_agg_tok_s"]
    candidate_c1 = candidate["prose"]["1"]["mean_agg_tok_s"]
    if baseline_c1 <= 0:
        raise ValueError("baseline C1 mean must be positive")
    ratio = candidate_c1 / baseline_c1
    return {
        "schema": SCHEMA,
        "gate": "throughput",
        "baseline": baseline,
        "candidate": candidate,
        "required_c1_ratio": 1.05,
        "c1_ratio": round(ratio, 9),
        "pass": ratio >= 1.05 - 1e-12,
    }


def e1_verdict(directory: Path) -> dict[str, Any]:
    required = (
        "profiled-probe.json",
        "unprofiled-probe.json",
        "nsys-buckets.json",
        "container.log",
        "e1_cuda_gpu_kern_sum.csv",
        "e1_cuda_kern_exec_sum.csv",
        "e1_cuda_gpu_trace.csv",
        "e1_cuda_api_trace.csv",
        "profile-wall-seconds.txt",
    )
    missing = [name for name in required if not (directory / name).is_file()]
    receipt: dict[str, Any] = {
        "schema": "glm53-e1-verdict-v2",
        "gate": "e1",
        "verdict": "INCONCLUSIVE",
        "valid": False,
        "pass": False,
        "issues": [],
    }
    if missing:
        receipt["issues"].append("missing required profiler receipts: " + ", ".join(missing))
        return receipt
    profiled = json.loads((directory / "profiled-probe.json").read_text())
    unprofiled = json.loads((directory / "unprofiled-probe.json").read_text())
    buckets = json.loads((directory / "nsys-buckets.json").read_text())
    steps = int(profiled["summary"]["verification_steps"])
    profiled_speed = float(profiled["summary"]["decode_tok_s_median"])
    unprofiled_speed = float(unprofiled["summary"]["decode_tok_s_median"])
    slowdown = max(0.0, 1.0 - profiled_speed / unprofiled_speed) if unprofiled_speed > 0 else 1.0

    def probe_identity(payload: dict[str, Any]) -> tuple[Any, ...]:
        rows = payload.get("rows") or []
        row_identity = tuple(
            (row.get("index"), row.get("kind"), row.get("completion_tokens"), row.get("finish_reason"))
            for row in rows
            if isinstance(row, dict)
        )
        summary = payload.get("summary") or {}
        return (
            payload.get("schema"),
            payload.get("model"),
            payload.get("max_tokens"),
            summary.get("requests"),
            summary.get("completion_tokens"),
            row_identity,
        )

    if probe_identity(profiled) != probe_identity(unprofiled):
        receipt["issues"].append("profiled/unprofiled probe provenance mismatch")
    required_buckets = {"fused_bookkeeping", "masked_row_copy", "routed_moe", "dense_gemm", "scalar_gather", "mla_attention", "mtp_verify", "other_gpu"}
    present_buckets = set((buckets.get("buckets") or {}).keys())
    if buckets.get("source_report") not in ("e1_cuda_gpu_kern_sum.csv", str(directory / "e1_cuda_gpu_kern_sum.csv")):
        receipt["issues"].append("nsys bucket source report mismatch")
    log = (directory / "container.log").read_text(errors="replace")
    lowered_log = log.lower()
    def finite_number(value: Any, name: str) -> float | None:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            receipt["issues"].append(f"non-numeric {name}")
            return None
        number = float(value)
        if number != number or number in (float("inf"), float("-inf")):
            receipt["issues"].append(f"non-finite {name}")
            return None
        return number

    if steps < 50:
        receipt["issues"].append("fewer than 50 verification steps")
    if slowdown > 0.20:
        receipt["issues"].append("profiler slowdown exceeds 20%")
    if present_buckets != required_buckets:
        receipt["issues"].append("kernel bucket set mismatch")
    for bucket_name in sorted(present_buckets & required_buckets):
        finite_number((buckets.get("buckets") or {}).get(bucket_name, {}).get("per_step_ms"), f"{bucket_name}.per_step_ms")
    engine_evidence = directory / "engine-core-graph-evidence.json"
    if not engine_evidence.is_file():
        receipt["issues"].append("missing engine-core/graph-node evidence")
    else:
        try:
            engine = json.loads(engine_evidence.read_text())
            if not (engine.get("schema") == "glm53-e1-engine-core-graph-evidence-v1" and engine.get("engine_core_capture") is True and engine.get("graph_node_capture") is True):
                receipt["issues"].append("invalid engine-core/graph-node evidence")
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            receipt["issues"].append("invalid engine-core/graph-node evidence")
    for expected_ack in ("profile_control ack start e1-profile-v2", "profile_control ack stop e1-profile-v2"):
        if expected_ack not in lowered_log:
            receipt["issues"].append("missing container profile control ACK")
            break
    for needle in ("profile control error", "profiler error", "cuda error", "nsys error", "report-generation error"):
        if needle in lowered_log:
            receipt["issues"].append("profiler/report error present: " + needle)
            break
    # Fail closed on API attribution: v2 accepts an explicit hash-bound attribution receipt.
    api_attr = directory / "api-attribution.json"
    source_sha256 = {
        name: __import__("hashlib").sha256((directory / name).read_bytes()).hexdigest()
        for name in (
            "e1_cuda_gpu_kern_sum.csv",
            "e1_cuda_kern_exec_sum.csv",
            "e1_cuda_gpu_trace.csv",
            "e1_cuda_api_trace.csv",
            "profiled-probe.json",
            "profile-wall-seconds.txt",
            "nsys-buckets.json",
        )
    }
    api_ms = None
    if api_attr.is_file():
        attribution = json.loads(api_attr.read_text())
        value = attribution.get("attributable_cuda_api_ms_per_step")
        if (
            attribution.get("schema") == "glm53-e1-api-attribution-v1"
            and attribution.get("complete") is True
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value >= 0
            and attribution.get("source_sha256") == source_sha256
        ):
            api_ms = float(value)
        else:
            receipt["issues"].append("invalid hash-bound CUDA API attribution")
    else:
        receipt["issues"].append("missing hash-bound CUDA API attribution")
    gpu_ms = 0.0
    bucket_payload = buckets.get("buckets") or {}
    for bucket_name in ("fused_bookkeeping", "scalar_gather"):
        bucket_value = bucket_payload.get(bucket_name, {}).get("per_step_ms")
        if isinstance(bucket_value, (int, float)) and not isinstance(bucket_value, bool):
            gpu_ms += float(bucket_value)
    recoverable = None if receipt["issues"] else gpu_ms + float(api_ms)
    receipt.update({
        "verification_steps": steps,
        "profile_slowdown_fraction": round(slowdown, 9),
        "gpu_recoverable_ms_per_step": round(gpu_ms, 9),
        "attributable_cuda_api_ms_per_step": api_ms,
        "recoverable_ms_per_step": recoverable,
        "source_sha256": source_sha256,
    })
    if recoverable is not None:
        receipt["valid"] = True
        receipt["pass"] = recoverable >= 2.0
        receipt["verdict"] = "PASS" if receipt["pass"] else "STOP"
    return receipt


def quality_verdict(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    identities = [(row.get("task_id"), row.get("repeat")) for row in rows]
    unique = len(set(identities)) == len(identities)
    valid = all(
        not row.get("invalid")
        and not row.get("protocol_error")
        and isinstance(row.get("pass"), bool)
        for row in rows
    )
    categories = {
        name: {
            "rows": sum(row.get("task_type") == name for row in rows),
            "correct": sum(row.get("task_type") == name and row.get("pass") is True for row in rows),
        }
        for name in ("humaneval", "gsm8k")
    }
    correct = sum(row.get("pass") is True for row in rows)
    passed = (
        len(rows) == 200
        and unique
        and valid
        and correct >= 190
        and all(category["rows"] == 100 and category["correct"] >= 80 for category in categories.values())
    )
    return {
        "schema": SCHEMA,
        "gate": "quality",
        "rows": len(rows),
        "unique_task_repeats": unique,
        "all_rows_valid": valid,
        "correct": correct,
        "categories": categories,
        "pass": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    acceptance = sub.add_parser("acceptance")
    acceptance.add_argument("baseline", type=Path)
    acceptance.add_argument("candidate", type=Path)
    acceptance.add_argument("output", type=Path)

    throughput = sub.add_parser("throughput")
    throughput.add_argument("baseline_dir", type=Path)
    throughput.add_argument("candidate_dir", type=Path)
    throughput.add_argument("output", type=Path)

    e1 = sub.add_parser("e1")
    e1.add_argument("directory", type=Path)
    e1.add_argument("output", type=Path)

    quality = sub.add_parser("quality")
    quality.add_argument("evidence", type=Path)
    quality.add_argument("output", type=Path)

    args = parser.parse_args()
    try:
        if args.command == "acceptance":
            verdict = acceptance_verdict(args.baseline, args.candidate)
        elif args.command == "throughput":
            verdict = throughput_verdict(args.baseline_dir, args.candidate_dir)
        elif args.command == "e1":
            verdict = e1_verdict(args.directory)
        else:
            verdict = quality_verdict(args.evidence)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    _write(args.output, verdict)
    print(f"WINDOW_VERDICT {args.command} pass={str(verdict['pass']).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
