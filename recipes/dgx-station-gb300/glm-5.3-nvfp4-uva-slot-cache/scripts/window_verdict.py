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

    quality = sub.add_parser("quality")
    quality.add_argument("evidence", type=Path)
    quality.add_argument("output", type=Path)

    args = parser.parse_args()
    try:
        if args.command == "acceptance":
            verdict = acceptance_verdict(args.baseline, args.candidate)
        elif args.command == "throughput":
            verdict = throughput_verdict(args.baseline_dir, args.candidate_dir)
        else:
            verdict = quality_verdict(args.evidence)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    _write(args.output, verdict)
    print(f"WINDOW_VERDICT {args.command} pass={str(verdict['pass']).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
