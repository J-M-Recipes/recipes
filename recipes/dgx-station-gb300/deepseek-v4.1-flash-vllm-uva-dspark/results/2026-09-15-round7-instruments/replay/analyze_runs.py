#!/usr/bin/env python3
"""Compare paired independent replay run receipts.

This intentionally reports per-run wall numbers plus mean/range only. It does not
bootstrap requests and does not infer significance from small run counts.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
from typing import Any, Dict, List, Optional, Tuple

CHECK_KEYS = (
    "fixture_sha256",
    "fixture_session_count",
    "expected_turn_count",
    "turn_count",
    "model",
    "workers",
    "max_tokens",
    "reasoning_effort",
    "comparison_config_hash",
    "request_config_hash",
    "cache_salt_sha256",
    "cache_state",
)

REQUIRED_KEYS = CHECK_KEYS + ("tag", "wall_s", "completion_tokens", "aggregate_completion_tok_s")


class AnalyzeError(RuntimeError):
    pass


def parse_run_arg(value: str) -> Tuple[str, pathlib.Path]:
    if "=" not in value:
        raise AnalyzeError("--run must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label:
        raise AnalyzeError("--run label must be nonempty")
    if not path:
        raise AnalyzeError("--run path must be nonempty")
    return label, pathlib.Path(path)


def load_receipt(path: pathlib.Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AnalyzeError(f"could not read receipt {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise AnalyzeError(f"receipt {path} must be a JSON object")
    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        raise AnalyzeError(f"receipt {path} missing keys: {', '.join(missing)}")
    if data.get("turn_count") != data.get("expected_turn_count"):
        raise AnalyzeError(f"receipt {path} is not a complete run")
    return data


def check_labels(receipts: List[Dict[str, Any]], keys=CHECK_KEYS) -> Dict[str, Any]:
    if not receipts:
        raise AnalyzeError("no receipts provided")
    baseline = {key: receipts[0].get(key) for key in keys}
    for idx, receipt in enumerate(receipts[1:], start=1):
        for key, expected in baseline.items():
            if receipt.get(key) != expected:
                raise AnalyzeError(
                    f"receipt {idx} label mismatch for {key}: {receipt.get(key)!r} != {expected!r}"
                )
    return baseline


def mean_range(values: List[float]) -> Dict[str, Any]:
    if not values:
        raise AnalyzeError("cannot summarize empty value list")
    return {"mean": statistics.fmean(values), "range": [min(values), max(values)]}


def compare_runs(run_specs: List[Tuple[str, pathlib.Path]]) -> Dict[str, Any]:
    if len(run_specs) < 2:
        raise AnalyzeError("at least two --run receipts are required")
    loaded = []
    by_label: Dict[str, List[Dict[str, Any]]] = {}
    label_order: List[str] = []
    for label, path in run_specs:
        receipt = load_receipt(path)
        row = {"label": label, "path": str(path), "receipt": receipt}
        loaded.append(row)
        if label not in by_label:
            by_label[label] = []
            label_order.append(label)
        by_label[label].append(row)
    if len(label_order) < 2:
        raise AnalyzeError("paired comparison requires at least two distinct labels")
    label_counts = {label: len(rows) for label, rows in by_label.items()}
    if len(set(label_counts.values())) != 1:
        raise AnalyzeError(f"paired labels must have equal run counts: {label_counts}")
    # Exact namespace/request hashes must match WITHIN pairs. Across independent
    # boot pairs only the namespace changes; semantic settings remain identical.
    shared_keys = tuple(k for k in CHECK_KEYS if k not in ("request_config_hash", "cache_salt_sha256"))
    checked = check_labels([row["receipt"] for row in loaded], shared_keys)
    for i in range(next(iter(label_counts.values()))):
        check_labels([by_label[label][i]["receipt"] for label in label_order])

    per_run = []
    by_label_summary: Dict[str, Any] = {}
    for label in label_order:
        rows = by_label[label]
        walls = [float(row["receipt"]["wall_s"]) for row in rows]
        tok_s_values = [float(row["receipt"]["aggregate_completion_tok_s"]) for row in rows]
        by_label_summary[label] = {
            "n": len(rows),
            "wall_s": mean_range(walls),
            "aggregate_completion_tok_s": mean_range(tok_s_values),
        }
        for row in rows:
            receipt = row["receipt"]
            per_run.append(
                {
                    "label": label,
                    "path": row["path"],
                    "tag": receipt["tag"],
                    "cache_state": receipt["cache_state"],
                    "wall_s": float(receipt["wall_s"]),
                    "completion_tokens": int(receipt["completion_tokens"]),
                    "aggregate_completion_tok_s": float(receipt["aggregate_completion_tok_s"]),
                }
            )

    pairs = []
    pair_count = next(iter(label_counts.values()))
    for idx in range(pair_count):
        per_label = {}
        for label in label_order:
            receipt = by_label[label][idx]["receipt"]
            per_label[label] = {
                "tag": receipt["tag"],
                "wall_s": float(receipt["wall_s"]),
                "aggregate_completion_tok_s": float(receipt["aggregate_completion_tok_s"]),
            }
        pair = {"pair_index": idx, "per_label": per_label}
        if len(label_order) == 2:
            a, b = label_order
            pair["wall_delta_s"] = per_label[b]["wall_s"] - per_label[a]["wall_s"]
            pair["aggregate_completion_tok_s_delta"] = (
                per_label[b]["aggregate_completion_tok_s"] - per_label[a]["aggregate_completion_tok_s"]
            )
        pairs.append(pair)

    return {
        "statistics_note": "no significance test; mean/range only",
        "metadata_checked": checked,
        "labels": label_order,
        "per_run": per_run,
        "by_label": by_label_summary,
        "pairs": pairs,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="LABEL=summary.json; repeat for paired runs")
    parser.add_argument("--output", help="write JSON analysis to this path; stdout when omitted")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    try:
        args = parse_args(argv)
        analysis = compare_runs([parse_run_arg(value) for value in args.run])
        text = json.dumps(analysis, sort_keys=True, indent=2) + "\n"
        if args.output:
            pathlib.Path(args.output).write_text(text, encoding="utf-8")
        else:
            sys.stdout.write(text)
        return 0
    except AnalyzeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
