#!/usr/bin/env python3
"""Local decode-audit postprocessor for tf_decode-shaped artifacts.

This script does not run decoding. It validates captured JSON artifacts shaped as
{label,prefix,gen,pieces:[{i,cat,n,tokens,lps,top1,finish}]} and emits a
JSON diagnostic comparing candidate decode histories to the observed two-run
reference floor. The output is diagnostic-only, not a statistical acceptance
or non-inferiority claim.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "decode-audit/v1"
LEGAL_FINISH_REASONS = {
    "length",
    "stop",
    "content_filter",
    "tool_calls",
    "function_call",
    "eos",
    "eos_token",
    "end",
    "end_turn",
}


def _role_name(role: str) -> str:
    return role.replace("_", "-")


def _label(artifact: dict[str, Any], role: str) -> str:
    value = artifact.get("label")
    return value if isinstance(value, str) and value else role


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _provenance_value(artifact: dict[str, Any], key: str) -> Any:
    meta = artifact.get("metadata")
    if isinstance(meta, dict) and meta.get(key):
        return meta.get(key)
    return artifact.get(key)


def _empty_report() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "diagnostic_only",
        "validation": {"errors": [], "warnings": []},
        "provenance": {
            "verified": False,
            "notes": ["local audit does not independently verify corpus/model provenance"],
        },
        "artifacts": [],
        "observed_floor": None,
        "candidates": [],
    }


def _load_artifact(path: Path, role: str, errors: list[str]) -> dict[str, Any] | None:
    if not path.exists():
        errors.append(f"{_role_name(role)} missing artifact: {path}")
        return None
    if not path.is_file():
        errors.append(f"{_role_name(role)} artifact is not a file: {path}")
        return None
    try:
        with path.open() as fh:
            data = json.load(fh)
    except Exception as exc:  # pragma: no cover - exact JSON errors vary by Python
        errors.append(f"{_role_name(role)} cannot read JSON artifact {path}: {exc}")
        return None
    if not isinstance(data, dict):
        errors.append(f"{_role_name(role)} artifact root must be an object: {path}")
        return None
    return data


def _validate_artifact(role: str, artifact: dict[str, Any], errors: list[str], warnings: list[str]) -> None:
    label = _label(artifact, role)
    if not isinstance(artifact.get("label"), str) or not artifact.get("label"):
        errors.append(f"{_role_name(role)} label must be a nonempty string")
    for key in ("prefix", "gen"):
        if not isinstance(artifact.get(key), int) or isinstance(artifact.get(key), bool) or artifact.get(key) < 0:
            errors.append(f"{_role_name(role)} {key} must be a nonnegative integer")

    pieces = artifact.get("pieces")
    if not isinstance(pieces, list) or not pieces:
        errors.append(f"{_role_name(role)} pieces must be a nonempty list")
        return

    seen_ids: set[Any] = set()
    for pos, piece in enumerate(pieces):
        where = f"{_role_name(role)} piece[{pos}]"
        if not isinstance(piece, dict):
            errors.append(f"{where} must be an object")
            continue
        piece_id = piece.get("i")
        try:
            duplicate_key = piece_id if hash(piece_id) is not None else repr(piece_id)
        except TypeError:
            duplicate_key = repr(piece_id)
        if duplicate_key in seen_ids:
            errors.append(f"{where} duplicate piece id {piece_id!r}")
        seen_ids.add(duplicate_key)
        if not isinstance(piece_id, int) or isinstance(piece_id, bool):
            errors.append(f"{where} piece id i must be an integer")
        if not isinstance(piece.get("cat"), str) or not piece.get("cat"):
            errors.append(f"{where} category must be a nonempty string")
        n = piece.get("n")
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            errors.append(f"{where} n must be a nonnegative integer")
            n = None
        arrays: dict[str, Any] = {name: piece.get(name) for name in ("tokens", "lps", "top1")}
        for name, value in arrays.items():
            if not isinstance(value, list):
                errors.append(f"{where} {name} must be an array")
        if n is not None:
            for name, value in arrays.items():
                if isinstance(value, list) and len(value) != n:
                    errors.append(f"{where} array length mismatch for {name}: len={len(value)} n={n}")
        lps = arrays.get("lps")
        if isinstance(lps, list):
            for j, lp in enumerate(lps):
                if not _is_number(lp):
                    errors.append(f"{where} lps[{j}] must be a finite numeric non-null logprob")
        top1 = arrays.get("top1")
        if isinstance(top1, list):
            for j, tok in enumerate(top1):
                if tok is None or (isinstance(tok, str) and tok == ""):
                    errors.append(f"{where} top1[{j}] must be a nonempty top1 token")
        finish = piece.get("finish")
        if finish not in LEGAL_FINISH_REASONS:
            errors.append(f"{where} illegal finish reason {finish!r}")

    missing_meta = [key for key in ("corpus", "model") if not _provenance_value(artifact, key)]
    if missing_meta:
        warnings.append(f"{_role_name(role)} {label}: missing metadata {', '.join(missing_meta)}; provenance unverified")


def _validate_alignment(base_role: str, base: dict[str, Any], role: str, artifact: dict[str, Any], errors: list[str]) -> None:
    if base.get("prefix") != artifact.get("prefix"):
        errors.append(f"{_role_name(role)} prefix {artifact.get('prefix')!r} does not match {_role_name(base_role)} {base.get('prefix')!r}")
    if base.get("gen") != artifact.get("gen"):
        errors.append(f"{_role_name(role)} gen {artifact.get('gen')!r} does not match {_role_name(base_role)} {base.get('gen')!r}")
    base_pieces = base.get("pieces") if isinstance(base.get("pieces"), list) else []
    pieces = artifact.get("pieces") if isinstance(artifact.get("pieces"), list) else []
    if len(base_pieces) != len(pieces):
        errors.append(f"{_role_name(role)} piece count {len(pieces)} does not match {_role_name(base_role)} piece count {len(base_pieces)}")
    for pos, (left, right) in enumerate(zip(base_pieces, pieces)):
        if not isinstance(left, dict) or not isinstance(right, dict):
            continue
        if left.get("i") != right.get("i"):
            errors.append(f"{_role_name(role)} piece id at position {pos} {right.get('i')!r} does not match {_role_name(base_role)} {left.get('i')!r}")
        if left.get("cat") != right.get("cat"):
            errors.append(f"{_role_name(role)} category at position {pos} {right.get('cat')!r} does not match {_role_name(base_role)} {left.get('cat')!r}")


def _new_bucket() -> dict[str, Any]:
    return {
        "pieces": 0,
        "eligible_tokens": 0,
        "compared_tokens": 0,
        "dlps": [],
        "same_history_top1_disagree": 0,
        "divergent_pieces": 0,
        "at0": 0,
        "le1": 0,
        "le8": 0,
        "le16": 0,
        "length_end_mismatches": 0,
        "unequal_length_pieces": 0,
        "finish_mismatches": 0,
        "first_diff_top1_eligible": 0,
        "first_diff_top1_agree": 0,
    }


def _add_piece_stats(bucket: dict[str, Any], x: dict[str, Any], y: dict[str, Any]) -> tuple[str | None, int | None]:
    bucket["pieces"] += 1
    nx = int(x["n"])
    ny = int(y["n"])
    limit = min(nx, ny)
    bucket["eligible_tokens"] += limit

    tokens_x = x["tokens"]
    tokens_y = y["tokens"]
    k = 0
    while k < limit and tokens_x[k] == tokens_y[k]:
        k += 1

    for j in range(k):
        dlp = abs(float(x["lps"][j]) - float(y["lps"][j]))
        bucket["dlps"].append(dlp)
        bucket["compared_tokens"] += 1
        if x["top1"][j] != y["top1"][j]:
            bucket["same_history_top1_disagree"] += 1

    div_kind: str | None = None
    div_pos: int | None = None
    if k < limit:
        div_kind = "token"
        div_pos = k
        bucket["first_diff_top1_eligible"] += 1
        if x["top1"][k] == y["top1"][k]:
            bucket["first_diff_top1_agree"] += 1
    elif nx != ny:
        div_kind = "length_end"
        div_pos = limit
        bucket["length_end_mismatches"] += 1

    if nx != ny:
        bucket["unequal_length_pieces"] += 1
    if x.get("finish") != y.get("finish"):
        bucket["finish_mismatches"] += 1

    if div_kind is not None:
        bucket["divergent_pieces"] += 1
        if div_pos == 0:
            bucket["at0"] += 1
        if div_pos is not None and div_pos <= 1:
            bucket["le1"] += 1
        if div_pos is not None and div_pos <= 8:
            bucket["le8"] += 1
        if div_pos is not None and div_pos <= 16:
            bucket["le16"] += 1
    return div_kind, div_pos


def _rate(numer: int, denom: int) -> float | None:
    return None if denom == 0 else numer / denom


def _summarize_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    dlps = sorted(bucket["dlps"])
    pieces = bucket["pieces"]
    compared = bucket["compared_tokens"]
    eligible = bucket["eligible_tokens"]
    mean = None if compared == 0 else sum(dlps) / compared
    p99 = None if compared == 0 else dlps[min(len(dlps) - 1, int(0.99 * len(dlps)))]
    max_dlp = None if compared == 0 else dlps[-1]
    top1_disagree = bucket["same_history_top1_disagree"]
    first_eligible = bucket["first_diff_top1_eligible"]
    first_agree = bucket["first_diff_top1_agree"]
    return {
        "metric_scope": "equal_generated_prefix_only_censored_after_first_difference",
        "pieces": pieces,
        "eligible_tokens": eligible,
        "compared_tokens": compared,
        "coverage": _rate(compared, eligible),
        "mean_abs_dlp": mean,
        "p99_abs_dlp": p99,
        "max_abs_dlp": max_dlp,
        "same_history_top1_disagree": {
            "eligible": compared,
            "disagree": top1_disagree,
            "rate": _rate(top1_disagree, compared),
        },
        "first_diff_top1_agreement": {
            "eligible": first_eligible,
            "agree": first_agree,
            "rate": _rate(first_agree, first_eligible),
        },
        "divergence": {
            "pieces": bucket["divergent_pieces"],
            "rate": _rate(bucket["divergent_pieces"], pieces),
            "at0": bucket["at0"],
            "at0_rate": _rate(bucket["at0"], pieces),
            "le1": bucket["le1"],
            "le1_rate": _rate(bucket["le1"], pieces),
            "le8": bucket["le8"],
            "le8_rate": _rate(bucket["le8"], pieces),
            "le16": bucket["le16"],
            "le16_rate": _rate(bucket["le16"], pieces),
            "length_end_mismatches": bucket["length_end_mismatches"],
            "length_end_mismatch_rate": _rate(bucket["length_end_mismatches"], pieces),
            "unequal_length_pieces": bucket["unequal_length_pieces"],
            "unequal_length_rate": _rate(bucket["unequal_length_pieces"], pieces),
            "finish_mismatches": bucket["finish_mismatches"],
            "finish_mismatch_rate": _rate(bucket["finish_mismatches"], pieces),
        },
    }


def compare_artifacts(a: dict[str, Any], b_ref: dict[str, Any]) -> dict[str, Any]:
    global_bucket = _new_bucket()
    per_cat: dict[str, dict[str, Any]] = {}
    examples: dict[str, list[dict[str, Any]]] = {
        "token_divergences": [],
        "length_end_mismatches": [],
        "finish_mismatches": [],
    }

    for x, y in zip(a["pieces"], b_ref["pieces"]):
        cat = x["cat"]
        cat_bucket = per_cat.setdefault(cat, _new_bucket())
        kind, pos = _add_piece_stats(global_bucket, x, y)
        _add_piece_stats(cat_bucket, x, y)
        piece_id = x["i"]
        if kind == "token" and len(examples["token_divergences"]) < 5:
            examples["token_divergences"].append({
                "piece_id": piece_id,
                "cat": cat,
                "pos": pos,
                "candidate_token": x["tokens"][pos],
                "reference_token": y["tokens"][pos],
                "candidate_top1": x["top1"][pos],
                "reference_top1": y["top1"][pos],
            })
        if kind == "length_end" and len(examples["length_end_mismatches"]) < 5:
            examples["length_end_mismatches"].append({
                "piece_id": piece_id,
                "cat": cat,
                "common_tokens": pos,
                "candidate_n": x["n"],
                "reference_n": y["n"],
            })
        if x.get("finish") != y.get("finish") and len(examples["finish_mismatches"]) < 5:
            examples["finish_mismatches"].append({
                "piece_id": piece_id,
                "cat": cat,
                "candidate_finish": x.get("finish"),
                "reference_finish": y.get("finish"),
            })

    out = _summarize_bucket(global_bucket)
    out.update({
        "A": _label(a, "A"),
        "B_ref": _label(b_ref, "B_ref"),
        "per_cat": {cat: _summarize_bucket(bucket) for cat, bucket in sorted(per_cat.items())},
        "examples": examples,
    })
    return out


def _metric_value(summary: dict[str, Any], metric: str) -> float | None:
    if metric == "mean_abs_dlp":
        return summary.get("mean_abs_dlp")
    if metric == "coverage":
        return summary.get("coverage")
    if metric in {"rate", "at0_rate", "le1_rate", "le8_rate", "le16_rate", "length_end_mismatch_rate"}:
        return summary.get("divergence", {}).get(metric)
    raise KeyError(metric)


def _worse_summary(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    metrics_max = ["mean_abs_dlp", "rate", "at0_rate", "le1_rate", "le8_rate", "le16_rate", "length_end_mismatch_rate"]
    out: dict[str, Any] = {"coverage": None}
    coverages = [c.get("coverage") for c in comparisons if c.get("coverage") is not None]
    out["coverage"] = min(coverages) if coverages else None
    for metric in metrics_max:
        values = [_metric_value(c, metric) for c in comparisons]
        numeric = [v for v in values if v is not None]
        out[metric] = max(numeric) if numeric else None
    cats = sorted({cat for comp in comparisons for cat in comp.get("per_cat", {})})
    out["per_cat"] = {}
    for cat in cats:
        cat_comps = [comp["per_cat"][cat] for comp in comparisons if cat in comp.get("per_cat", {})]
        cat_worst: dict[str, Any] = {}
        coverages = [c.get("coverage") for c in cat_comps if c.get("coverage") is not None]
        cat_worst["coverage"] = min(coverages) if coverages else None
        for metric in metrics_max:
            values = [_metric_value(c, metric) for c in cat_comps]
            numeric = [v for v in values if v is not None]
            cat_worst[metric] = max(numeric) if numeric else None
        out["per_cat"][cat] = cat_worst
    return out


def _add_flag(flags: list[dict[str, Any]], scope: str, metric: str, candidate: float, floor: float, category: str | None = None) -> None:
    flag: dict[str, Any] = {"scope": scope, "metric": metric, "candidate": candidate, "floor": floor}
    if category is not None:
        flag["category"] = category
    flags.append(flag)


def regression_flags(floor: dict[str, Any], candidate_worst: dict[str, Any]) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    metrics = ["mean_abs_dlp", "coverage", "rate", "at0_rate", "le1_rate", "le8_rate", "le16_rate", "length_end_mismatch_rate"]
    for metric in metrics:
        cand = candidate_worst.get(metric)
        base = _metric_value(floor, metric)
        if cand is None or base is None:
            continue
        if metric == "coverage":
            if cand < base:
                _add_flag(flags, "global", metric, cand, base)
        elif cand > base:
            _add_flag(flags, "global", metric, cand, base)

    for cat, base_summary in floor.get("per_cat", {}).items():
        cand_summary = candidate_worst.get("per_cat", {}).get(cat)
        if not cand_summary:
            continue
        for metric in metrics:
            cand = cand_summary.get(metric)
            base = _metric_value(base_summary, metric)
            if cand is None or base is None:
                continue
            if metric == "coverage":
                if cand < base:
                    _add_flag(flags, "category", metric, cand, base, cat)
            elif cand > base:
                _add_flag(flags, "category", metric, cand, base, cat)
    return flags


def audit(reference_a: Path | str, reference_b: Path | str, candidates: Iterable[Path | str]) -> dict[str, Any]:
    report = _empty_report()
    errors = report["validation"]["errors"]
    warnings = report["validation"]["warnings"]
    specs: list[tuple[str, Path]] = [
        ("reference_a", Path(reference_a)),
        ("reference_b", Path(reference_b)),
    ]
    specs.extend((f"candidate_{idx}", Path(path)) for idx, path in enumerate(candidates))

    loaded: dict[str, dict[str, Any]] = {}
    for role, path in specs:
        data = _load_artifact(path, role, errors)
        if data is None:
            continue
        loaded[role] = data
        _validate_artifact(role, data, errors, warnings)
        report["artifacts"].append({"role": role, "path": str(path), "label": _label(data, role)})

    for note in warnings:
        if "missing metadata" in note:
            report["provenance"]["notes"].append(note)

    base = loaded.get("reference_a")
    if base is not None:
        for role, data in loaded.items():
            if role != "reference_a":
                _validate_alignment("reference_a", base, role, data, errors)

    if errors:
        report["status"] = "invalid"
        return report

    ref_a = loaded["reference_a"]
    ref_b = loaded["reference_b"]
    floor = compare_artifacts(ref_a, ref_b)
    floor["role"] = "observed_floor"
    report["observed_floor"] = floor

    for role in sorted(r for r in loaded if r.startswith("candidate_")):
        candidate = loaded[role]
        comps = [compare_artifacts(candidate, ref_a), compare_artifacts(candidate, ref_b)]
        worst = _worse_summary(comps)
        report["candidates"].append({
            "role": role,
            "label": _label(candidate, role),
            "comparisons": comps,
            "worst_vs_references": worst,
            "regression_flags": regression_flags(floor, worst),
        })
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit tf_decode-shaped JSON artifacts locally.")
    parser.add_argument("--reference-a", required=True, help="first reference decode JSON")
    parser.add_argument("--reference-b", required=True, help="second reference decode JSON")
    parser.add_argument("--candidate", action="append", required=True, help="candidate decode JSON; repeatable")
    args = parser.parse_args(argv)

    out = audit(args.reference_a, args.reference_b, args.candidate)
    json.dump(out, sys.stdout, indent=2, sort_keys=True, allow_nan=False)
    sys.stdout.write("\n")
    return 0 if out["status"] != "invalid" else 2


if __name__ == "__main__":
    raise SystemExit(main())
