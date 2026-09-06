"""Synthetic decode-audit tests.

All fixtures in this file are hand-written synthetic JSON. They are contract tests
for the local postprocessor only and are not empirical model evidence.
"""
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "recipes" / "dgx-station-gb300" / "glm-5.3-nvfp4-uva-slot-cache" / "scripts" / "audit_decode.py"

spec = importlib.util.spec_from_file_location("audit_decode", SCRIPT)
audit_decode = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_decode)


def piece(i, cat="prose", tokens=None, lps=None, top1=None, finish="length"):
    tokens = ["A", "B", "C"] if tokens is None else tokens
    lps = [-0.1, -0.2, -0.3] if lps is None else lps
    top1 = list(tokens) if top1 is None else top1
    return {"i": i, "cat": cat, "n": len(tokens), "tokens": tokens, "lps": lps, "top1": top1, "finish": finish}


def artifact(label="run", pieces=None, prefix=4, gen=3, metadata=None):
    return {
        "label": label,
        "prefix": prefix,
        "gen": gen,
        "metadata": ({"corpus": "synthetic-fixture", "model": "synthetic-model"} if metadata is None else metadata),
        "pieces": [piece(0, "prose"), piece(1, "code")] if pieces is None else pieces,
    }


def write_json(tmp_path, name, data):
    p = tmp_path / name
    p.write_text(json.dumps(data, allow_nan=True))
    return p


def run_audit(tmp_path, ref_a, ref_b, *candidates):
    paths = [write_json(tmp_path, "ref_a.json", ref_a), write_json(tmp_path, "ref_b.json", ref_b)]
    cand_paths = [write_json(tmp_path, f"cand{idx}.json", c) for idx, c in enumerate(candidates)]
    return audit_decode.audit(paths[0], paths[1], cand_paths)


def test_aligned_identical_synthetic_fixture_is_diagnostic_only_with_zero_metrics(tmp_path):
    ref_a = artifact("ref-a", metadata={})
    ref_b = artifact("ref-b", metadata={})
    cand = artifact("candidate", metadata={})

    out = run_audit(tmp_path, ref_a, ref_b, cand)

    assert out["status"] == "diagnostic_only"
    assert out["validation"]["errors"] == []
    assert out["provenance"]["verified"] is False
    assert any("missing metadata" in note for note in out["provenance"]["notes"])
    floor = out["observed_floor"]
    assert floor["compared_tokens"] == floor["eligible_tokens"] == 6
    assert floor["mean_abs_dlp"] == 0.0
    assert floor["coverage"] == 1.0
    worst = out["candidates"][0]["worst_vs_references"]
    assert worst["mean_abs_dlp"] == 0.0
    assert worst["coverage"] == 1.0
    assert out["candidates"][0]["regression_flags"] == []


def test_early_divergence_at_zero_censors_equal_prefix_logprob_metrics(tmp_path):
    ref = artifact("ref", pieces=[piece(0, tokens=["A", "B"], lps=[-1.0, -2.0], top1=["A", "B"] )])
    cand = artifact("candidate", pieces=[piece(0, tokens=["X", "Y"], lps=[-9.0, -9.0], top1=["A", "Y"] )])

    out = run_audit(tmp_path, ref, ref, cand)

    comp = out["candidates"][0]["comparisons"][0]
    assert comp["eligible_tokens"] == 2
    assert comp["compared_tokens"] == 0
    assert comp["coverage"] == 0.0
    assert comp["mean_abs_dlp"] is None
    assert comp["divergence"]["at0"] == 1
    assert comp["divergence"]["le1"] == 1
    assert comp["divergence"]["le8"] == 1
    assert comp["divergence"]["le16"] == 1
    assert comp["first_diff_top1_agreement"] == {"eligible": 1, "agree": 1, "rate": 1.0}


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda a: a.update({"pieces": a["pieces"][:1]}), "piece count"),
        (lambda a: a["pieces"].append(piece(2, "extra")), "piece count"),
        (lambda a: a.update({"pieces": list(reversed(a["pieces"]))}), "piece id"),
        (lambda a: a["pieces"].__setitem__(1, piece(0, "code")), "duplicate piece id"),
        (lambda a: a.update({"prefix": 99}), "prefix"),
        (lambda a: a.update({"gen": 99}), "gen"),
        (lambda a: a["pieces"].__setitem__(1, piece(1, "wrong-cat")), "category"),
    ],
)
def test_misaligned_piece_counts_ids_categories_prefix_or_gen_are_invalid(tmp_path, mutate, expected):
    ref_a = artifact("ref-a")
    ref_b = artifact("ref-b")
    cand = artifact("candidate")
    mutate(cand)

    out = run_audit(tmp_path, ref_a, ref_b, cand)

    assert out["status"] == "invalid"
    assert any(expected in err for err in out["validation"]["errors"]), out["validation"]["errors"]


@pytest.mark.parametrize(
    "bad_piece, expected",
    [
        (piece(0, lps=[-0.1, None, -0.3]), "finite numeric non-null logprob"),
        (piece(0, lps=[-0.1, math.nan, -0.3]), "finite numeric non-null logprob"),
        ({"i": 0, "cat": "prose", "n": 3, "tokens": ["A"], "lps": [-0.1], "top1": ["A"], "finish": "length"}, "array length"),
        (piece(0, top1=["A", "", "C"]), "nonempty top1"),
        (piece(0, finish="surprise"), "illegal finish reason"),
    ],
)
def test_null_nan_malformed_arrays_and_illegal_finish_are_invalid(tmp_path, bad_piece, expected):
    ref_a = artifact("ref-a")
    ref_b = artifact("ref-b")
    cand = artifact("candidate", pieces=[bad_piece, piece(1, "code")])

    out = run_audit(tmp_path, ref_a, ref_b, cand)

    assert out["status"] == "invalid"
    assert any(expected in err for err in out["validation"]["errors"]), out["validation"]["errors"]


def test_different_generated_lengths_and_finish_mismatches_are_reported(tmp_path):
    ref = artifact("ref", pieces=[piece(0, tokens=["A", "B", "C"], lps=[-1, -2, -3], finish="length")])
    cand = artifact("cand", pieces=[piece(0, tokens=["A", "B"], lps=[-1, -2], finish="stop")])

    out = run_audit(tmp_path, ref, ref, cand)

    comp = out["candidates"][0]["comparisons"][0]
    assert comp["divergence"]["length_end_mismatches"] == 1
    assert comp["divergence"]["unequal_length_pieces"] == 1
    assert comp["divergence"]["finish_mismatches"] == 1
    assert comp["examples"]["length_end_mismatches"][0]["piece_id"] == 0
    assert comp["examples"]["finish_mismatches"][0]["candidate_finish"] == "stop"


def test_category_regression_is_flagged_even_when_global_mean_is_masked(tmp_path):
    # Floor: prose has high natural drift, code has none.
    ref_a = artifact("ref-a", pieces=[
        piece(0, "prose", tokens=["A", "B"], lps=[0.0, 0.0]),
        piece(1, "code", tokens=["C", "D"], lps=[0.0, 0.0]),
    ])
    ref_b = artifact("ref-b", pieces=[
        piece(0, "prose", tokens=["A", "B"], lps=[-10.0, -10.0]),
        piece(1, "code", tokens=["C", "D"], lps=[0.0, 0.0]),
    ])
    cand = artifact("cand", pieces=[
        piece(0, "prose", tokens=["A", "B"], lps=[-5.0, -5.0]),
        piece(1, "code", tokens=["C", "D"], lps=[-1.0, -1.0]),
    ])

    out = run_audit(tmp_path, ref_a, ref_b, cand)

    flags = out["candidates"][0]["regression_flags"]
    assert not any(f["scope"] == "global" and f["metric"] == "mean_abs_dlp" for f in flags), flags
    assert any(f["scope"] == "category" and f["category"] == "code" and f["metric"] == "mean_abs_dlp" for f in flags), flags


def test_cli_missing_files_emit_json_and_nonzero_exit(tmp_path):
    missing = tmp_path / "missing.json"
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--reference-a", str(missing),
        "--reference-b", str(missing),
        "--candidate", str(missing),
    ]

    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    assert proc.returncode != 0
    out = json.loads(proc.stdout)
    assert out["status"] == "invalid"
    assert any("missing artifact" in err for err in out["validation"]["errors"])
