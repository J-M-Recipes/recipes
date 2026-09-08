import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VERDICT = REPO_ROOT / "recipes/dgx-station-gb300/glm-5.3-nvfp4-uva-slot-cache/scripts/window_verdict.py"


def run(*args):
    return subprocess.run(
        [sys.executable, str(VERDICT), *map(str, args)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def acceptance(path: Path, length: float):
    completion_tokens = int(round(length * 100))
    rows = [
        {
            "index": index,
            "kind": kind,
            "completion_tokens": completion_tokens,
            "metric_delta": {"drafts": 100, "draft_tokens": 100, "accepted_tokens": 90},
            "acceptance_length": length,
        }
        for index, kind in enumerate(("prose", "prose", "code", "code"))
    ]
    path.write_text(json.dumps({
        "schema": "glm53-dflash2-uva-acceptance-v1",
        "model": "glm-5.3-big",
        "max_tokens": 512,
        "rows": rows,
        "summary": {
            "requests": 4,
            "completion_tokens": completion_tokens * 4,
            "verification_steps": 400,
            "acceptance_length_weighted": length,
        },
    }))


def benches(directory: Path, c1_values):
    directory.mkdir()
    for index, c1 in enumerate(c1_values, 1):
        rows = [
            {"C": 1, "agg_tok_s": c1},
            {"C": 4, "agg_tok_s": c1 * 3},
            {"C": 8, "agg_tok_s": c1 * 5},
        ]
        (directory / f"bench-prose-rep{index}.txt").write_text(
            "\n".join("BENCH " + json.dumps(row) for row in rows) + "\n"
        )
    (directory / "bench-code-rep1.txt").write_text(
        "\n".join(
            "BENCH " + json.dumps({"C": c, "agg_tok_s": 10 * c}) for c in (1, 4, 8)
        ) + "\n"
    )


def test_acceptance_and_throughput_verdicts(tmp_path):
    baseline = tmp_path / "k1.json"
    candidate = tmp_path / "k2.json"
    acceptance(baseline, 3.0)
    acceptance(candidate, 3.15)
    out = tmp_path / "acceptance-verdict.json"
    result = run("acceptance", baseline, candidate, out)
    assert result.returncode == 0, result.stderr
    verdict = json.loads(out.read_text())
    assert verdict["pass"] is True
    assert verdict["gain"] == 0.15

    k1 = tmp_path / "k1"
    k2 = tmp_path / "k2"
    benches(k1, [10.0, 11.0, 9.0])
    benches(k2, [10.5, 11.55, 9.45])
    out = tmp_path / "throughput-verdict.json"
    result = run("throughput", k1, k2, out)
    assert result.returncode == 0, result.stderr
    verdict = json.loads(out.read_text())
    assert verdict["pass"] is True
    assert verdict["c1_ratio"] == 1.05
    assert verdict["baseline"]["prose"]["1"]["samples"] == 3


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("schema", "wrong", "acceptance receipt schema mismatch"),
        ("model", "wrong", "acceptance receipt model mismatch"),
        ("max_tokens", 64, "acceptance receipt max_tokens must equal 512"),
    ),
)
def test_acceptance_verdict_rejects_wrong_receipt_identity(tmp_path, field, value, message):
    baseline = tmp_path / "k1.json"
    candidate = tmp_path / "k2.json"
    acceptance(baseline, 3.0)
    acceptance(candidate, 3.15)
    payload = json.loads(candidate.read_text())
    payload[field] = value
    candidate.write_text(json.dumps(payload))

    result = run("acceptance", baseline, candidate, tmp_path / "verdict.json")

    assert result.returncode != 0
    assert message in result.stderr


def test_acceptance_verdict_rejects_missing_speculative_metric(tmp_path):
    baseline = tmp_path / "k1.json"
    candidate = tmp_path / "k2.json"
    acceptance(baseline, 3.0)
    acceptance(candidate, 3.15)
    payload = json.loads(candidate.read_text())
    del payload["rows"][0]["metric_delta"]["accepted_tokens"]
    candidate.write_text(json.dumps(payload))

    result = run("acceptance", baseline, candidate, tmp_path / "verdict.json")

    assert result.returncode != 0
    assert "row 0 has incomplete speculative metrics" in result.stderr


@pytest.mark.parametrize(("field", "value"), (("index", 3), ("kind", "code")))
def test_acceptance_verdict_rejects_wrong_row_provenance(tmp_path, field, value):
    baseline = tmp_path / "k1.json"
    candidate = tmp_path / "k2.json"
    acceptance(baseline, 3.0)
    acceptance(candidate, 3.15)
    payload = json.loads(candidate.read_text())
    payload["rows"][0][field] = value
    candidate.write_text(json.dumps(payload))

    result = run("acceptance", baseline, candidate, tmp_path / "verdict.json")

    assert result.returncode != 0
    assert "row 0 provenance mismatch" in result.stderr


def test_acceptance_verdict_rejects_mismatched_completion_summary(tmp_path):
    baseline = tmp_path / "k1.json"
    candidate = tmp_path / "k2.json"
    acceptance(baseline, 3.0)
    acceptance(candidate, 3.15)
    payload = json.loads(candidate.read_text())
    payload["summary"]["completion_tokens"] += 1
    candidate.write_text(json.dumps(payload))

    result = run("acceptance", baseline, candidate, tmp_path / "verdict.json")

    assert result.returncode != 0
    assert "summary completion tokens do not match rows" in result.stderr


def test_acceptance_verdict_rejects_tampered_summary(tmp_path):
    baseline = tmp_path / "k1.json"
    candidate = tmp_path / "k2.json"
    acceptance(baseline, 3.0)
    acceptance(candidate, 3.15)
    payload = json.loads(candidate.read_text())
    payload["summary"]["acceptance_length_weighted"] = 9.0
    candidate.write_text(json.dumps(payload))

    result = run("acceptance", baseline, candidate, tmp_path / "verdict.json")

    assert result.returncode != 0
    assert "weighted acceptance length does not match row counters" in result.stderr


def test_acceptance_verdict_rejects_mismatched_verification_steps(tmp_path):
    baseline = tmp_path / "k1.json"
    candidate = tmp_path / "k2.json"
    acceptance(baseline, 3.0)
    acceptance(candidate, 3.15)
    payload = json.loads(candidate.read_text())
    payload["summary"]["verification_steps"] = 399
    candidate.write_text(json.dumps(payload))

    result = run("acceptance", baseline, candidate, tmp_path / "verdict.json")

    assert result.returncode != 0
    assert "summary verification steps do not match row counters" in result.stderr


def test_quality_verdict_requires_complete_unique_valid_190_and_80_per_category(tmp_path):
    rows = []
    for kind in ("humaneval", "gsm8k"):
        for task in range(50):
            for repeat in range(2):
                rows.append({
                    "task_id": f"{kind}-{task}",
                    "task_type": kind,
                    "repeat": repeat,
                    "pass": not (task == 0 and repeat == 0),
                    "invalid": False,
                    "protocol_error": False,
                })
    evidence = tmp_path / "quality.jsonl"
    evidence.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    out = tmp_path / "quality-summary.json"
    result = run("quality", evidence, out)
    assert result.returncode == 0, result.stderr
    verdict = json.loads(out.read_text())
    assert verdict["pass"] is True
    assert verdict["rows"] == 200
    assert verdict["correct"] == 198
    assert verdict["categories"]["humaneval"] == {"correct": 99, "rows": 100}

    rows[-1]["invalid"] = True
    evidence.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    result = run("quality", evidence, out)
    assert result.returncode == 0, result.stderr
    assert json.loads(out.read_text())["pass"] is False
