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


def test_e1_verdict_remains_inconclusive_without_engine_core_graph_evidence(tmp_path):
    probe = {
        "schema": "glm53-dflash2-uva-acceptance-v1",
        "model": "glm-5.3-big",
        "max_tokens": 64,
        "summary": {"verification_steps": 128, "decode_tok_s_median": 100.0},
    }
    (tmp_path / "profiled-probe.json").write_text(json.dumps(probe))
    (tmp_path / "unprofiled-probe.json").write_text(json.dumps(probe))
    buckets = {name: {"per_step_ms": 0.0} for name in ("fused_bookkeeping", "masked_row_copy", "routed_moe", "scalar_gather", "mla_attention", "mtp_verify", "other_gpu")}
    buckets["fused_bookkeeping"]["per_step_ms"] = 1.5
    buckets["scalar_gather"]["per_step_ms"] = 0.6
    (tmp_path / "nsys-buckets.json").write_text(json.dumps({"buckets": buckets}))
    (tmp_path / "container.log").write_text("PROFILE_CONTROL ACK START e1-profile-v2\nPROFILE_CONTROL ACK STOP e1-profile-v2\nclean engine log")
    (tmp_path / "profile-wall-seconds.txt").write_text("1.0\n")
    for name in ("e1_cuda_gpu_kern_sum.csv", "e1_cuda_kern_exec_sum.csv", "e1_cuda_gpu_trace.csv", "e1_cuda_api_trace.csv"):
        (tmp_path / name).write_text("Name,Total Time (ms),Instances\n")
    api_hash = __import__('hashlib').sha256((tmp_path / "e1_cuda_api_trace.csv").read_bytes()).hexdigest()
    (tmp_path / "api-attribution.json").write_text(json.dumps({"schema":"glm53-e1-api-attribution-v1","complete":True,"attributable_cuda_api_ms_per_step":0.1,"source_sha256":{"e1_cuda_api_trace.csv":api_hash}}))
    out = tmp_path / "e1-verdict.json"
    result = run("e1", tmp_path, out)
    assert result.returncode == 0, result.stderr
    receipt = json.loads(out.read_text())
    assert receipt["valid"] is False
    assert receipt["pass"] is False
    assert receipt["verdict"] == "INCONCLUSIVE"
    assert "missing engine-core/graph-node evidence" in receipt["issues"]



def e1_receipts(directory: Path):
    rows = [{'index': i, 'kind': kind, 'completion_tokens': 64, 'finish_reason': 'length', 'metric_delta': {'drafts': 32, 'draft_tokens': 32, 'accepted_tokens': 30}} for i, kind in enumerate(('prose','prose','code','code'))]
    probe = {
        "schema": "glm53-dflash2-uva-acceptance-v1",
        "model": "glm-5.3-big",
        "max_tokens": 64,
        "rows": rows,
        "summary": {"requests": 4, "completion_tokens": 256, "verification_steps": 128, "decode_tok_s_median": 100.0, "acceptance_length_weighted": 2.0},
    }
    (directory / "profiled-probe.json").write_text(json.dumps(probe))
    (directory / "unprofiled-probe.json").write_text(json.dumps(probe))
    buckets = {name: {"per_step_ms": 0.0} for name in ("fused_bookkeeping", "masked_row_copy", "routed_moe", "scalar_gather", "mla_attention", "mtp_verify", "other_gpu")}
    buckets["fused_bookkeeping"]["per_step_ms"] = 1.5
    buckets["scalar_gather"]["per_step_ms"] = 0.6
    (directory / "nsys-buckets.json").write_text(json.dumps({"schema":"glm53-nsys-buckets-v1", "steps": 128, "source_report": "e1_cuda_gpu_kern_sum.csv", "buckets": buckets}))
    (directory / "container.log").write_text("PROFILE_CONTROL ACK START e1-profile-v2\nPROFILE_CONTROL ACK STOP e1-profile-v2\nclean engine log")
    for name in ("e1_cuda_gpu_kern_sum.csv", "e1_cuda_kern_exec_sum.csv", "e1_cuda_gpu_trace.csv", "e1_cuda_api_trace.csv"):
        (directory / name).write_text("Name,Total Time (ms),Instances\n")
    (directory / "engine-core-graph-evidence.json").write_text(json.dumps({"schema":"glm53-e1-engine-core-graph-evidence-v1", "engine_core_capture": True, "graph_node_capture": True}))
    hashes = {name: __import__('hashlib').sha256((directory / name).read_bytes()).hexdigest() for name in ("e1_cuda_gpu_kern_sum.csv", "e1_cuda_kern_exec_sum.csv", "e1_cuda_gpu_trace.csv", "e1_cuda_api_trace.csv", "profiled-probe.json", "profile-wall-seconds.txt", "nsys-buckets.json") if (directory / name).is_file()}
    (directory / "profile-wall-seconds.txt").write_text("1.0\n")
    hashes["profile-wall-seconds.txt"] = __import__('hashlib').sha256((directory / "profile-wall-seconds.txt").read_bytes()).hexdigest()
    hashes["nsys-buckets.json"] = __import__('hashlib').sha256((directory / "nsys-buckets.json").read_bytes()).hexdigest()
    (directory / "api-attribution.json").write_text(json.dumps({"schema":"glm53-e1-api-attribution-v1","complete":True,"attributable_cuda_api_ms_per_step":0.1,"source_sha256": hashes}))


def test_e1_verdict_requires_api_attribution_to_bind_all_current_inputs(tmp_path):
    e1_receipts(tmp_path)
    attr = json.loads((tmp_path / "api-attribution.json").read_text())
    attr["source_sha256"] = {"e1_cuda_api_trace.csv": attr["source_sha256"]["e1_cuda_api_trace.csv"]}
    (tmp_path / "api-attribution.json").write_text(json.dumps(attr))
    out = tmp_path / "e1-verdict.json"
    result = run("e1", tmp_path, out)
    assert result.returncode == 0, result.stderr
    receipt = json.loads(out.read_text())
    assert receipt["valid"] is False
    assert "invalid hash-bound CUDA API attribution" in receipt["issues"]


def test_e1_verdict_rejects_profiled_unprofiled_probe_mismatch(tmp_path):
    e1_receipts(tmp_path)
    profiled = json.loads((tmp_path / "profiled-probe.json").read_text())
    profiled["rows"][0]["kind"] = "code"
    (tmp_path / "profiled-probe.json").write_text(json.dumps(profiled))
    out = tmp_path / "e1-verdict.json"
    result = run("e1", tmp_path, out)
    assert result.returncode == 0, result.stderr
    receipt = json.loads(out.read_text())
    assert receipt["valid"] is False
    assert "profiled/unprofiled probe provenance mismatch" in receipt["issues"]


def test_e1_verdict_rejects_nsys_bucket_source_report_mismatch(tmp_path):
    e1_receipts(tmp_path)
    buckets = json.loads((tmp_path / "nsys-buckets.json").read_text())
    buckets["source_report"] = "stale.csv"
    (tmp_path / "nsys-buckets.json").write_text(json.dumps(buckets))
    out = tmp_path / "e1-verdict.json"
    result = run("e1", tmp_path, out)
    assert result.returncode == 0, result.stderr
    receipt = json.loads(out.read_text())
    assert receipt["valid"] is False
    assert "nsys bucket source report mismatch" in receipt["issues"]
