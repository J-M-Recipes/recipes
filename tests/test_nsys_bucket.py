import importlib.util
import hashlib
import json
import math
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "recipes/dgx-station-gb300/glm-5.3-nvfp4-uva-slot-cache/scripts/nsys_bucket.py"
CONTROL = REPO_ROOT / "recipes/dgx-station-gb300/glm-5.3-nvfp4-uva-slot-cache/scripts/nsys_capture_control.py"
REAL_E1_GPU_SUM = REPO_ROOT / "recipes/dgx-station-gb300/glm-5.3-nvfp4-uva-slot-cache/results/2026-09-08-e1-v2-live/live-receipts/e1_cuda_gpu_kern_sum.csv"


def _trace_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _interval_payload(trace: Path, intervals, **extra):
    payload = {
        "schema": "glm53-nsys-phase-intervals-v1",
        "run_id": "unit-run",
        "clock_domain": "nsys-gpu-globaltimer",
        "trace_sha256": _trace_sha256(trace),
        "time_unit": "ns",
        "intervals": intervals,
    }
    payload.update(extra)
    return payload


def _load_module():
    spec = importlib.util.spec_from_file_location("nsys_bucket", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_buckets_kernel_summary_and_accounts_for_wall_residual(tmp_path):
    report = tmp_path / "cuda_gpu_kern_sum.csv"
    report.write_text(
        'Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name\n'
        '10,10000000,50,0,0,0,0,0,"fused_bookkeeping"\n'
        '12,12000000,100,0,0,0,0,0,"masked_row_copy"\n'
        '20,20000000,50,0,0,0,0,0,"trtllm_fp4_block_scale_routed_moe"\n'
        '5,5000000,200,0,0,0,0,0,"index_put_kernel"\n'
        '8,8000000,50,0,0,0,0,0,"mla_fwd_kernel"\n'
        '4,4000000,50,0,0,0,0,0,"mtp_speculator_kernel"\n'
        '6,6000000,100,0,0,0,0,0,"vectorized_elementwise_kernel"\n'
    )
    module = _load_module()

    result = module.summarize_report(report, steps=50, wall_ms=100.0)

    assert result["schema"] == "glm53-nsys-buckets-v1"
    assert result["steps"] == 50
    assert result["gpu_total_ms"] == 65.0
    assert result["wall_residual_ms"] == 35.0
    assert result["accounted_wall_fraction"] == 0.65
    assert result["buckets"]["fused_bookkeeping"]["total_ms"] == 10.0
    assert result["buckets"]["masked_row_copy"]["per_step_ms"] == 0.24
    assert result["buckets"]["routed_moe"]["total_ms"] == 20.0
    assert result["buckets"]["scalar_gather"]["total_ms"] == 5.0
    assert result["buckets"]["mla_attention"]["total_ms"] == 8.0
    assert result["buckets"]["mtp_verify"]["total_ms"] == 4.0
    assert result["buckets"]["other_gpu"]["total_ms"] == 6.0


def test_e1_v2_real_grouped_gemm_kernel_names_are_bucketed(tmp_path):
    report = tmp_path / "cuda_gpu_kern_sum.csv"
    report.write_text(
        'Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name\n'
        '1,1563525595,300,0,0,0,0,0,"bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x32x512u2_s5_et128x32_m256x32x64_c2x1x1_rM_TN_transOut_schedS_biasFp32M_bN_tma_tmaSf_rgTma_clmp_swiGlu_dynB_sm100f"\n'
        '1,780136061,300,0,0,0,0,0,"bmm_Bfloat16_E2m1E2m1_Fp32_Ab16_Bb16_t128x32x512u2_s4_et128x32_m256x32x64_c2x1x1_rM_TN_transOut_schedS_biasFp32M_bN_rgTma_clmp_dynB_sm100f"\n'
        '1,453966014,11869,0,0,0,0,0,"nvjet_sm103_tst_64x8_64x16_2x1_v_bz_splitK_TNT"\n'
        '1,114124800,35199,0,0,0,0,0,"void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, float, __nv_bfloat16, float, __nv_bfloat16, (bool)0, float, __nv_bfloat16, __nv_bfloat16, (bool)1, (bool)0, (bool)0, (bool)0>(cublasLt::cublasSplitKParams<T6>, const T4 *, const T10 *, T9 *, T5 *, const T6 *, const T6 *, const T11 *, const T4 *, T11 *, void *, long, T6 *, int *, T6 *, T6 *, const T6 *, const T6 *, const T6 *, const T6 *, const T6 *)"\n'
    )
    module = _load_module()

    result = module.summarize_report(report, steps=140, wall_ms=8217.965)

    assert result["buckets"]["routed_moe"]["total_ms"] == 2343.661656
    assert result["buckets"]["routed_moe"]["instances"] == 600
    assert result["buckets"]["dense_gemm"]["total_ms"] == 568.090814
    assert result["buckets"]["dense_gemm"]["instances"] == 47068
    assert result["buckets"]["other_gpu"]["total_ms"] == 0.0


def test_actual_e1_v2_kernel_summary_fixture_preserves_legacy_bucket_totals():
    module = _load_module()

    result = module.summarize_report(REAL_E1_GPU_SUM, steps=140, wall_ms=8217.965)

    assert result["source_report"] == "e1_cuda_gpu_kern_sum.csv"
    assert result["kernel_rows"] == 108
    assert result["gpu_total_ms"] == 8389.216241
    assert result["overlap_ms"] == 171.251241
    assert result["buckets"]["routed_moe"]["instances"] == 22050
    assert result["buckets"]["dense_gemm"]["instances"] == 124007


def test_phase_trace_requires_explicit_intervals_and_aggregates_only_wholly_contained_kernels(tmp_path):
    trace = tmp_path / "e1_cuda_gpu_trace.csv"
    trace.write_text(
        'Start (ns),Duration (ns),Grid X,Block X,Stream,Name\n'
        '100,25,1,1,7,"fused_bookkeeping"\n'
        '140,10,1,1,7,"masked_row_copy"\n'
        '200,40,1,1,7,"bmm_E2m1_E2m1E2m1_Fp32"\n'
        '260,15,1,1,7,"nvjet_sm103_tst_64x8"\n'
        '400,50,1,1,7,"mla_fwd_kernel"\n'
    )
    intervals = tmp_path / "decode-intervals.json"
    intervals.write_text(json.dumps(_interval_payload(
        trace,
        [
            {"phase": "prefill", "start": 0, "end": 180},
            {"phase": "decode", "start": 180, "end": 320, "steps": 2},
        ],
    )))
    module = _load_module()

    result = module.summarize_phase_trace(trace, intervals, steps=2, wall_ms=1.0)

    assert result["schema"] == "glm53-nsys-phase-buckets-v1"
    assert result["methodology"] == "timestamp interval analysis; kernels are bucketed only when wholly contained in one explicit interval"
    assert result["inputs"]["gpu_trace"]["path"] == "e1_cuda_gpu_trace.csv"
    assert result["inputs"]["gpu_trace"]["sha256"] == _trace_sha256(trace)
    assert result["inputs"]["intervals"]["path"] == "decode-intervals.json"
    assert result["interval_contract"]["run_id"] == "unit-run"
    assert result["interval_contract"]["clock_domain"] == "nsys-gpu-globaltimer"
    assert result["interval_contract"]["trace_sha256"] == _trace_sha256(trace)
    assert result["interval_contract"]["attribution"] == "external-explicit-intervals"
    assert result["steps"] == 2
    assert result["wall_ms"] == 1.0
    assert result["outside_interval"]["kernel_rows"] == 1
    assert result["phases"]["prefill"]["kernel_rows"] == 2
    assert result["phases"]["prefill"]["gpu_total_ms"] == 0.000035
    assert "gpu_per_step_ms" not in result["phases"]["prefill"]
    assert "per_step_ms" not in result["phases"]["prefill"]["buckets"]["masked_row_copy"]
    assert result["phases"]["prefill"]["buckets"]["fused_bookkeeping"]["total_ms"] == 0.000025
    assert result["phases"]["prefill"]["buckets"]["masked_row_copy"]["instances"] == 1
    assert result["phases"]["decode"]["kernel_rows"] == 2
    assert result["phases"]["decode"]["gpu_total_ms"] == 0.000055
    assert result["phases"]["decode"]["gpu_per_step_ms"] == 0.0000275
    assert result["phases"]["decode"]["buckets"]["routed_moe"]["total_ms"] == 0.00004
    assert result["phases"]["decode"]["buckets"]["routed_moe"]["per_step_ms"] == 0.00002
    assert result["phases"]["decode"]["buckets"]["dense_gemm"]["total_ms"] == 0.000015
    assert result["totals"]["gpu_total_ms"] == 0.00009


def test_phase_trace_fails_closed_when_kernel_crosses_interval_boundary(tmp_path):
    trace = tmp_path / "trace.csv"
    trace.write_text('Start (ns),Duration (ns),Name\n90,20,"masked_row_copy"\n')
    intervals = tmp_path / "intervals.json"
    intervals.write_text(json.dumps(_interval_payload(trace, [{"phase": "decode", "start": 100, "end": 200}])) )
    module = _load_module()

    with pytest.raises(ValueError, match="crosses interval boundary"):
        module.summarize_phase_trace(trace, intervals)


@pytest.mark.parametrize(
    "interval_payload,error",
    [
        ({"intervals": [{"phase": "a", "start": 0, "end": 10}, {"phase": "b", "start": 9, "end": 20}]}, "overlap"),
        ({"intervals": [{"phase": "decode", "start": -1, "end": 10}]}, "non-negative"),
        ({"intervals": [{"phase": "decode", "start": 0, "end": "nan"}]}, "finite"),
        ({"intervals": [{"phase": "decode", "start": 10, "end": 10}]}, "greater than start"),
        ({"schema": "wrong", "intervals": [{"phase": "decode", "start": 0, "end": 10}]}, "schema"),
        ({"run_id": "", "intervals": [{"phase": "decode", "start": 0, "end": 10}]}, "run_id"),
        ({"clock_domain": "", "intervals": [{"phase": "decode", "start": 0, "end": 10}]}, "clock_domain"),
        ({"trace_sha256": "00", "intervals": [{"phase": "decode", "start": 0, "end": 10}]}, "trace_sha256"),
        ({"intervals": [{"phase": "decode", "start": 0, "start_ns": 1, "end": 10}]}, "conflicting aliases"),
    ],
)
def test_phase_trace_fails_closed_for_bad_interval_contracts(tmp_path, interval_payload, error):
    trace = tmp_path / "trace.csv"
    trace.write_text('Start (ns),Duration (ns),Name\n1,1,"kernel"\n')
    intervals = tmp_path / "intervals.json"
    complete_payload = _interval_payload(trace, interval_payload.pop("intervals"), **interval_payload)
    intervals.write_text(json.dumps(complete_payload))
    module = _load_module()

    with pytest.raises(ValueError, match=error):
        module.summarize_phase_trace(trace, intervals)


@pytest.mark.parametrize(
    "row,error",
    [
        ('-1,1,"kernel"\n', "non-negative"),
        ('1,-1,"kernel"\n', "positive"),
        ('1,nan,"kernel"\n', "finite"),
        ('1,2,\n', "Name"),
    ],
)
def test_phase_trace_fails_closed_for_malformed_gpu_trace_rows(tmp_path, row, error):
    trace = tmp_path / "trace.csv"
    trace.write_text('Start (ns),Duration (ns),Name\n' + row)
    intervals = tmp_path / "intervals.json"
    intervals.write_text(json.dumps(_interval_payload(trace, [{"phase": "decode", "start": 0, "end": 10}])) )
    module = _load_module()

    with pytest.raises(ValueError, match=error):
        module.summarize_phase_trace(trace, intervals)


def test_phase_cli_is_opt_in_and_writes_phase_summary(tmp_path):
    trace = tmp_path / "trace.csv"
    trace.write_text('Start (ns),Duration (ns),Name\n0,10,"fused_bookkeeping"\n')
    intervals = tmp_path / "intervals.json"
    intervals.write_text(json.dumps(_interval_payload(trace, [{"phase": "decode", "start": 0, "end": 10}])) )
    output = tmp_path / "summary.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(trace),
            "--gpu-trace",
            "--decode-intervals",
            str(intervals),
            "--output",
            str(output),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(output.read_text())
    assert summary["schema"] == "glm53-nsys-phase-buckets-v1"
    assert summary["phases"]["decode"]["buckets"]["fused_bookkeeping"]["total_ms"] == 0.00001
    assert "NSYS_PHASE_BUCKETS" in result.stdout


def test_phase_trace_preserves_one_ns_boundary_above_float_precision(tmp_path):
    base = 2**53
    trace = tmp_path / "trace.csv"
    trace.write_text(
        'Start (ns),Duration (ns),Name\n'
        f'{base},1,"masked_row_copy"\n'
        f'{base + 1},1,"bmm_E2m1_E2m1E2m1_Fp32"\n'
    )
    intervals = tmp_path / "intervals.json"
    intervals.write_text(json.dumps(_interval_payload(trace, [
        {"phase": "prefill", "start": str(base), "end": str(base + 1)},
        {"phase": "decode", "start": str(base + 1), "end": str(base + 2), "steps": 1},
    ])))
    module = _load_module()

    result = module.summarize_phase_trace(trace, intervals, steps=1)

    assert result["phases"]["prefill"]["buckets"]["masked_row_copy"]["total_ms"] == 0.000001
    assert result["phases"]["decode"]["buckets"]["routed_moe"]["total_ms"] == 0.000001


def test_phase_trace_normalizes_decimal_units_only_when_integral_ns(tmp_path):
    trace = tmp_path / "trace.csv"
    trace.write_text('Start (us),Duration (us),Name\n9007199254740.992,0.001,"masked_row_copy"\n')
    intervals = tmp_path / "intervals.json"
    intervals.write_text(json.dumps(_interval_payload(trace, [
        {"phase": "decode", "start": 9007199254740992, "end": 9007199254740993, "steps": 1},
    ])))
    module = _load_module()

    result = module.summarize_phase_trace(trace, intervals, steps=1)

    assert result["phases"]["decode"]["buckets"]["masked_row_copy"]["total_ms"] == 0.000001


def test_phase_trace_rejects_fractional_ns_after_unit_normalization(tmp_path):
    trace = tmp_path / "trace.csv"
    trace.write_text('Start (us),Duration (us),Name\n1.0005,0.001,"masked_row_copy"\n')
    intervals = tmp_path / "intervals.json"
    intervals.write_text(json.dumps(_interval_payload(trace, [
        {"phase": "decode", "start": 1000, "end": 1002},
    ])))
    module = _load_module()

    with pytest.raises(ValueError, match="integral ns"):
        module.summarize_phase_trace(trace, intervals)


def test_phase_trace_rejects_duplicate_json_keys_and_csv_headers(tmp_path):
    trace = tmp_path / "trace.csv"
    trace.write_text('Start (ns),Duration (ns),Name,Name\n0,1,"a","b"\n')
    intervals = tmp_path / "intervals.json"
    intervals.write_text(json.dumps(_interval_payload(trace, [{"phase": "decode", "start": 0, "end": 1}])))
    module = _load_module()

    with pytest.raises(ValueError, match="duplicate.*header"):
        module.summarize_phase_trace(trace, intervals)

    trace.write_text('Start (ns),Duration (ns),Name\n0,1,"a"\n')
    intervals.write_text('{"schema":"glm53-nsys-phase-intervals-v1","schema":"glm53-nsys-phase-intervals-v1","run_id":"unit-run","clock_domain":"nsys-gpu-globaltimer","trace_sha256":"' + _trace_sha256(trace) + '","intervals":[{"phase":"decode","start":0,"end":1}]}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        module.summarize_phase_trace(trace, intervals)


def test_phase_trace_rejects_nonfinite_steps_and_wall_in_api(tmp_path):
    trace = tmp_path / "trace.csv"
    trace.write_text('Start (ns),Duration (ns),Name\n0,1,"a"\n')
    intervals = tmp_path / "intervals.json"
    intervals.write_text(json.dumps(_interval_payload(trace, [{"phase": "decode", "start": 0, "end": 1, "steps": 1}])))
    module = _load_module()

    with pytest.raises(ValueError, match="steps must be a finite positive integer"):
        module.summarize_phase_trace(trace, intervals, steps=math.inf)
    with pytest.raises(ValueError, match="wall_ms must be finite and positive"):
        module.summarize_phase_trace(trace, intervals, wall_ms=math.nan)


def test_phase_trace_parses_json_number_tokens_exactly_above_float_precision(tmp_path):
    base = 9007199254740992
    trace = tmp_path / "trace.csv"
    trace.write_text(
        'Start (ns),Duration (ns),Name\n'
        f'{base},1,"masked_row_copy"\n'
        f'{base + 1},1,"bmm_E2m1_E2m1E2m1_Fp32"\n'
    )
    intervals = tmp_path / "intervals.json"
    payload = (
        '{"schema":"glm53-nsys-phase-intervals-v1",'
        '"run_id":"unit-run",'
        '"clock_domain":"nsys-gpu-globaltimer",'
        f'"trace_sha256":"{_trace_sha256(trace)}",'
        '"time_unit":"ns",'
        '"intervals":['
        f'{{"phase":"prefill","start":{base}e0,"end":{base + 1}e0}},'
        f'{{"phase":"decode","start":{base + 1}e0,"end":{base + 2}e0,"steps":1}}'
        ']}'
    )
    intervals.write_text(payload)
    module = _load_module()

    result = module.summarize_phase_trace(trace, intervals, steps=1)

    assert result["phases"]["prefill"]["buckets"]["masked_row_copy"]["total_ms"] == 0.000001
    assert result["phases"]["decode"]["buckets"]["routed_moe"]["total_ms"] == 0.000001


def test_phase_trace_rejects_ns_aliases_under_non_ns_interval_unit(tmp_path):
    trace = tmp_path / "trace.csv"
    trace.write_text('Start (ns),Duration (ns),Name\n1000,1,"masked_row_copy"\n')
    intervals = tmp_path / "intervals.json"
    intervals.write_text(json.dumps(_interval_payload(
        trace,
        [{"phase": "decode", "start_ns": 1000, "end_ns": 1001}],
        time_unit="us",
    )))
    module = _load_module()

    with pytest.raises(ValueError, match="_ns aliases require intervals time_unit ns"):
        module.summarize_phase_trace(trace, intervals)


def test_phase_trace_sums_decode_steps_across_intervals_and_reconciles_decode_only(tmp_path):
    trace = tmp_path / "trace.csv"
    trace.write_text(
        'Start (ns),Duration (ns),Name\n'
        '0,10,"fused_bookkeeping"\n'
        '100,20,"masked_row_copy"\n'
        '200,30,"bmm_E2m1_E2m1E2m1_Fp32"\n'
    )
    intervals = tmp_path / "intervals.json"
    intervals.write_text(json.dumps(_interval_payload(trace, [
        {"phase": "prefill", "start": 0, "end": 10},
        {"phase": "decode", "start": 100, "end": 120, "steps": 2},
        {"phase": "decode", "start": 200, "end": 230, "steps": 3},
    ])))
    module = _load_module()

    result = module.summarize_phase_trace(trace, intervals, steps=5, wall_ms=0.00005)

    assert result["steps"] == 5
    assert result["phases"]["decode"]["gpu_total_ms"] == 0.00005
    assert result["phases"]["decode"]["gpu_per_step_ms"] == 0.00001
    assert result["phases"]["decode"]["wall_ms"] == 0.00005
    assert result["phases"]["decode"]["accounted_wall_fraction"] == 1.0
    assert result["totals"] == {"gpu_total_ms": 0.00006}


def test_phase_trace_requires_all_decode_intervals_to_declare_steps_when_any_do(tmp_path):
    trace = tmp_path / "trace.csv"
    trace.write_text('Start (ns),Duration (ns),Name\n0,1,"a"\n10,1,"b"\n')
    intervals = tmp_path / "intervals.json"
    intervals.write_text(json.dumps(_interval_payload(trace, [
        {"phase": "decode", "start": 0, "end": 1, "steps": 1},
        {"phase": "decode", "start": 10, "end": 11},
    ])))
    module = _load_module()

    with pytest.raises(ValueError, match="all decode intervals must declare steps"):
        module.summarize_phase_trace(trace, intervals)


def test_phase_trace_allows_cli_steps_total_when_decode_intervals_have_no_declarations(tmp_path):
    trace = tmp_path / "trace.csv"
    trace.write_text('Start (ns),Duration (ns),Name\n0,1,"a"\n10,1,"b"\n')
    intervals = tmp_path / "intervals.json"
    intervals.write_text(json.dumps(_interval_payload(trace, [
        {"phase": "decode", "start": 0, "end": 1},
        {"phase": "decode", "start": 10, "end": 11},
    ])))
    module = _load_module()

    result = module.summarize_phase_trace(trace, intervals, steps=2)

    assert result["phases"]["decode"]["gpu_per_step_ms"] == 0.000001


def test_phase_trace_rejects_extra_csv_cells_conflicting_name_aliases_and_float_overflow(tmp_path):
    trace = tmp_path / "trace.csv"
    intervals = tmp_path / "intervals.json"
    module = _load_module()

    trace.write_text('Start (ns),Duration (ns),Name\n0,1,"a",extra\n')
    intervals.write_text(json.dumps(_interval_payload(trace, [{"phase": "decode", "start": 0, "end": 1}])))
    with pytest.raises(ValueError, match="extra CSV cells"):
        module.summarize_phase_trace(trace, intervals)

    trace.write_text('Start (ns),Duration (ns),Name,Kernel Name\n0,1,"a","b"\n')
    intervals.write_text(json.dumps(_interval_payload(trace, [{"phase": "decode", "start": 0, "end": 1}])))
    with pytest.raises(ValueError, match="conflicting Name and Kernel Name"):
        module.summarize_phase_trace(trace, intervals)

    trace.write_text('Start (ns),Duration (ns),Name\n0,1e309,"a"\n')
    intervals.write_text(json.dumps(_interval_payload(trace, [{"phase": "decode", "start": 0, "end": 10**310}])))
    with pytest.raises(ValueError, match="too large to emit as finite milliseconds"):
        module.summarize_phase_trace(trace, intervals)


def test_capture_control_atomically_waits_for_matching_ack(tmp_path):
    control = tmp_path / "nsys-control"

    def acknowledge():
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if control.exists() and control.read_text().strip() == "START e1-run":
                control.write_text("ACK START e1-run\n")
                return
            time.sleep(0.005)
        raise AssertionError("controller never wrote START command")

    thread = threading.Thread(target=acknowledge)
    thread.start()
    result = subprocess.run(
        [sys.executable, str(CONTROL), str(control), "START", "e1-run", "--timeout", "1", "--poll", "0.005"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    thread.join()

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ACK START e1-run"
    assert not (tmp_path / "nsys-control.tmp").exists()


def test_capture_control_times_out_without_ack(tmp_path):
    control = tmp_path / "nsys-control"
    result = subprocess.run(
        [sys.executable, str(CONTROL), str(control), "STOP", "e1-run", "--timeout", "0.02", "--poll", "0.005"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    assert "timed out waiting for ACK STOP e1-run" in result.stderr


def test_cli_writes_json_and_reports_overlap_when_gpu_work_exceeds_wall(tmp_path):
    report = tmp_path / "cuda_gpu_kern_sum.csv"
    report.write_text(
        'Time (%),Total Time (us),Instances,Avg (us),Name\n'
        '100,120000,10,12000,"trtllm_fp4_block_scale_routed_moe"\n'
    )
    output = tmp_path / "summary.json"

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(report), "--steps", "10", "--wall-ms", "100", "--output", str(output)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(output.read_text())
    assert summary["gpu_total_ms"] == 120.0
    assert summary["wall_residual_ms"] == 0.0
    assert summary["overlap_ms"] == 20.0
    assert summary["accounted_wall_fraction"] == 1.0


def test_legacy_summary_rejects_duplicate_headers_and_nonfinite_inputs(tmp_path):
    report = tmp_path / "cuda_gpu_kern_sum.csv"
    report.write_text('Total Time (ns),Total Time (ns),Instances,Name\n1,2,1,"kernel"\n')
    module = _load_module()

    with pytest.raises(ValueError, match="duplicate.*header"):
        module.summarize_report(report, steps=1, wall_ms=1.0)

    report.write_text('Total Time (ns),Instances,Name\n1,1,"kernel"\n')
    with pytest.raises(ValueError, match="steps must be a finite positive integer"):
        module.summarize_report(report, steps=True, wall_ms=1.0)
    with pytest.raises(ValueError, match="wall_ms must be finite and positive"):
        module.summarize_report(report, steps=1, wall_ms=math.inf)
