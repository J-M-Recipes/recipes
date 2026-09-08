import importlib.util
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "recipes/dgx-station-gb300/glm-5.3-nvfp4-uva-slot-cache/scripts/nsys_bucket.py"
CONTROL = REPO_ROOT / "recipes/dgx-station-gb300/glm-5.3-nvfp4-uva-slot-cache/scripts/nsys_capture_control.py"


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
