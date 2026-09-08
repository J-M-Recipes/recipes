import ast
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RECIPE = REPO_ROOT / "recipes/dgx-station-gb300/glm-5.3-nvfp4-uva-slot-cache"
PATCHES = RECIPE / "patches"
RUNTIME_GPU_MODEL_RUNNER = Path("/Users/jamesmeadlock/.hermes/profiles/milo/work/k2-v3-runtime-source/complete/vllm/v1/worker/gpu_model_runner.py")
EXPECTED_GPU_MODEL_RUNNER_SHA256 = "7f2890eefca1efe25565bf1c7e5906a87948ae922610a7aaac620b28b46f26aa"
PYTHON = "/Users/jamesmeadlock/hermes/jm-recipes/recipes/.venv/bin/python"


def _load_instrumentation():
    spec = importlib.util.spec_from_file_location(
        "slot_cache_window_instrumentation",
        PATCHES / "slot_cache_window_instrumentation.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CachedReqs:
    def __init__(self, req_ids=(), context=()):
        self.req_ids = list(req_ids)
        self._context = set(context)

    def is_context_phase(self, req_id):
        return req_id in self._context


def sched(*, total=0, new=(), cached=(), context=(), spec=None, num_tokens=None, engine_step=None):
    if num_tokens is None:
        num_tokens = {rid: 1 for rid in [*new, *cached]}
    output = SimpleNamespace(
        total_num_scheduled_tokens=total,
        scheduled_new_reqs=list(new),
        scheduled_cached_reqs=CachedReqs(cached, context),
        scheduled_spec_decode_tokens=spec or {},
        num_scheduled_tokens=num_tokens,
    )
    if engine_step is not None:
        output.engine_step = engine_step
    return output


def test_phase_classifier_uses_scheduled_active_requests_not_all_cached_requests():
    inst = _load_instrumentation()

    assert inst.classify_slot_cache_phase(sched(total=0))["canonical_phase"] == "zero_work"
    assert inst.classify_slot_cache_phase(sched(total=4, new=["n"]))["canonical_phase"] == "prefill"
    assert inst.classify_slot_cache_phase(sched(total=4, cached=["c"], context=["c"]))["canonical_phase"] == "prefill"
    assert inst.classify_slot_cache_phase(sched(total=1, cached=["d"]))["canonical_phase"] == "decode"

    phase = inst.classify_slot_cache_phase(
        sched(total=3, cached=["d"], spec={"d": [10, 11]}, num_tokens={"d": 3})
    )
    assert phase["has_spec_verify"] is True
    assert phase["has_decode"] is False
    assert phase["canonical_phase"] == "spec_verify"

    mixed = inst.classify_slot_cache_phase(sched(total=5, new=["n"], cached=["d"], spec={"d": [1]}))
    assert mixed["has_mixed"] is True
    assert mixed["canonical_phase"] == "mixed"
    assert "k1" not in json.dumps(phase).lower()
    assert "k2" not in json.dumps(phase).lower()


def test_trace_marker_has_stable_correlation_fields_without_host_timestamps():
    inst = _load_instrumentation()
    phase = inst.classify_slot_cache_phase(sched(total=1, cached=["d"]))
    marker = inst.make_slot_cache_trace_id(
        run_id="run-a",
        engine_generation=2,
        seq=3,
        engine_step=4,
        boundary="step_complete",
        phase_flags=phase,
        graph_mode="FULL",
        k_mode="K2",
        has_drafter_config=True,
        drafter_runs_model_forward=False,
    )
    assert marker == "slotcache:run-a:gen:2:seq:3:step:4:boundary:step_complete:phase:decode:flags:decode,drafter:graph:FULL:k:K2"
    row = inst.snapshot_to_json_row(
        trace_id=marker,
        provenance={"run_id": "run-a", "source_sha": "a" * 64, "engine_generation": 2, "seq": 3},
        metadata={"engine_step": 4, "boundary": "step_complete", "phase_flags": phase},
        layers={"0": {"misses": 0, "routes": 1, "steps": 1}},
    )
    assert row["metadata"]["trace_id"] == marker
    assert "start_ns" not in row and "end_ns" not in row


def test_window_controller_parses_bounded_windows_and_rejects_path_escape_or_overwrite(tmp_path, monkeypatch):
    inst = _load_instrumentation()
    out = tmp_path / "safe"
    out.mkdir()
    monkeypatch.setenv("SLOT_CACHE_QUIESCENT_SNAPSHOTS", "1")
    monkeypatch.setenv("SLOT_CACHE_SNAPSHOT_DIR", str(out))
    monkeypatch.setenv("SLOT_CACHE_WINDOW_STEPS", "2:4,10:12")
    monkeypatch.setenv("SLOT_CACHE_RUN_ID", "run-a")
    monkeypatch.setenv("SLOT_CACHE_SOURCE_SHA", "b" * 64)
    monkeypatch.setenv("SLOT_CACHE_K_MODE", "K1")
    ctl = inst.SlotCacheWindowController.from_env()
    assert ctl.enabled is True
    assert ctl.should_snapshot(2) == "start"
    assert ctl.should_snapshot(4) == "end"
    assert ctl.should_snapshot(3) is None
    first = ctl.reserve_output_path()
    assert first.parent == out
    first.write_text("occupied")
    with pytest.raises(FileExistsError):
        ctl.reserve_output_path()

    monkeypatch.setenv("SLOT_CACHE_SNAPSHOT_DIR", str(out / ".." / "escape"))
    with pytest.raises(ValueError, match="must already exist"):
        inst.SlotCacheWindowController.from_env()


def test_snapshot_device_helper_fail_closes_and_private_copies_with_fake_tensors(monkeypatch):
    sys.path.insert(0, str(PATCHES))
    try:
        import slot_cache_hook
    finally:
        sys.path.pop(0)

    class Scalar:
        def __init__(self, value, device="cuda:0"):
            self.value = value
            self.device = device
            self.item_called = False
            self.cpu_called = False

        def clone_value(self):
            return self.value

        def item(self):
            self.item_called = True
            raise AssertionError("item forbidden")

        def cpu(self):
            self.cpu_called = True
            raise AssertionError("cpu forbidden")

    class FakeTorch:
        int64 = "int64"

        @staticmethod
        def empty(shape, dtype=None, device=None):
            return [[None for _ in range(shape[1])] for _ in range(shape[0])]

    registry = {}
    for layer in list(range(75))[::-1]:
        registry[layer] = SimpleNamespace(
            name=f"model.layers.{layer}.mlp.experts",
            misses=Scalar(layer),
            routes=Scalar(layer + 100),
            step=Scalar(7),
        )
    monkeypatch.setattr(slot_cache_hook, "_registry", registry)
    monkeypatch.setattr(slot_cache_hook, "torch", FakeTorch)

    assert slot_cache_hook.slot_cache_registry_ready(75) is True
    snap = slot_cache_hook.slot_cache_snapshot_device(
        trace_id="slotcache:run",
        provenance={"run_id": "run", "source_sha": "c" * 64, "engine_generation": 1, "seq": 1},
        metadata={"boundary": "step_complete"},
        expected_layers=75,
    )
    assert snap.layer_ids == list(range(75))
    assert snap.counters_device[0] == [0, 100, 7]
    assert snap.counters_device[74] == [74, 174, 7]
    registry[0].misses.value = 999
    assert snap.counters_device[0] == [0, 100, 7]

    del registry[73]
    assert slot_cache_hook.slot_cache_snapshot_device(
        trace_id="slotcache:run",
        provenance={"run_id": "run", "source_sha": "c" * 64, "engine_generation": 1, "seq": 2},
        metadata={"boundary": "step_complete"},
        expected_layers=75,
    ) is None


def _apply_patch(tmp_path):
    out = tmp_path / "gpu_model_runner.patched.py"
    script = RECIPE / "scripts/apply_slot_cache_instrumentation_patch.py"
    result = __import__("subprocess").run(
        [PYTHON, str(script), "--source", str(RUNTIME_GPU_MODEL_RUNNER), "--output", str(out)],
        cwd=REPO_ROOT,
        text=True,
        stdout=__import__("subprocess").PIPE,
        stderr=__import__("subprocess").PIPE,
        check=False,
    )
    return result, out


def test_patch_script_exact_hash_guard_path_safety_and_compile(tmp_path):
    assert hashlib.sha256(RUNTIME_GPU_MODEL_RUNNER.read_bytes()).hexdigest() == EXPECTED_GPU_MODEL_RUNNER_SHA256
    result, out = _apply_patch(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "source_sha256=7f2890eefca1efe25565bf1c7e5906a87948ae922610a7aaac620b28b46f26aa" in result.stdout
    assert out.exists()
    bad = tmp_path / "bad.py"
    bad.write_text(RUNTIME_GPU_MODEL_RUNNER.read_text().replace("class GPUModelRunner", "class GPUModelRunnerX", 1))
    blocked = __import__("subprocess").run(
        [PYTHON, str(RECIPE / "scripts/apply_slot_cache_instrumentation_patch.py"), "--source", str(bad), "--output", str(tmp_path / "bad.out")],
        text=True,
        stdout=__import__("subprocess").PIPE,
        stderr=__import__("subprocess").PIPE,
        check=False,
    )
    assert blocked.returncode == 2
    assert "sha256 mismatch" in blocked.stderr


def test_patched_gpu_runner_call_order_and_no_per_step_sync_in_instrumentation_path(tmp_path):
    result, out = _apply_patch(tmp_path)
    assert result.returncode == 0, result.stderr
    __import__("py_compile").compile(str(out), doraise=True)
    source = out.read_text()
    async_ctor = source.rindex("AsyncGPUModelRunnerOutput(")
    assert source.index("self.eplb_step()") < source.index("slot_snapshot = self.slot_cache_window_controller.maybe_snapshot_device") < source.index('record_function_or_nullcontext("gpu_model_runner: ModelRunnerOutput")') < async_ctor
    assert source.index("self._model_forward(") < source.index("self.slot_cache_window_controller.mark_target_forward_boundary") < source.index('record_function_or_nullcontext("gpu_model_runner: postprocess")')
    per_step_slice = source[source.index("slot_snapshot = self.slot_cache_window_controller.maybe_snapshot_device"):source.index('record_function_or_nullcontext("gpu_model_runner: ModelRunnerOutput")')]
    forbidden = [".item(", ".cpu(", ".tolist(", "torch.cuda.synchronize(", "torch.save("]
    assert not any(token in per_step_slice for token in forbidden)
    tree = ast.parse(source)
    async_inits = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", "")) == "AsyncGPUModelRunnerOutput"]
    assert any(any(kw.arg == "slot_cache_snapshot" for kw in call.keywords) for call in async_inits)


def test_async_output_ast_behavior_waits_stream_and_finishes_snapshot_before_abort_fault(tmp_path):
    result, out = _apply_patch(tmp_path)
    assert result.returncode == 0, result.stderr
    tree = ast.parse(out.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AsyncGPUModelRunnerOutput")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    get_output = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "get_output")
    get_src = ast.get_source_segment(out.read_text(), get_output)
    assert get_src.index("self.async_copy_ready_event.synchronize()") < get_src.index("finalize_slot_cache_snapshot") < get_src.index("if self._has_fault is not None")

    calls = []

    class FakeTensor:
        def __init__(self, data):
            self.data = data
            self.shape = (1, 1)

        def to(self, target, non_blocking=False):
            calls.append(("to", target, non_blocking, self.data))
            return FakeTensor([row[:] for row in self.data])

        def tolist(self):
            return [row[:] for row in self.data]

    class FakeEvent:
        def __init__(self, blocking=False):
            calls.append(("event", blocking))

        def record(self):
            calls.append("record")

        def synchronize(self):
            calls.append("synchronize")

    class FakeCuda:
        Stream = object

        @staticmethod
        def Event(blocking=False):
            return FakeEvent(blocking=blocking)

        @staticmethod
        def current_stream():
            calls.append("current_stream")
            return "default-stream"

        @staticmethod
        def stream(stream):
            class Ctx:
                def __enter__(self):
                    calls.append(("enter_stream", stream.name))

                def __exit__(self, *_exc):
                    calls.append(("exit_stream", stream.name))
            return Ctx()

    class FakeTorch:
        cuda = FakeCuda
        Tensor = FakeTensor

    class FakeStream:
        name = "copy-stream"

        def wait_stream(self, stream):
            calls.append(("wait_stream", stream))

    namespace = {
        "AsyncModelRunnerOutput": object,
        "ModelRunnerOutput": object,
        "torch": FakeTorch,
        "LogprobsTensors": object,
        "RoutedExpertsTensors": object,
        "SlotCacheDeviceSnapshot": object,
        "SlotCacheWindowController": object,
        "Any": object,
        "get_ep_all2all_manager": lambda: None,
        "RejectionSampler": object,
        "nans_to_dict": lambda *_args: {},
        "envs": SimpleNamespace(VLLM_RAISE_ON_LOGIT_NANS=False),
        "raise_if_nan_logits": lambda *_args: None,
        "finalize_slot_cache_snapshot": lambda controller, snapshot, cpu: calls.append(("finalize", controller, snapshot, cpu.data)),
    }
    module = ast.Module(body=[cls], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, filename=str(out), mode="exec"), namespace)
    output = SimpleNamespace(sampled_token_ids=[], logprobs=None, req_id_to_index={})
    snapshot = SimpleNamespace(counters_device=FakeTensor([[1, 2, 3]]))
    async_output = namespace["AsyncGPUModelRunnerOutput"](
        model_runner_output=output,
        sampled_token_ids=FakeTensor([[42]]),
        logprobs_tensors=None,
        invalid_req_indices=[],
        async_output_copy_stream=FakeStream(),
        vocab_size=100,
        slot_cache_snapshot=snapshot,
        slot_cache_window_controller="controller",
    )
    assert ("wait_stream", "default-stream") in calls
    assert calls.index(("wait_stream", "default-stream")) < calls.index(("to", "cpu", True, [[1, 2, 3]]))
    returned = async_output.get_output()
    assert returned is output
    assert calls.index("synchronize") < next(i for i, c in enumerate(calls) if isinstance(c, tuple) and c[0] == "finalize")


def test_runtime_canary_acceptance_doc_lists_unproven_stream_and_correlation_claims():
    doc = (RECIPE / "RUNTIME-CANARY-ACCEPTANCE.md").read_text()
    required = [
        "not proven by local source-level tests",
        "current/default stream",
        "no per-step sync",
        "Nsight/profiler correlation",
        "target/draft contamination",
        "abort/stale path",
        "GPU timestamps must come from profiler traces",
    ]
    for phrase in required:
        assert phrase in doc



def test_nvtx_range_preserves_original_body_exception_and_records_cleanup_failure(monkeypatch):
    inst = _load_instrumentation()
    events = []

    class Nvtx:
        @staticmethod
        def range_push(trace_id):
            events.append(("push", trace_id))

        @staticmethod
        def range_pop():
            events.append(("pop", None))
            raise RuntimeError("synthetic pop failure")

    class FakeTorch:
        cuda = SimpleNamespace(nvtx=Nvtx)

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    ctl = inst.SlotCacheWindowController(enabled=True, run_id="nvtx")
    original = ValueError("body identity must survive")

    with pytest.raises(ValueError) as raised:
        with ctl.nvtx_range("slotcache:identity"):
            raise original

    assert raised.value is original
    assert events == [("push", "slotcache:identity"), ("pop", None)]
    assert ctl.instrumentation_failures[-1]["operation"] == "nvtx_range_pop"
    assert ctl.instrumentation_failures[-1]["error_type"] == "RuntimeError"


def test_nvtx_range_push_failure_records_and_yields_body_once(monkeypatch):
    inst = _load_instrumentation()
    events = []

    class Nvtx:
        @staticmethod
        def range_push(trace_id):
            events.append(("push", trace_id))
            raise RuntimeError("synthetic push failure")

        @staticmethod
        def range_pop():
            events.append(("pop", None))

    class FakeTorch:
        cuda = SimpleNamespace(nvtx=Nvtx)

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    ctl = inst.SlotCacheWindowController(enabled=True, run_id="nvtx")
    body_entries = 0

    with ctl.nvtx_range("slotcache:push-fails"):
        body_entries += 1

    assert body_entries == 1
    assert events == [("push", "slotcache:push-fails")]
    assert ctl.instrumentation_failures[-1]["operation"] == "nvtx_range_push"


def test_nvtx_range_cleanup_failure_on_success_is_recorded_not_raised(monkeypatch):
    inst = _load_instrumentation()

    class Nvtx:
        @staticmethod
        def range_push(trace_id):
            pass

        @staticmethod
        def range_pop():
            raise RuntimeError("synthetic pop failure")

    class FakeTorch:
        cuda = SimpleNamespace(nvtx=Nvtx)

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    ctl = inst.SlotCacheWindowController(enabled=True, run_id="nvtx")
    with ctl.nvtx_range("slotcache:cleanup-success"):
        pass
    assert ctl.instrumentation_failures[-1]["operation"] == "nvtx_range_pop"


def test_generated_import_init_seam_disabled_noop_and_enabled_import_errors_are_clear(tmp_path, monkeypatch):
    inst = _load_instrumentation()
    script_spec = importlib.util.spec_from_file_location(
        "apply_slot_cache_instrumentation_patch",
        RECIPE / "scripts/apply_slot_cache_instrumentation_patch.py",
    )
    script = importlib.util.module_from_spec(script_spec)
    assert script_spec.loader is not None
    script_spec.loader.exec_module(script)
    patched = script.patch_source(RUNTIME_GPU_MODEL_RUNNER.read_text())
    assert "SlotCacheDeviceSnapshot" not in patched
    import_prefix = patched.split("import numpy as np", 1)[0]
    init_line = "if SlotCacheWindowController is None:"
    init_start = patched.index(init_line) - 8
    init_end = patched.index("\n\n        self.check_ep_fault = False", init_start)
    init_src = "def init_seam(self):\n" + "\n".join(
        "    " + line[8:] for line in patched[init_start:init_end].splitlines()
    ) + "\n"
    # Real adapter present: disabled mode is an actual no-op controller None.
    monkeypatch.syspath_prepend(str(PATCHES))
    monkeypatch.delenv("SLOT_CACHE_QUIESCENT_SNAPSHOTS", raising=False)
    ns = {"Any": object}
    exec(import_prefix + "\n" + init_src, ns)
    runner = SimpleNamespace()
    ns["init_seam"](runner)
    assert runner.slot_cache_window_controller is None

    # Missing adapter is tolerated only while disabled.
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(sys.modules, "slot_cache_window_instrumentation", None)
    ns = {"Any": object}
    exec(import_prefix + "\n" + init_src, ns)
    runner = SimpleNamespace()
    ns["init_seam"](runner)
    assert runner.slot_cache_window_controller is None

    # Enabled mode must surface a clear import/config error instead of silent None.
    monkeypatch.setenv("SLOT_CACHE_QUIESCENT_SNAPSHOTS", "1")
    with pytest.raises(RuntimeError, match="slot-cache quiescent snapshots are enabled"):
        ns["init_seam"](SimpleNamespace())


def test_generated_launch_and_complete_seams_share_one_fallback_step_identity(monkeypatch):
    inst = _load_instrumentation()
    pushes = []
    calls = []

    class Nvtx:
        @staticmethod
        def range_push(trace_id):
            pushes.append(trace_id)

        @staticmethod
        def range_pop():
            pass

    class FakeTorch:
        cuda = SimpleNamespace(nvtx=Nvtx)

    class Hook:
        @staticmethod
        def slot_cache_snapshot_device(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                trace_id=kwargs["trace_id"],
                layer_ids=[0],
                counters_device=[[0, 1, 1]],
                provenance=kwargs["provenance"],
                metadata=kwargs["metadata"],
            )

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(sys.modules, "slot_cache_hook", Hook)
    ctl = inst.SlotCacheWindowController(
        enabled=True,
        snapshot_dir=Path("/tmp"),
        windows=(inst.SlotCacheWindow(1, 2),),
        run_id="align",
        k_mode="K2",
    )

    first = sched(total=1, cached=["d"])
    with ctl.nvtx_range_for_boundary(
        scheduler_output=first,
        boundary="target_forward",
        cudagraph_stats=SimpleNamespace(mode="FULL"),
    ):
        pass
    with ctl.nvtx_range_for_boundary(
        scheduler_output=first,
        boundary="draft_forward",
        cudagraph_stats=SimpleNamespace(mode="FULL"),
        draft_metadata={"has_drafter_config": True},
    ):
        pass
    first_snap = ctl.maybe_snapshot_device(
        metadata=ctl.make_metadata(
            scheduler_output=first,
            boundary="step_complete",
            cudagraph_stats=SimpleNamespace(mode="FULL"),
            draft_metadata={"has_drafter_config": True},
        )
    )

    second = sched(total=1, cached=["d"])
    with ctl.nvtx_range_for_boundary(
        scheduler_output=second,
        boundary="target_forward",
        cudagraph_stats=SimpleNamespace(mode="FULL"),
    ):
        pass
    with ctl.nvtx_range_for_boundary(
        scheduler_output=second,
        boundary="draft_forward",
        cudagraph_stats=SimpleNamespace(mode="FULL"),
        draft_metadata={"has_drafter_config": True},
    ):
        pass
    second_snap = ctl.maybe_snapshot_device(
        metadata=ctl.make_metadata(
            scheduler_output=second,
            boundary="step_complete",
            cudagraph_stats=SimpleNamespace(mode="FULL"),
            draft_metadata={"has_drafter_config": True},
        )
    )

    assert first_snap is not None
    assert second_snap is not None
    assert [first_snap.metadata["engine_step"], second_snap.metadata["engine_step"]] == [1, 2]
    assert all(":step:1:" in trace_id for trace_id in [pushes[0], pushes[1], first_snap.trace_id])
    assert all(":step:2:" in trace_id for trace_id in [pushes[3], pushes[4], second_snap.trace_id])
    assert [c["metadata"]["window_endpoint"] for c in calls] == ["start", "end"]
    assert ctl.local_step == 2


def test_noncomplete_fallback_boundaries_do_not_pair_with_later_zero_work_or_deferred_steps(monkeypatch, tmp_path):
    inst = _load_instrumentation()
    pushes = []

    class Nvtx:
        @staticmethod
        def range_push(trace_id):
            pushes.append(trace_id)

        @staticmethod
        def range_pop():
            pass

    class FakeTorch:
        cuda = SimpleNamespace(nvtx=Nvtx)

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(sys.modules, "slot_cache_hook", None)
    ctl = inst.SlotCacheWindowController(
        enabled=True,
        snapshot_dir=tmp_path,
        windows=(inst.SlotCacheWindow(1, 2), inst.SlotCacheWindow(4, 5)),
        run_id="boundaries",
    )

    aborted = sched(total=1, cached=["d"])
    with ctl.nvtx_range_for_boundary(scheduler_output=aborted, boundary="target_forward"):
        pass
    with ctl.nvtx_range_for_boundary(scheduler_output=aborted, boundary="abort_after_execute"):
        pass
    assert all(":step:1:" in trace_id for trace_id in pushes[:2])

    zero = sched(total=0)
    ctl.maybe_snapshot_device(metadata=ctl.make_metadata(scheduler_output=zero, boundary="step_complete"))
    rows = [json.loads(line) for line in ctl.reserve_output_path().read_text().splitlines()]
    assert rows[-1]["metadata"]["reason"] == "zero_work"
    assert rows[-1]["metadata"]["engine_step"] == 2

    deferred = sched(total=1, cached=["d"])
    with ctl.nvtx_range_for_boundary(scheduler_output=deferred, boundary="target_forward"):
        pass
    with ctl.nvtx_range_for_boundary(scheduler_output=deferred, boundary="deferred_after_execute"):
        pass
    assert all(":step:3:" in trace_id for trace_id in pushes[2:4])

    explicit = sched(total=1, cached=["d"], engine_step=4)
    with ctl.nvtx_range_for_boundary(scheduler_output=explicit, boundary="target_forward"):
        pass
    ctl.maybe_snapshot_device(metadata=ctl.make_metadata(scheduler_output=explicit, boundary="step_complete"))
    rows = [json.loads(line) for line in ctl.reserve_output_path().read_text().splitlines()]
    assert rows[-1]["metadata"]["reason"].startswith("import_error:")
    assert rows[-1]["metadata"]["engine_step"] == 4
    assert rows[-1]["metadata"]["valid_for_campaign"] is False
    assert ":step:4:" in pushes[-1]
    assert ctl.local_step == 4


def test_runner_local_step_exact_endpoints_and_declared_engine_step_consistency(monkeypatch):
    inst = _load_instrumentation()
    calls = []
    class Hook:
        @staticmethod
        def slot_cache_snapshot_device(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                trace_id=kwargs["trace_id"],
                layer_ids=[0],
                counters_device=[[0, 1, 1]],
                provenance=kwargs["provenance"],
                metadata=kwargs["metadata"],
            )
    monkeypatch.setitem(sys.modules, "slot_cache_hook", Hook)
    ctl = inst.SlotCacheWindowController(
        enabled=True,
        snapshot_dir=Path("/tmp"),
        windows=(inst.SlotCacheWindow(1, 2),),
        run_id="r",
    )
    for _ in range(2):
        meta = ctl.make_metadata(scheduler_output=sched(total=1, cached=["d"]), boundary="step_complete")
        snap = ctl.maybe_snapshot_device(metadata=meta)
        assert snap is not None
    assert [c["metadata"]["window_endpoint"] for c in calls] == ["start", "end"]
    assert [c["metadata"]["engine_step"] for c in calls] == [1, 2]
    assert all(c["metadata"]["scheduler_current_step_missing"] is True for c in calls)
    assert all(f":step:{c['metadata']['engine_step']}:" in c["trace_id"] for c in calls)


def test_endpoint_receipts_for_notready_zero_work_jumps_end_without_start_and_shutdown(tmp_path, monkeypatch):
    inst = _load_instrumentation()
    class NotReadyHook:
        @staticmethod
        def slot_cache_snapshot_device(**kwargs):
            return None
    monkeypatch.setitem(sys.modules, "slot_cache_hook", NotReadyHook)
    ctl = inst.SlotCacheWindowController(enabled=True, snapshot_dir=tmp_path, windows=(inst.SlotCacheWindow(1, 2),), run_id="r")
    snap = ctl.maybe_snapshot_device(metadata=ctl.make_metadata(scheduler_output=sched(total=1, cached=["d"]), boundary="step_complete"))
    assert snap is None
    rows = [json.loads(line) for line in ctl.reserve_output_path().read_text().splitlines()]
    assert rows[-1]["schema"] == "slot-cache-quiescent-receipt-v1"
    assert rows[-1]["metadata"]["status"] == "skipped"
    assert rows[-1]["metadata"]["reason"] == "all75_not_ready"

    ctl2 = inst.SlotCacheWindowController(enabled=True, snapshot_dir=tmp_path, windows=(inst.SlotCacheWindow(1, 2),), run_id="z")
    ctl2.maybe_snapshot_device(metadata=ctl2.make_metadata(scheduler_output=sched(total=0), boundary="step_complete"))
    zrows = [json.loads(line) for line in ctl2.reserve_output_path().read_text().splitlines()]
    assert zrows[-1]["metadata"]["reason"] == "zero_work"

    ctl3 = inst.SlotCacheWindowController(enabled=True, snapshot_dir=tmp_path, windows=(inst.SlotCacheWindow(2, 4),), run_id="jump")
    meta = ctl3.make_metadata(scheduler_output=SimpleNamespace(engine_step=5, total_num_scheduled_tokens=1, num_scheduled_tokens={"d":1}, scheduled_cached_reqs=CachedReqs(["d"]), scheduled_new_reqs=[], scheduled_spec_decode_tokens={}), boundary="step_complete")
    assert ctl3.maybe_snapshot_device(metadata=meta) is None
    jrows = [json.loads(line) for line in ctl3.reserve_output_path().read_text().splitlines()]
    assert [r["metadata"]["reason"] for r in jrows] == ["missed_endpoint", "missed_endpoint"]

    ctl4 = inst.SlotCacheWindowController(enabled=True, snapshot_dir=tmp_path, windows=(inst.SlotCacheWindow(1, 2),), run_id="endonly")
    end_meta = ctl4.make_metadata(scheduler_output=SimpleNamespace(engine_step=2, total_num_scheduled_tokens=1, num_scheduled_tokens={"d":1}, scheduled_cached_reqs=CachedReqs(["d"]), scheduled_new_reqs=[], scheduled_spec_decode_tokens={}), boundary="step_complete")
    assert ctl4.maybe_snapshot_device(metadata=end_meta) is None
    erows = [json.loads(line) for line in ctl4.reserve_output_path().read_text().splitlines()]
    assert erows[-1]["metadata"]["reason"] == "end_without_start"

    ctl5 = inst.SlotCacheWindowController(enabled=True, snapshot_dir=tmp_path, windows=(inst.SlotCacheWindow(10, 12),), run_id="shutdown")
    inst.finalize_slot_cache_windows(ctl5, reason="shutdown")
    srows = [json.loads(line) for line in ctl5.reserve_output_path().read_text().splitlines()]
    assert len(srows) == 2
    assert all(r["metadata"]["status"] == "skipped" for r in srows)
    assert all(r["metadata"]["reason"] == "shutdown" for r in srows)


def test_phase_classifier_filters_stale_spec_requests_against_active_scheduled_ids():
    inst = _load_instrumentation()
    phase = inst.classify_slot_cache_phase(
        sched(total=1, cached=["decode"], spec={"stale": [1, 2]}, num_tokens={"decode": 1})
    )
    assert phase["has_spec_verify"] is False
    assert phase["has_decode"] is True
    assert phase["canonical_phase"] == "decode"


def test_counter_scope_default_is_not_campaign_valid_without_external_canary_proof(monkeypatch):
    inst = _load_instrumentation()
    ctl = inst.SlotCacheWindowController(enabled=True, counter_scope="target_slot_cache")
    meta = ctl.make_metadata(scheduler_output=sched(total=1, cached=["d"]), boundary="step_complete")
    assert meta["counter_scope"] == "target_slot_cache"
    assert meta["valid_for_campaign"] is False
    monkeypatch.setenv("SLOT_CACHE_COUNTER_SCOPE", "target_slot_cache")
    monkeypatch.setenv("SLOT_CACHE_QUIESCENT_SNAPSHOTS", "1")
    d = Path(os.environ.get("PYTEST_TMPDIR", "/tmp"))
    monkeypatch.setenv("SLOT_CACHE_SNAPSHOT_DIR", str(d))
    ctl2 = inst.SlotCacheWindowController.from_env()
    assert ctl2.make_metadata(scheduler_output=sched(total=1, cached=["d"]), boundary="step_complete")["valid_for_campaign"] is False
