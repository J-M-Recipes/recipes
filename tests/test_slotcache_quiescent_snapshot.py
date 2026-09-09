import ast
import json
import os
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PATCHES = REPO_ROOT / "recipes/dgx-station-gb300/glm-5.3-nvfp4-uva-slot-cache/patches"


PROVENANCE_1 = {
    "run_id": "run-20260908T010203Z-a1b2c3d4",
    "source_sha": "0123456789abcdef0123456789abcdef01234567",
    "engine_generation": 7,
    "seq": 1,
}
PROVENANCE_2 = dict(PROVENANCE_1, seq=2)


def _stats_module():
    sys.path.insert(0, str(PATCHES))
    try:
        sys.modules.pop("slot_cache_stats", None)
        import slot_cache_stats
        return slot_cache_stats
    finally:
        sys.path.pop(0)


def _snapshot(*, provenance=PROVENANCE_1, layers=None):
    return {
        "schema": "slot-cache-quiescent-snapshot-v1",
        "provenance": dict(provenance),
        "layers": layers
        if layers is not None
        else {
            "3": {"misses": 12, "routes": 80, "steps": 10},
            "4": {"misses": 8, "routes": 72, "steps": 10},
        },
    }


def test_quiescent_snapshot_validation_requires_canonical_explicit_provenance():
    stats = _stats_module()

    assert stats.validate_counter_snapshot(_snapshot(), expected_layers=[3, 4]) == _snapshot()

    bad_cases = [
        ({k: v for k, v in PROVENANCE_1.items() if k != "run_id"}, "run_id"),
        (dict(PROVENANCE_1, run_id=""), "run_id"),
        (dict(PROVENANCE_1, source_sha="not-a-sha"), "source_sha"),
        (dict(PROVENANCE_1, source_sha=PROVENANCE_1["source_sha"].upper()), "source_sha"),
        (dict(PROVENANCE_1, engine_generation=True), "engine_generation"),
        (dict(PROVENANCE_1, engine_generation=-1), "engine_generation"),
        (dict(PROVENANCE_1, seq=0), "seq"),
        (dict(PROVENANCE_1, seq="2"), "seq"),
        (dict(PROVENANCE_1, source="engine-safe-point"), "provenance"),
    ]
    for provenance, message in bad_cases:
        with pytest.raises(ValueError, match=message):
            stats.validate_counter_snapshot(_snapshot(provenance=provenance), expected_layers=[3, 4])


def test_quiescent_snapshot_validation_rejects_noninteger_impossible_or_unaligned_counters():
    stats = _stats_module()

    def snapshot(layer3):
        return _snapshot(
            layers={
                "3": layer3,
                "4": {"misses": 1, "routes": 8, "steps": 2},
            }
        )

    with pytest.raises(ValueError, match="integer"):
        stats.validate_counter_snapshot(snapshot({"misses": 1.5, "routes": 8, "steps": 2}), expected_layers=[3, 4])
    with pytest.raises(ValueError, match="misses cannot exceed routes"):
        stats.validate_counter_snapshot(snapshot({"misses": 9, "routes": 8, "steps": 2}), expected_layers=[3, 4])
    with pytest.raises(ValueError, match="aligned steps"):
        stats.validate_counter_snapshot(snapshot({"misses": 1, "routes": 8, "steps": 3}), expected_layers=[3, 4])


def test_expected_layers_are_materialized_once_and_must_be_nonempty_unique_canonical():
    stats = _stats_module()

    yielded = []

    def layer_generator():
        for layer in (3, 4):
            yielded.append(layer)
            yield layer

    assert stats.validate_counter_snapshot(_snapshot(), expected_layers=layer_generator())["layers"] == _snapshot()["layers"]
    assert yielded == [3, 4]

    bad_expected = [
        ([], "expected_layers must not be empty"),
        ([3, 3], "duplicate"),
        (["03", 4], "canonical"),
        (["٣", 4], "canonical"),
        (["３", 4], "canonical"),
        ([-1, 4], "canonical"),
        ([True, 4], "canonical"),
    ]
    for expected_layers, message in bad_expected:
        with pytest.raises(ValueError, match=message):
            stats.validate_counter_snapshot(_snapshot(), expected_layers=expected_layers)


def test_delta_materializes_generator_across_both_snapshots():
    stats = _stats_module()
    delta = stats.counter_snapshot_delta(
        _snapshot(), _snapshot(provenance=PROVENANCE_2),
        expected_layers=(layer for layer in (3, 4)),
    )
    assert delta['layers'] == {
        '3': {'misses': 0, 'routes': 0, 'steps': 0},
        '4': {'misses': 0, 'routes': 0, 'steps': 0},
    }


def test_quiescent_snapshot_delta_requires_same_run_source_generation_and_increasing_seq():
    stats = _stats_module()
    start = _snapshot(provenance=PROVENANCE_1)
    end = _snapshot(
        provenance=PROVENANCE_2,
        layers={
            "3": {"misses": 14, "routes": 132, "steps": 14},
            "4": {"misses": 21, "routes": 136, "steps": 14},
        },
    )

    delta = stats.counter_snapshot_delta(start, end, expected_layers=[3, 4])

    assert delta["provenance"] == {"start": PROVENANCE_1, "end": PROVENANCE_2}
    assert delta["layers"] == {
        "3": {"misses": 2, "routes": 52, "steps": 4},
        "4": {"misses": 13, "routes": 64, "steps": 4},
    }

    for changed in (
        dict(PROVENANCE_2, run_id="run-20260908T010203Z-deadbeef"),
        dict(PROVENANCE_2, source_sha="abcdef0123456789abcdef0123456789abcdef01"),
        dict(PROVENANCE_2, engine_generation=8),
        dict(PROVENANCE_2, seq=1),
    ):
        with pytest.raises(ValueError, match="provenance|seq"):
            stats.counter_snapshot_delta(start, dict(end, provenance=changed), expected_layers=[3, 4])


def test_summarize_snapshot_delta_recomputes_from_valid_raw_and_rejects_tampering():
    stats = _stats_module()
    start = _snapshot(provenance=PROVENANCE_1)
    end = _snapshot(
        provenance=PROVENANCE_2,
        layers={
            "3": {"misses": 14, "routes": 132, "steps": 14},
            "4": {"misses": 21, "routes": 136, "steps": 14},
        },
    )
    delta = stats.counter_snapshot_delta(start, end, expected_layers=[3, 4])

    malicious_negative_hit = dict(
        delta,
        layers={
            "3": {"misses": 100, "routes": 1, "steps": 1},
            "4": {"misses": 100, "routes": 1, "steps": 1},
        },
    )
    with pytest.raises(ValueError, match="delta layers do not match raw snapshots"):
        stats.summarize_snapshot_delta(malicious_negative_hit)

    malicious_negative_counter = dict(delta, layers={"3": {"misses": -100, "routes": 1, "steps": 1}})
    with pytest.raises(ValueError, match="delta layers do not match raw snapshots|integer|expected layers"):
        stats.summarize_snapshot_delta(malicious_negative_counter)

    impossible_raw = dict(
        delta,
        raw={
            "start": start,
            "end": dict(end, layers={"3": {"misses": 200, "routes": 132, "steps": 14}, "4": end["layers"]["4"]}),
        },
    )
    with pytest.raises(ValueError, match="misses cannot exceed routes"):
        stats.summarize_snapshot_delta(impossible_raw)

    unmatched_raw = dict(delta, raw={"start": start, "end": dict(end, provenance=dict(PROVENANCE_2, seq=3))})
    with pytest.raises(ValueError, match="delta provenance does not match raw snapshots|delta layers do not match raw snapshots"):
        stats.summarize_snapshot_delta(unmatched_raw)

    assert stats.summarize_snapshot_delta(delta) == {
        "hit_rate": 1.0 - (15 / 116),
        "misses_per_step": 15 / 8,
        "routes_per_step": 116 / 8,
        "layers": {
            "3": {"hit_rate": 1.0 - (2 / 52), "misses_per_step": 0.5, "routes_per_step": 13.0},
            "4": {"hit_rate": 1.0 - (13 / 64), "misses_per_step": 3.25, "routes_per_step": 16.0},
        },
    }


def _load_start_stats_thread_function():
    tree = ast.parse((PATCHES / "slot_cache_hook.py").read_text())
    needed_names = {"_start_stats_thread"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in needed_names]
    assert len(nodes) == 1
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "_stats_thread": None,
        "_STATS_MODE": "legacy",
        "_QUIESCENT_TELEMETRY_BLOCKED": "quiescent snapshot telemetry requires an engine-owned safe point",
        "_LOG": lambda message: namespace["logs"].append(message),
        "_registry": {},
        "_ring_dump": lambda: namespace["events"].append("ring_dump"),
        "summarize_window": lambda **kwargs: None,
        "os": os,
        "logs": [],
        "events": [],
    }
    exec(compile(module, filename="slot_cache_hook.py", mode="exec"), namespace)
    return namespace


def test_quiescent_mode_returns_before_thread_start_or_cuda_counter_access(monkeypatch):
    namespace = _load_start_stats_thread_function()
    monkeypatch.setenv("SLOT_CACHE_STATS_SEC", "20")
    namespace["_STATS_MODE"] = "quiescent"

    namespace["_start_stats_thread"]()

    assert namespace["_stats_thread"] is None
    assert namespace["events"] == []
    assert namespace["logs"] == ["quiescent snapshot telemetry requires an engine-owned safe point"]


def test_sitecustomize_registers_slot_cache_hook_as_import_identity_for_snapshots(tmp_path, monkeypatch):
    monkeypatch.setenv("SLOT_CACHE", "1")
    monkeypatch.setenv("SLOT_CACHE_HOOK", str(PATCHES / "slot_cache_hook.py"))
    monkeypatch.delenv("EXACT_PIN", raising=False)
    monkeypatch.delenv("ROUTE_TRACE_DIR", raising=False)
    monkeypatch.delenv("VLLM_AUTOTUNE_CACHE_KEY", raising=False)
    monkeypatch.setenv("SLOT_CACHE_STATS_SEC", "0")
    monkeypatch.setenv("SLOT_CACHE_QUIESCENT_SNAPSHOTS", "1")
    monkeypatch.setenv("SLOT_CACHE_SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.setenv("SLOT_CACHE_WINDOW_STEPS", "100:164")
    monkeypatch.setenv("SLOT_CACHE_SOURCE_SHA", "a" * 64)
    monkeypatch.setenv("SLOT_CACHE_RUN_ID", "run-k1-offline")
    monkeypatch.setenv("SLOT_CACHE_K_MODE", "K1")
    monkeypatch.setenv("SLOT_CACHE_EXPECTED_LAYERS", "75")
    old_meta_path = list(sys.meta_path)
    old_path = list(sys.path)
    sys.modules.pop("slot_cache_hook", None)
    sys.modules.pop("slot_cache_window_instrumentation", None)
    try:
        sitecustomize_globals = runpy.run_path(str(PATCHES / "sitecustomize.py"))
        producer_hook = sitecustomize_globals["_m"]
        assert sys.modules["slot_cache_hook"] is producer_hook

        class Scalar:
            device = "cuda:0"
            def __init__(self, value):
                self.value = value
            def clone_value(self):
                return self.value

        producer_hook._registry = {
            layer: SimpleNamespace(
                name=f"model.layers.{layer}.mlp.experts",
                misses=Scalar(layer),
                routes=Scalar(layer + 100),
                step=Scalar(7),
            )
            for layer in range(3, 78)
        }
        producer_hook.torch = None

        sys.path.insert(0, str(PATCHES))
        try:
            import slot_cache_window_instrumentation as inst
        finally:
            sys.path.pop(0)

        controller = inst.SlotCacheWindowController.from_env()
        snapshot = controller.maybe_snapshot_device(
            metadata={
                "engine_step": 100,
                "boundary": "step_complete",
                "phase_flags": {"canonical_phase": "decode", "has_decode": True},
                "valid_for_campaign": False,
                "campaign_validity_blocker": "external_canary_not_proven",
            }
        )

        assert snapshot is not None
        assert snapshot.layer_ids == list(range(3, 78))
        controller.finalize_slot_cache_snapshot(snapshot)
        rows = [json.loads(line) for line in controller.reserve_output_path().read_text().splitlines()]
        assert rows[0]["schema"] == "slot-cache-quiescent-snapshot-v1"
        assert rows[0]["metadata"]["valid_for_campaign"] is False
        assert rows[0]["layers"]["77"] == {"misses": 77, "routes": 177, "steps": 7}
    finally:
        sys.meta_path[:] = old_meta_path
        sys.path[:] = old_path
        sys.modules.pop("slot_cache_hook", None)
        sys.modules.pop("slot_cache_window_instrumentation", None)


def _run_sitecustomize_with_temp_slot_hook(monkeypatch, hook_path):
    monkeypatch.setenv("SLOT_CACHE", "1")
    monkeypatch.setenv("SLOT_CACHE_HOOK", str(hook_path))
    monkeypatch.delenv("EXACT_PIN", raising=False)
    monkeypatch.delenv("ROUTE_TRACE_DIR", raising=False)
    monkeypatch.delenv("VLLM_AUTOTUNE_CACHE_KEY", raising=False)
    return runpy.run_path(str(PATCHES / "sitecustomize.py"))


def test_sitecustomize_exec_failure_restores_prior_slot_cache_hook_binding(tmp_path, monkeypatch):
    prior_hook = SimpleNamespace(marker="prior")
    hook_path = tmp_path / "slot_cache_hook.py"
    hook_path.write_text("PARTIAL = 'exec-started'\nraise RuntimeError('boom during exec')\n")
    old_meta_path = list(sys.meta_path)
    sys.modules["slot_cache_hook"] = prior_hook
    try:
        _run_sitecustomize_with_temp_slot_hook(monkeypatch, hook_path)
        assert sys.modules["slot_cache_hook"] is prior_hook
        assert not hasattr(sys.modules["slot_cache_hook"], "PARTIAL")
    finally:
        sys.meta_path[:] = old_meta_path
        if sys.modules.get("slot_cache_hook") is prior_hook:
            sys.modules.pop("slot_cache_hook", None)


def test_sitecustomize_install_failure_restores_prior_slot_cache_hook_binding(tmp_path, monkeypatch):
    prior_hook = SimpleNamespace(marker="prior")
    hook_path = tmp_path / "slot_cache_hook.py"
    hook_path.write_text("PARTIAL = 'install-started'\ndef install():\n    raise RuntimeError('boom during install')\n")
    old_meta_path = list(sys.meta_path)
    sys.modules["slot_cache_hook"] = prior_hook
    try:
        _run_sitecustomize_with_temp_slot_hook(monkeypatch, hook_path)
        assert sys.modules["slot_cache_hook"] is prior_hook
        assert not hasattr(sys.modules["slot_cache_hook"], "PARTIAL")
    finally:
        sys.meta_path[:] = old_meta_path
        if sys.modules.get("slot_cache_hook") is prior_hook:
            sys.modules.pop("slot_cache_hook", None)


def test_sitecustomize_repeated_execution_reuses_installed_slot_cache_hook_module(tmp_path, monkeypatch):
    hook_path = tmp_path / "slot_cache_hook.py"
    hook_path.write_text("_registry = {}\ndef install():\n    _registry.setdefault('install_count', 0)\n    _registry['install_count'] += 1\n")
    old_meta_path = list(sys.meta_path)
    sys.modules.pop("slot_cache_hook", None)
    try:
        first_globals = _run_sitecustomize_with_temp_slot_hook(monkeypatch, hook_path)
        first_hook = first_globals["_m"]
        first_registry = first_hook._registry
        first_registry["sentinel"] = object()

        second_globals = _run_sitecustomize_with_temp_slot_hook(monkeypatch, hook_path)

        assert second_globals["_m"] is first_hook
        assert sys.modules["slot_cache_hook"] is first_hook
        assert first_hook._registry is first_registry
        assert first_hook._registry["install_count"] == 1
    finally:
        sys.meta_path[:] = old_meta_path
        sys.modules.pop("slot_cache_hook", None)


def test_unknown_stats_mode_raises_before_thread_start_or_cuda_counter_access(monkeypatch):
    namespace = _load_start_stats_thread_function()
    monkeypatch.setenv("SLOT_CACHE_STATS_SEC", "20")
    namespace["_STATS_MODE"] = "mystery"

    with pytest.raises(ValueError, match="unknown SLOT_CACHE_STATS_MODE"):
        namespace["_start_stats_thread"]()

    assert namespace["_stats_thread"] is None
    assert namespace["events"] == []
