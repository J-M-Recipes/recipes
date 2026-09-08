"""Opt-in engine-owned slot-cache quiescent window instrumentation.

This module is deliberately sidecar-only. Baseline serving behavior is unchanged
unless SLOT_CACHE_QUIESCENT_SNAPSHOTS=1 and a pre-existing snapshot directory is
provided. Counter D2H is performed only at configured window endpoints. The sync
endpoint path is intentionally outside the measured window and may add endpoint
overhead; it never polls CUDA buffers in a background thread and never synchronizes
per step.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any

SNAPSHOT_SCHEMA = "slot-cache-quiescent-snapshot-v1"
RECEIPT_SCHEMA = "slot-cache-quiescent-receipt-v1"
SOURCE_ARCHIVE_SHA256 = "73042b4db4e106d4a1caab38b48da0f2157dce588c0be4df2de9598a3ecd0bd5"
IMAGE_SHA = "sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14"
_GPU_MODEL_RUNNER_SOURCE_SHA256 = "7f2890eefca1efe25565bf1c7e5906a87948ae922610a7aaac620b28b46f26aa"
_SHA_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")


@dataclass(frozen=True)
class SlotCacheWindow:
    start: int
    end: int


class SlotCacheWindowController:
    def __init__(
        self,
        *,
        enabled: bool,
        snapshot_dir: Path | None = None,
        windows: tuple[SlotCacheWindow, ...] = (),
        run_id: str = "",
        source_sha: str = SOURCE_ARCHIVE_SHA256,
        engine_generation: int = 1,
        expected_layers: int = 75,
        k_mode: str = "unknown",
        recipe_sha: str = "unknown",
        image_sha: str = IMAGE_SHA,
        counter_scope: str = "target_slot_cache",
        target_forward_snapshots: bool = False,
    ):
        self.enabled = enabled
        self.snapshot_dir = snapshot_dir
        self.windows = windows
        self.run_id = run_id
        self.source_sha = source_sha
        self.engine_generation = engine_generation
        self.expected_layers = expected_layers
        self.k_mode = k_mode
        self.recipe_sha = recipe_sha
        self.image_sha = image_sha
        self.counter_scope = counter_scope
        self.target_forward_snapshots = target_forward_snapshots
        self.local_step = 0
        self.seq = 0
        self._output_path: Path | None = None
        self._sidecar_created = False
        self._endpoint_status: dict[tuple[int, str], str] = {}
        self._pending_fallback_steps: dict[int, int] = {}
        self.instrumentation_failures: list[dict[str, str]] = []

    @classmethod
    def disabled(cls) -> "SlotCacheWindowController":
        return cls(enabled=False)

    @classmethod
    def from_env(cls) -> "SlotCacheWindowController":
        if os.environ.get("SLOT_CACHE_QUIESCENT_SNAPSHOTS", "0") != "1":
            return cls.disabled()
        snapshot_dir_s = os.environ.get("SLOT_CACHE_SNAPSHOT_DIR")
        if not snapshot_dir_s:
            raise ValueError("SLOT_CACHE_SNAPSHOT_DIR is required when quiescent snapshots are enabled")
        snapshot_dir = Path(snapshot_dir_s).expanduser().resolve(strict=False)
        if not snapshot_dir.exists() or not snapshot_dir.is_dir():
            raise ValueError("SLOT_CACHE_SNAPSHOT_DIR must already exist and be a directory")
        expected_layers = _positive_int_env("SLOT_CACHE_EXPECTED_LAYERS", 75)
        windows = _parse_windows(os.environ.get("SLOT_CACHE_WINDOW_STEPS", ""))
        run_id = os.environ.get("SLOT_CACHE_RUN_ID", "slot-cache-local")
        source_sha = os.environ.get("SLOT_CACHE_SOURCE_SHA", SOURCE_ARCHIVE_SHA256)
        if _SHA_RE.fullmatch(source_sha) is None:
            raise ValueError("SLOT_CACHE_SOURCE_SHA must be a canonical lowercase SHA")
        generation = _positive_int_env("SLOT_CACHE_ENGINE_GENERATION", 1)
        return cls(
            enabled=True,
            snapshot_dir=snapshot_dir,
            windows=windows,
            run_id=run_id,
            source_sha=source_sha,
            engine_generation=generation,
            expected_layers=expected_layers,
            k_mode=os.environ.get("SLOT_CACHE_K_MODE", "unknown"),
            recipe_sha=os.environ.get("SLOT_CACHE_RECIPE_SHA", "unknown"),
            image_sha=os.environ.get("SLOT_CACHE_IMAGE_SHA", IMAGE_SHA),
            counter_scope=os.environ.get("SLOT_CACHE_COUNTER_SCOPE", "target_slot_cache"),
            target_forward_snapshots=os.environ.get("SLOT_CACHE_TARGET_FORWARD_SNAPSHOTS", "0") == "1",
        )

    def reserve_output_path(self) -> Path:
        if self.snapshot_dir is None:
            raise ValueError("snapshot_dir is required")
        if self._output_path is None:
            path = (self.snapshot_dir / f"slot-cache-snapshots-{os.getpid()}-{id(self):x}.jsonl").resolve(strict=False)
            if path.parent != self.snapshot_dir:
                raise ValueError("snapshot output path escaped SLOT_CACHE_SNAPSHOT_DIR")
            if path.exists():
                raise FileExistsError(f"refusing to overwrite existing snapshot sidecar: {path}")
            self._output_path = path
        if self._output_path.exists() and not self._sidecar_created:
            raise FileExistsError(f"refusing to overwrite existing snapshot sidecar: {self._output_path}")
        return self._output_path

    def should_snapshot(self, engine_step: int) -> str | None:
        for window in self.windows:
            if engine_step == window.start:
                return "start"
            if engine_step == window.end:
                return "end"
        return None

    def make_metadata(self, *, scheduler_output: Any, boundary: str, cudagraph_stats: Any = None, draft_metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        phase_flags = classify_slot_cache_phase(scheduler_output)
        engine_step = self._metadata_engine_step(scheduler_output, boundary)
        graph_mode = _graph_mode(cudagraph_stats)
        draft_metadata = dict(draft_metadata or {})
        metadata = {
            "engine_step": engine_step,
            "scheduler_current_step_missing": _scheduler_engine_step(scheduler_output) is None,
            "boundary": boundary,
            "phase_flags": phase_flags,
            "graph_mode": graph_mode,
            "expected_layers": self.expected_layers,
            "image_sha": self.image_sha,
            "recipe_sha": self.recipe_sha,
            "k_mode": self.k_mode,
            "counter_scope": self.counter_scope,
            "valid_for_campaign": False,
            "campaign_validity_blocker": "external_canary_not_proven",
        }
        metadata.update(draft_metadata)
        return metadata

    def _record_instrumentation_failure(self, operation: str, exc: BaseException) -> None:
        self.instrumentation_failures.append({
            "operation": operation,
            "error_type": type(exc).__name__,
            "message": str(exc),
        })

    @contextmanager
    def nvtx_range(self, trace_id: str | None):
        if not self.enabled or not trace_id:
            yield
            return
        try:
            import torch  # type: ignore
        except Exception as exc:
            self._record_instrumentation_failure("nvtx_import", exc)
            yield
            return
        try:
            torch.cuda.nvtx.range_push(trace_id)
        except Exception as exc:
            self._record_instrumentation_failure("nvtx_range_push", exc)
            yield
            return
        body_exc: BaseException | None = None
        try:
            yield
        except BaseException as exc:
            body_exc = exc
            raise
        finally:
            try:
                torch.cuda.nvtx.range_pop()
            except Exception as exc:
                self._record_instrumentation_failure("nvtx_range_pop", exc)

    def nvtx_range_for_boundary(self, *, scheduler_output: Any, boundary: str, cudagraph_stats: Any = None, draft_metadata: Mapping[str, Any] | None = None):
        if not self.enabled:
            return nullcontext()
        metadata = self.make_metadata(
            scheduler_output=scheduler_output,
            boundary=boundary,
            cudagraph_stats=cudagraph_stats,
            draft_metadata=draft_metadata,
        )
        trace_id = make_slot_cache_trace_id(
            run_id=self.run_id,
            engine_generation=self.engine_generation,
            seq=self.seq + 1,
            engine_step=int(metadata.get("engine_step", self.local_step + 1)),
            boundary=boundary,
            phase_flags=metadata.get("phase_flags", {}),
            graph_mode=str(metadata.get("graph_mode", "unknown")),
            k_mode=self.k_mode,
            has_drafter_config=bool(metadata.get("has_drafter_config", False)),
            drafter_runs_model_forward=bool(metadata.get("drafter_runs_model_forward", False)),
        )
        return self.nvtx_range(trace_id)

    def mark_target_forward_boundary(self, *, scheduler_output: Any, cudagraph_stats: Any = None) -> None:
        # Optional marker only; default disabled so target-forward telemetry is not
        # mixed with the full-step campaign window.
        if not (self.enabled and self.target_forward_snapshots):
            return
        metadata = self.make_metadata(scheduler_output=scheduler_output, boundary="target_forward_done", cudagraph_stats=cudagraph_stats)
        self.maybe_snapshot_device(metadata=metadata)

    def mark_failed_boundary(self, *, scheduler_output: Any, boundary: str, reason: str, cudagraph_stats: Any = None) -> None:
        if not self.enabled:
            return
        metadata = self.make_metadata(
            scheduler_output=scheduler_output,
            boundary=boundary,
            cudagraph_stats=cudagraph_stats,
        )
        engine_step = int(metadata.get("engine_step", self.local_step))
        marker = make_slot_cache_trace_id(
            run_id=self.run_id,
            engine_generation=self.engine_generation,
            seq=self.seq + 1,
            engine_step=engine_step,
            boundary=boundary,
            phase_flags=metadata.get("phase_flags", {}),
            graph_mode=str(metadata.get("graph_mode", "unknown")),
            k_mode=self.k_mode,
        )
        with self.nvtx_range(marker):
            pass
        endpoint = self.should_snapshot(engine_step)
        if endpoint is not None:
            self._emit_endpoint_receipt(engine_step, endpoint, metadata, status="failure", reason=reason)

    def maybe_snapshot_device(self, *, metadata: Mapping[str, Any]) -> Any | None:
        if not self.enabled:
            return None
        full_metadata = dict(metadata)
        boundary = str(full_metadata.get("boundary", "step_complete"))
        engine_step = int(full_metadata.get("engine_step", self.local_step + 1))
        if boundary == "step_complete":
            self.local_step = max(self.local_step, engine_step)
        self._emit_missed_before(engine_step)
        endpoint = self.should_snapshot(engine_step)
        if endpoint is None:
            return None
        phase_flags = dict(full_metadata.get("phase_flags", {}) or {})
        if phase_flags.get("zero_work"):
            self._emit_endpoint_receipt(engine_step, endpoint, full_metadata, status="skipped", reason="zero_work")
            return None
        if endpoint == "end" and self._endpoint_status.get((self._window_start_for_end(engine_step), "start")) != "snapshot":
            self._emit_endpoint_receipt(engine_step, endpoint, full_metadata, status="failure", reason="end_without_start")
            return None
        self.seq += 1
        provenance = self._provenance()
        marker = make_slot_cache_trace_id(
            run_id=self.run_id,
            engine_generation=self.engine_generation,
            seq=self.seq,
            engine_step=engine_step,
            boundary=boundary,
            phase_flags=phase_flags,
            graph_mode=str(full_metadata.get("graph_mode", "unknown")),
            k_mode=self.k_mode,
            has_drafter_config=bool(full_metadata.get("has_drafter_config", False)),
            drafter_runs_model_forward=bool(full_metadata.get("drafter_runs_model_forward", False)),
        )
        full_metadata.update({"engine_step": engine_step, "window_endpoint": endpoint, "trace_id": marker})
        try:
            import slot_cache_hook  # type: ignore
        except Exception as exc:
            self._emit_endpoint_receipt(engine_step, endpoint, full_metadata, status="failure", reason=f"import_error:{type(exc).__name__}")
            return None
        with self.nvtx_range(marker):
            snapshot = slot_cache_hook.slot_cache_snapshot_device(
                trace_id=marker,
                provenance=provenance,
                metadata=full_metadata,
                expected_layers=self.expected_layers,
            )
        if snapshot is None:
            self._emit_endpoint_receipt(engine_step, endpoint, full_metadata, status="skipped", reason="all75_not_ready")
            return None
        self._endpoint_status[(engine_step, endpoint)] = "snapshot"
        return snapshot

    def _metadata_engine_step(self, scheduler_output: Any, boundary: str) -> int:
        explicit_step = _scheduler_engine_step(scheduler_output)
        scheduler_id = id(scheduler_output)
        if explicit_step is not None:
            self._pending_fallback_steps.pop(scheduler_id, None)
            if boundary == "step_complete":
                self.local_step = max(self.local_step, explicit_step)
            return explicit_step
        if scheduler_id in self._pending_fallback_steps:
            engine_step = self._pending_fallback_steps[scheduler_id]
        else:
            engine_step = self.local_step + 1
            self._pending_fallback_steps[scheduler_id] = engine_step
        if boundary in {"step_complete", "abort_after_execute", "deferred_after_execute", "target_forward_exception", "sample_exception"}:
            self.local_step = max(self.local_step, engine_step)
            self._pending_fallback_steps.pop(scheduler_id, None)
        return engine_step

    def _provenance(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "source_sha": self.source_sha,
            "engine_generation": self.engine_generation,
            "seq": self.seq,
        }

    def _window_start_for_end(self, end_step: int) -> int:
        for window in self.windows:
            if window.end == end_step:
                return window.start
        return end_step

    def _emit_missed_before(self, engine_step: int) -> None:
        for window in self.windows:
            for step, endpoint in ((window.start, "start"), (window.end, "end")):
                if step < engine_step and (step, endpoint) not in self._endpoint_status:
                    meta = self.make_receipt_metadata(step, endpoint, {}, status="skipped", reason="missed_endpoint")
                    self._write_json_row(receipt_to_json_row(provenance=self._provenance(), metadata=meta))
                    self._endpoint_status[(step, endpoint)] = "skipped"

    def _emit_endpoint_receipt(self, engine_step: int, endpoint: str, metadata: Mapping[str, Any], *, status: str, reason: str) -> None:
        self._endpoint_status[(engine_step, endpoint)] = status
        meta = self.make_receipt_metadata(engine_step, endpoint, metadata, status=status, reason=reason)
        self._write_json_row(receipt_to_json_row(provenance=self._provenance(), metadata=meta))

    def make_receipt_metadata(self, engine_step: int, endpoint: str, metadata: Mapping[str, Any], *, status: str, reason: str) -> dict[str, Any]:
        meta = dict(metadata)
        meta.update({
            "engine_step": engine_step,
            "window_endpoint": endpoint,
            "status": status,
            "reason": reason,
            "valid_for_campaign": False,
        })
        return meta

    def finalize_windows(self, *, reason: str = "shutdown") -> None:
        if not self.enabled:
            return
        for window in self.windows:
            for step, endpoint in ((window.start, "start"), (window.end, "end")):
                if (step, endpoint) not in self._endpoint_status:
                    meta = self.make_receipt_metadata(step, endpoint, {"boundary": "step_complete"}, status="skipped", reason=reason)
                    self._write_json_row(receipt_to_json_row(provenance=self._provenance(), metadata=meta))
                    self._endpoint_status[(step, endpoint)] = "skipped"

    def finalize_slot_cache_snapshot(self, snapshot: Any, counters_cpu: Any | None = None) -> None:
        if snapshot is None:
            return
        layers = _counters_to_layers(getattr(snapshot, "layer_ids"), counters_cpu if counters_cpu is not None else getattr(snapshot, "counters_device"))
        row = snapshot_to_json_row(
            trace_id=getattr(snapshot, "trace_id"),
            provenance=getattr(snapshot, "provenance"),
            metadata=getattr(snapshot, "metadata"),
            layers=layers,
        )
        self._write_json_row(row)

    def _write_json_row(self, row: Mapping[str, Any]) -> None:
        path = self.reserve_output_path()
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n")
        self._sidecar_created = True


def _positive_int_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _parse_windows(value: str) -> tuple[SlotCacheWindow, ...]:
    if not value:
        return ()
    windows = []
    for piece in value.split(","):
        start_s, sep, end_s = piece.partition(":")
        if not sep:
            raise ValueError("SLOT_CACHE_WINDOW_STEPS entries must be start:end")
        start, end = int(start_s), int(end_s)
        if start <= 0 or end <= start:
            raise ValueError("SLOT_CACHE_WINDOW_STEPS requires positive bounded start:end windows")
        windows.append(SlotCacheWindow(start, end))
    return tuple(windows)


def classify_slot_cache_phase(scheduler_output: Any) -> dict[str, Any]:
    num_tokens = getattr(scheduler_output, "num_scheduled_tokens", {}) or {}
    active_req_ids = set(num_tokens)
    scheduled_new = list(getattr(scheduler_output, "scheduled_new_reqs", []) or [])
    cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
    cached_req_ids = [rid for rid in getattr(cached, "req_ids", []) if rid in active_req_ids]
    raw_spec_tokens = getattr(scheduler_output, "scheduled_spec_decode_tokens", {}) or {}
    spec_tokens = {rid: toks for rid, toks in raw_spec_tokens.items() if rid in active_req_ids}
    spec_req_ids = set(spec_tokens)
    total = int(getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0)

    has_prefill = bool(scheduled_new)
    for req_id in cached_req_ids:
        try:
            if cached is not None and cached.is_context_phase(req_id):
                has_prefill = True
        except Exception:
            pass
    has_spec_verify = bool(spec_tokens)
    has_decode = False
    for req_id in cached_req_ids:
        is_context = False
        try:
            is_context = bool(cached is not None and cached.is_context_phase(req_id))
        except Exception:
            is_context = False
        if not is_context and req_id not in spec_req_ids:
            has_decode = True
    zero_work = total == 0
    flags = [has_prefill, has_decode, has_spec_verify]
    has_mixed = sum(1 for flag in flags if flag) > 1
    if zero_work:
        canonical = "zero_work"
    elif has_mixed:
        canonical = "mixed"
    elif has_prefill:
        canonical = "prefill"
    elif has_decode:
        canonical = "decode"
    elif has_spec_verify:
        canonical = "spec_verify"
    else:
        canonical = "zero_work"
    return {
        "zero_work": zero_work,
        "has_prefill": has_prefill,
        "has_decode": has_decode,
        "has_spec_verify": has_spec_verify,
        "has_mixed": has_mixed,
        "canonical_phase": canonical,
        "num_scheduled_tokens": total,
        "num_scheduled_reqs": len(num_tokens),
    }


def make_slot_cache_trace_id(*, run_id: str, engine_generation: int, seq: int, engine_step: int, boundary: str, phase_flags: Mapping[str, Any], graph_mode: str, k_mode: str, has_drafter_config: bool = False, drafter_runs_model_forward: bool = False) -> str:
    flags = []
    if phase_flags.get("has_prefill"):
        flags.append("prefill")
    if phase_flags.get("has_decode"):
        flags.append("decode")
    if phase_flags.get("has_spec_verify"):
        flags.append("spec_verify")
    if has_drafter_config:
        flags.append("drafter")
    if drafter_runs_model_forward:
        flags.append("drafter_forward")
    if not flags:
        flags.append("none")
    return (
        f"slotcache:{run_id}:gen:{engine_generation}:seq:{seq}:step:{engine_step}:"
        f"boundary:{boundary}:phase:{phase_flags.get('canonical_phase', 'unknown')}:"
        f"flags:{','.join(flags)}:graph:{graph_mode}:k:{k_mode}"
    )


def snapshot_to_json_row(*, trace_id: str, provenance: Mapping[str, Any], metadata: Mapping[str, Any], layers: Mapping[str, Any]) -> dict[str, Any]:
    meta = dict(metadata)
    meta["trace_id"] = trace_id
    return {
        "schema": SNAPSHOT_SCHEMA,
        "provenance": dict(provenance),
        "metadata": meta,
        "layers": dict(layers),
    }


def receipt_to_json_row(*, provenance: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": RECEIPT_SCHEMA,
        "provenance": dict(provenance),
        "metadata": dict(metadata),
    }


def finalize_slot_cache_snapshot(controller: SlotCacheWindowController | None, snapshot: Any, counters_cpu: Any | None = None) -> None:
    if controller is not None:
        controller.finalize_slot_cache_snapshot(snapshot, counters_cpu)


def finalize_slot_cache_windows(controller: SlotCacheWindowController | None, reason: str = "shutdown") -> None:
    if controller is not None:
        controller.finalize_windows(reason=reason)


def _scheduler_engine_step(scheduler_output: Any) -> int | None:
    value = getattr(scheduler_output, "engine_step", None)
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _graph_mode(cudagraph_stats: Any) -> str:
    if cudagraph_stats is None:
        return "unknown"
    for attr in ("mode", "cudagraph_mode", "graph_mode"):
        value = getattr(cudagraph_stats, attr, None)
        if value is not None:
            return str(getattr(value, "name", value))
    return str(cudagraph_stats)


def _counters_to_layers(layer_ids: Any, counters: Any) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    rows = counters.tolist() if hasattr(counters, "tolist") else counters
    for layer_id, row in zip(layer_ids, rows):
        result[str(int(layer_id))] = {
            "misses": int(row[0]),
            "routes": int(row[1]),
            "steps": int(row[2]),
        }
    return result
