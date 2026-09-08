"""Pure helpers for slot-cache telemetry.

The cache hook counts misses and routed expert uses on device. Hit rate must use
observed routed uses as its denominator: speculative verification can route more
than one token in a single engine step, so a fixed top-k denominator is wrong.

Quiescent snapshot telemetry is intentionally pure here. It validates and
summarizes snapshots captured by an engine-owned safe point; it does not make a
background thread read CUDA counters. Live capture integration is BLOCKED until
there is a proven quiescent engine callback that owns the CUDA stream/barrier.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
import re

SNAPSHOT_SCHEMA = "slot-cache-quiescent-snapshot-v1"
DELTA_SCHEMA = "slot-cache-quiescent-delta-v1"
_COUNTERS = ("misses", "routes", "steps")
_PROVENANCE_FIELDS = ("run_id", "source_sha", "engine_generation", "seq")
_SOURCE_SHA_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")


def summarize_window(
    *, delta_misses: int, delta_routes: int, delta_steps: int
) -> dict[str, float] | None:
    """Summarize one monotonic-counter window, or reject an empty/reset window."""
    if delta_misses < 0 or delta_routes <= 0 or delta_steps <= 0:
        return None
    return {
        "hit_rate": 1.0 - (delta_misses / delta_routes),
        "misses_per_step": delta_misses / delta_steps,
        "routes_per_step": delta_routes / delta_steps,
    }


def _canonical_layer_key(layer: int | str) -> str:
    if isinstance(layer, bool):
        raise ValueError("expected_layers must contain canonical non-negative integer layer ids")
    if isinstance(layer, int):
        if layer < 0:
            raise ValueError("expected_layers must contain canonical non-negative integer layer ids")
        return str(layer)
    if isinstance(layer, str):
        if re.fullmatch(r"0|[1-9][0-9]*", layer) is None:
            raise ValueError("expected_layers must contain canonical non-negative integer layer ids")
        return layer
    raise ValueError("expected_layers must contain canonical non-negative integer layer ids")


def _expected_layer_keys(expected_layers: Iterable[int | str]) -> list[str]:
    materialized = list(expected_layers)
    if not materialized:
        raise ValueError("expected_layers must not be empty")
    keys = [_canonical_layer_key(layer) for layer in materialized]
    if len(set(keys)) != len(keys):
        raise ValueError("expected_layers must not contain duplicate layers")
    return sorted(keys, key=lambda item: int(item))


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _require_counter(value: object, layer: str, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"layer {layer} {field} must be an integer")
    if value < 0:
        raise ValueError(f"layer {layer} {field} must be non-negative")
    return value


def _require_positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"provenance {field} must be a positive integer")
    return value


def _validate_provenance(value: object) -> dict[str, object]:
    provenance = _require_mapping(value, "provenance")
    if set(provenance) != set(_PROVENANCE_FIELDS):
        raise ValueError(f"provenance must contain exactly {list(_PROVENANCE_FIELDS)}")
    run_id = provenance["run_id"]
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("provenance run_id must be a non-empty string")
    source_sha = provenance["source_sha"]
    if not isinstance(source_sha, str) or _SOURCE_SHA_RE.fullmatch(source_sha) is None:
        raise ValueError("provenance source_sha must be a canonical lowercase SHA")
    return {
        "run_id": run_id,
        "source_sha": source_sha,
        "engine_generation": _require_positive_int(provenance["engine_generation"], "engine_generation"),
        "seq": _require_positive_int(provenance["seq"], "seq"),
    }


def _validate_delta_provenance(start: Mapping[str, object], end: Mapping[str, object]) -> None:
    for field in ("run_id", "source_sha", "engine_generation"):
        if start[field] != end[field]:
            raise ValueError(f"snapshot provenance {field} must match")
    if end["seq"] <= start["seq"]:
        raise ValueError("snapshot provenance seq must increase")


def validate_counter_snapshot(
    snapshot: Mapping[str, object], *, expected_layers: Iterable[int | str]
) -> dict[str, object]:
    """Validate and return a canonical quiescent raw-counter snapshot.

    Requirements are deliberately strict: exact layer set, raw non-negative
    integer counters, misses no larger than observed routes, aligned step counts
    across layers, and explicit provenance describing where the snapshot came
    from. This helper does not infer any unobserved route or miss data.
    """
    snapshot = _require_mapping(snapshot, "snapshot")
    if snapshot.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError(f"schema must be {SNAPSHOT_SCHEMA}")
    provenance = _validate_provenance(snapshot.get("provenance"))
    layers = _require_mapping(snapshot.get("layers"), "layers")
    expected = _expected_layer_keys(expected_layers)
    actual = set(layers)
    if actual != set(expected):
        raise ValueError(f"expected layers {expected}, got {sorted(actual)}")

    canonical_layers: dict[str, dict[str, int]] = {}
    aligned_steps: int | None = None
    for layer in expected:
        counters = _require_mapping(layers[layer], f"layer {layer}")
        extra = set(counters) - set(_COUNTERS)
        missing = set(_COUNTERS) - set(counters)
        if extra or missing:
            raise ValueError(f"layer {layer} counters must be exactly {list(_COUNTERS)}")
        row = {field: _require_counter(counters[field], layer, field) for field in _COUNTERS}
        if row["misses"] > row["routes"]:
            raise ValueError(f"layer {layer} misses cannot exceed routes")
        if aligned_steps is None:
            aligned_steps = row["steps"]
        elif row["steps"] != aligned_steps:
            raise ValueError("all layers must have consistent aligned steps")
        canonical_layers[layer] = row

    return {
        "schema": SNAPSHOT_SCHEMA,
        "provenance": provenance,
        "layers": canonical_layers,
    }


def _build_delta_from_validated(
    start_v: Mapping[str, object], end_v: Mapping[str, object]
) -> dict[str, object]:
    _validate_delta_provenance(start_v["provenance"], end_v["provenance"])
    layers: dict[str, dict[str, int]] = {}
    aligned_delta_steps: int | None = None
    for layer in start_v["layers"]:
        s = start_v["layers"][layer]
        e = end_v["layers"][layer]
        row = {}
        for field in _COUNTERS:
            delta = e[field] - s[field]
            if delta < 0:
                raise ValueError(f"counter reset for layer {layer} {field}")
            row[field] = delta
        if row["misses"] > row["routes"]:
            raise ValueError(f"layer {layer} delta misses cannot exceed delta routes")
        if aligned_delta_steps is None:
            aligned_delta_steps = row["steps"]
        elif row["steps"] != aligned_delta_steps:
            raise ValueError("all layers must have consistent aligned delta steps")
        layers[layer] = row
    return {
        "schema": DELTA_SCHEMA,
        "provenance": {
            "start": deepcopy(start_v["provenance"]),
            "end": deepcopy(end_v["provenance"]),
        },
        "layers": layers,
        "raw": {"start": deepcopy(start_v), "end": deepcopy(end_v)},
    }


def counter_snapshot_delta(
    start: Mapping[str, object],
    end: Mapping[str, object],
    *,
    expected_layers: Iterable[int | str],
) -> dict[str, object]:
    """Return a validated delta between two quiescent raw-counter snapshots."""
    expected_layers = _expected_layer_keys(expected_layers)
    start_v = validate_counter_snapshot(start, expected_layers=expected_layers)
    end_v = validate_counter_snapshot(end, expected_layers=expected_layers)
    return _build_delta_from_validated(start_v, end_v)


def _summarize_delta_layers(layers: Mapping[str, object]) -> dict[str, object] | None:
    per_layer: dict[str, dict[str, float]] = {}
    total_misses = total_routes = total_steps = 0
    for layer in sorted(layers, key=lambda item: int(item) if str(item).isdigit() else str(item)):
        counters = _require_mapping(layers[layer], f"layer {layer}")
        row = {field: _require_counter(counters[field], str(layer), field) for field in _COUNTERS}
        summary = summarize_window(
            delta_misses=row["misses"], delta_routes=row["routes"], delta_steps=row["steps"]
        )
        if summary is not None:
            per_layer[str(layer)] = summary
        total_misses += row["misses"]
        total_routes += row["routes"]
        total_steps += row["steps"]
    total = summarize_window(
        delta_misses=total_misses, delta_routes=total_routes, delta_steps=total_steps
    )
    if total is None:
        return None
    return {**total, "layers": per_layer}


def summarize_snapshot_delta(delta: Mapping[str, object]) -> dict[str, object] | None:
    """Summarize a snapshot delta after recomputing it from validated raw snapshots."""
    delta = _require_mapping(delta, "delta")
    if delta.get("schema") != DELTA_SCHEMA:
        raise ValueError(f"schema must be {DELTA_SCHEMA}")
    raw = _require_mapping(delta.get("raw"), "raw")
    raw_start = _require_mapping(raw.get("start"), "raw start")
    raw_end = _require_mapping(raw.get("end"), "raw end")
    raw_layers = _require_mapping(raw_start.get("layers"), "raw start layers")
    expected_layers = _expected_layer_keys(raw_layers.keys())
    start_v = validate_counter_snapshot(raw_start, expected_layers=expected_layers)
    end_v = validate_counter_snapshot(raw_end, expected_layers=expected_layers)
    recomputed = _build_delta_from_validated(start_v, end_v)

    expected_provenance = recomputed["provenance"]
    if delta.get("provenance") != expected_provenance:
        raise ValueError("delta provenance does not match raw snapshots")
    supplied_layers = _require_mapping(delta.get("layers"), "layers")
    if supplied_layers != recomputed["layers"]:
        raise ValueError("delta layers do not match raw snapshots")
    return _summarize_delta_layers(recomputed["layers"])
