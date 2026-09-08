#!/usr/bin/env python3
"""Offline replay of GLM-5.3 router traces for cache and prefetch experiments."""

from __future__ import annotations

import argparse
import heapq
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

SCHEMA = "glm53-slotcache-offline-sim-v1"
POLICIES = ("lru", "lfu", "static", "hybrid25", "hybrid50", "hybrid75")


def _policy_parts(policy: str) -> tuple[str, float]:
    if policy.startswith("hybrid"):
        return "hybrid", int(policy.removeprefix("hybrid")) / 100.0
    return policy, 0.5


def _finalize_metrics(hits: int, misses: int) -> dict[str, int | float]:
    accesses = hits + misses
    return {
        "hits": hits,
        "misses": misses,
        "accesses": accesses,
        "hit_rate": hits / accesses if accesses else 0.0,
    }


def simulate_cache(
    rows: np.ndarray,
    capacity: int,
    policy: str,
    *,
    train_rows: np.ndarray | None = None,
    pin_fraction: float = 0.5,
) -> dict[str, int | float]:
    """Replay one layer and score only ``rows``; optional training rows warm policy state."""
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if policy not in {"lru", "lfu", "static", "hybrid"}:
        raise ValueError(f"unsupported policy: {policy}")
    if not 0.0 <= pin_fraction <= 1.0:
        raise ValueError("pin_fraction must be between 0 and 1")

    def valid_experts(source: np.ndarray | None):
        if source is None:
            return
        for expert_value in source.flat:
            expert = int(expert_value)
            if expert >= 0:
                yield expert

    hits = misses = 0
    if policy in {"static", "hybrid"}:
        if train_rows is None:
            raise ValueError(f"{policy} requires train_rows")
        train_flat = train_rows[train_rows >= 0].astype(np.int64, copy=False)
        frequencies = np.bincount(train_flat, minlength=1)
        order = sorted(range(len(frequencies)), key=lambda expert: (-int(frequencies[expert]), expert))
        if policy == "static":
            pinned = set(order[:capacity])
            for expert in valid_experts(rows):
                if expert in pinned:
                    hits += 1
                else:
                    misses += 1
        else:
            pin_count = min(capacity, max(0, int(capacity * pin_fraction)))
            pinned = set(order[:pin_count])
            dynamic_capacity = capacity - len(pinned)
            dynamic: OrderedDict[int, None] = OrderedDict()

            def touch_dynamic(expert: int, *, score: bool) -> None:
                nonlocal hits, misses
                if expert in pinned:
                    if score:
                        hits += 1
                    return
                if expert in dynamic:
                    if score:
                        hits += 1
                    dynamic.move_to_end(expert)
                    return
                if score:
                    misses += 1
                if dynamic_capacity:
                    dynamic[expert] = None
                    if len(dynamic) > dynamic_capacity:
                        dynamic.popitem(last=False)

            for expert in valid_experts(train_rows):
                touch_dynamic(expert, score=False)
            for expert in valid_experts(rows):
                touch_dynamic(expert, score=True)
    elif policy == "lru":
        resident: OrderedDict[int, None] = OrderedDict()

        def touch_lru(expert: int, *, score: bool) -> None:
            nonlocal hits, misses
            if expert in resident:
                if score:
                    hits += 1
                resident.move_to_end(expert)
            else:
                if score:
                    misses += 1
                resident[expert] = None
                if len(resident) > capacity:
                    resident.popitem(last=False)

        for expert in valid_experts(train_rows):
            touch_lru(expert, score=False)
        for expert in valid_experts(rows):
            touch_lru(expert, score=True)
    else:
        frequencies: dict[int, int] = {}
        last_touch: dict[int, int] = {}
        resident_set: set[int] = set()
        heap: list[tuple[int, int, int]] = []
        clock = 0

        def touch_lfu(expert: int, *, score: bool) -> None:
            nonlocal hits, misses, clock
            clock += 1
            frequencies[expert] = frequencies.get(expert, 0) + 1
            if expert in resident_set:
                if score:
                    hits += 1
            else:
                if score:
                    misses += 1
                if len(resident_set) >= capacity:
                    while heap:
                        frequency, touched, victim = heapq.heappop(heap)
                        if (
                            victim in resident_set
                            and frequencies[victim] == frequency
                            and last_touch[victim] == touched
                        ):
                            resident_set.remove(victim)
                            break
                resident_set.add(expert)
            last_touch[expert] = clock
            heapq.heappush(heap, (frequencies[expert], clock, expert))

        for expert in valid_experts(train_rows):
            touch_lfu(expert, score=False)
        for expert in valid_experts(rows):
            touch_lfu(expert, score=True)
    return _finalize_metrics(hits, misses)


def train_transition(source: np.ndarray, target: np.ndarray, *, experts: int) -> np.ndarray:
    """Learn P(target expert | source expert) from aligned routed-expert sets."""
    if source.shape[0] != target.shape[0]:
        raise ValueError("source and target must have the same number of rows")
    counts = np.zeros((experts, experts), dtype=np.float64)
    if not len(source):
        return counts
    sources = np.repeat(source.astype(np.int64, copy=False), target.shape[1], axis=1)
    targets = np.tile(target.astype(np.int64, copy=False), (1, source.shape[1]))
    valid = (sources >= 0) & (sources < experts) & (targets >= 0) & (targets < experts)
    np.add.at(counts, (sources[valid], targets[valid]), 1.0)
    totals = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, totals, out=np.zeros_like(counts), where=totals != 0)


def rank_predictions(
    transition: np.ndarray,
    source_row: np.ndarray,
    *,
    resident: set[int],
    limit: int,
) -> list[int]:
    """Rank positive-evidence transition targets, excluding experts already resident."""
    sources = sorted({int(item) for item in source_row if 0 <= int(item) < transition.shape[0]})
    if not sources or limit <= 0:
        return []
    scores = transition[sources].sum(axis=0)
    candidates = [expert for expert, score in enumerate(scores) if score > 0 and expert not in resident]
    candidates.sort(key=lambda expert: (-float(scores[expert]), expert))
    return candidates[:limit]


def score_prefetch_limits(
    source_rows: np.ndarray,
    target_rows: np.ndarray,
    transition: np.ndarray,
    *,
    capacity: int,
    limits: tuple[int, ...],
    warm_target_rows: np.ndarray | None = None,
) -> dict[int, dict[str, int | float]]:
    """Score multiple prediction budgets against one demand-only LRU replay."""
    if source_rows.shape[0] != target_rows.shape[0]:
        raise ValueError("source_rows and target_rows must be aligned")
    resident: OrderedDict[int, None] = OrderedDict()

    def demand(expert: int) -> None:
        if expert in resident:
            resident.move_to_end(expert)
        else:
            resident[expert] = None
            if len(resident) > capacity:
                resident.popitem(last=False)

    if warm_target_rows is not None:
        for expert_value in warm_target_rows.flat:
            expert = int(expert_value)
            if expert >= 0:
                demand(expert)
    counters = {limit: {"issued": 0, "useful": 0, "demand_misses": 0} for limit in limits}
    for source_row, target_row in zip(source_rows, target_rows):
        resident_set = set(resident)
        predictions = rank_predictions(
            transition,
            source_row,
            resident=resident_set,
            limit=max(limits),
        )
        targets = {int(item) for item in target_row if int(item) >= 0}
        misses = targets - resident_set
        for limit in limits:
            issued = predictions[:limit]
            counters[limit]["issued"] += len(issued)
            counters[limit]["useful"] += len(set(issued) & misses)
            counters[limit]["demand_misses"] += len(misses)
        for expert_value in target_row:
            expert = int(expert_value)
            if expert >= 0:
                demand(expert)
    output: dict[int, dict[str, int | float]] = {}
    for limit, counts in counters.items():
        issued = counts["issued"]
        useful = counts["useful"]
        demand_misses = counts["demand_misses"]
        output[limit] = {
            **counts,
            "precision": useful / issued if issued else 0.0,
            "recall": useful / demand_misses if demand_misses else 0.0,
        }
    return output


def score_prefetch(
    source_rows: np.ndarray,
    target_rows: np.ndarray,
    transition: np.ndarray,
    *,
    capacity: int,
    limit: int,
    warm_target_rows: np.ndarray | None = None,
) -> dict[str, int | float]:
    return score_prefetch_limits(
        source_rows,
        target_rows,
        transition,
        capacity=capacity,
        limits=(limit,),
        warm_target_rows=warm_target_rows,
    )[limit]


def greedy_allocate(
    curves: dict[int, dict[int, float]],
    *,
    total: int,
    minimum: int,
    maximum: int,
    step: int,
) -> dict[int, int]:
    """Allocate a fixed slot budget by the largest next measured hit-rate gain."""
    if not curves:
        raise ValueError("curves must not be empty")
    allocation = {layer: minimum for layer in sorted(curves)}
    remaining = total - minimum * len(allocation)
    if remaining < 0 or remaining % step:
        raise ValueError("total is incompatible with minimum and step")
    while remaining:
        candidates: list[tuple[float, int]] = []
        for layer, current in allocation.items():
            following = current + step
            if following <= maximum and following in curves[layer] and current in curves[layer]:
                gain = curves[layer][following] - curves[layer][current]
                candidates.append((gain, layer))
        if not candidates:
            raise ValueError("cannot spend complete slot budget within curve bounds")
        _, best_layer = max(candidates, key=lambda item: (item[0], -item[1]))
        allocation[best_layer] += step
        remaining -= step
    return allocation


def build_segment_splits(
    steps: list[tuple[float, int, int, int]],
    *,
    row_count: int,
    marks_path: Path,
    train_fraction: float = 0.7,
) -> dict[str, dict[str, np.ndarray]]:
    """Return chronological decode-only train/held-out row indices for each marked segment."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    timestamps = np.concatenate(
        [np.full(tokens, timestamp, dtype=np.float64) for timestamp, tokens, _, _ in steps]
    )[:row_count]
    decode = np.concatenate(
        [np.full(tokens, tokens <= 8, dtype=np.bool_) for _, tokens, _, _ in steps]
    )[:row_count]
    marks: list[tuple[str, float]] = []
    for line in marks_path.read_text().splitlines():
        if not line.startswith("MARK "):
            continue
        _, name, timestamp = line.split()
        marks.append((name, float(timestamp)))
    if not marks:
        raise ValueError(f"no MARK lines in {marks_path}")
    splits: dict[str, dict[str, np.ndarray]] = {}
    previous_end = float("-inf")
    row_ids = np.arange(row_count, dtype=np.int64)
    for name, end in marks:
        indices = row_ids[decode & (timestamps > previous_end) & (timestamps <= end)]
        if len(indices) < 2:
            raise ValueError(f"segment {name!r} has fewer than two decode rows")
        cut = min(len(indices) - 1, max(1, int(len(indices) * train_fraction)))
        splits[name] = {"train": indices[:cut], "held_out": indices[cut:]}
        previous_end = end
    return splits


def load_trace(trace_dir: Path) -> tuple[np.ndarray, np.ndarray, list[tuple[float, int, int, int]]]:
    arrays: list[np.ndarray] = []
    steps: list[tuple[float, int, int, int]] = []
    for binary in sorted(trace_dir.glob("trace-*.i16")):
        step_path = binary.with_suffix(".steps")
        if not step_path.exists():
            raise ValueError(f"missing step sidecar: {step_path}")
        file_steps = []
        for line in step_path.read_text().splitlines():
            timestamp, tokens, layers, topk = line.split()
            file_steps.append((float(timestamp), int(tokens), int(layers), int(topk)))
        if not file_steps:
            raise ValueError(f"empty step sidecar: {step_path}")
        layers, topk = file_steps[0][2:]
        raw = np.fromfile(binary, dtype=np.int16)
        row_width = layers * topk
        complete_rows = raw.size // row_width
        arrays.append(raw[: complete_rows * row_width].reshape(complete_rows, layers, topk))
        steps.extend(file_steps)
    if not arrays:
        raise ValueError(f"no trace-*.i16 files in {trace_dir}")
    routed = np.concatenate(arrays)
    tagged_rows = sum(tokens for _, tokens, _, _ in steps)
    routed = routed[:tagged_rows]
    tags = np.concatenate(
        [np.full(tokens, 0 if tokens <= 8 else 1, dtype=np.int8) for _, tokens, _, _ in steps]
    )[: len(routed)]
    return routed, tags, steps


def _score_policy(
    routed: np.ndarray,
    splits: dict[str, dict[str, np.ndarray]],
    allocation: dict[int, int],
    policy_name: str,
) -> dict[str, object]:
    policy, pin_fraction = _policy_parts(policy_name)
    aggregate_hits = aggregate_misses = 0
    per_segment: dict[str, dict[str, int | float]] = {}
    for segment, split in splits.items():
        hits = misses = 0
        for layer, capacity in allocation.items():
            result = simulate_cache(
                routed[split["held_out"], layer, :],
                capacity,
                policy,
                train_rows=routed[split["train"], layer, :],
                pin_fraction=pin_fraction,
            )
            hits += int(result["hits"])
            misses += int(result["misses"])
        per_segment[segment] = _finalize_metrics(hits, misses)
        aggregate_hits += hits
        aggregate_misses += misses
    return {
        "aggregate": _finalize_metrics(aggregate_hits, aggregate_misses),
        "per_segment": per_segment,
    }


def _derive_allocation(
    routed: np.ndarray,
    train_indices: np.ndarray,
    layers: list[int],
    policy_name: str,
    *,
    total: int,
    minimum: int,
    maximum: int,
    step: int,
    curve_sample: int,
) -> tuple[dict[int, int], dict[int, dict[int, float]]]:
    stride = max(1, len(train_indices) // curve_sample)
    sampled = train_indices[::stride][:curve_sample]
    cut = min(len(sampled) - 1, max(1, int(len(sampled) * 0.7)))
    seed, tune = sampled[:cut], sampled[cut:]
    policy, pin_fraction = _policy_parts(policy_name)
    capacities = list(range(minimum, maximum + 1, step))
    curves: dict[int, dict[int, float]] = {}
    for layer in layers:
        curves[layer] = {}
        for capacity in capacities:
            result = simulate_cache(
                routed[tune, layer, :],
                capacity,
                policy,
                train_rows=routed[seed, layer, :],
                pin_fraction=pin_fraction,
            )
            curves[layer][capacity] = float(result["hit_rate"])
    return (
        greedy_allocate(
            curves,
            total=total,
            minimum=minimum,
            maximum=maximum,
            step=step,
        ),
        curves,
    )


def _previous_single_indices(steps: list[tuple[float, int, int, int]], row_count: int) -> np.ndarray:
    previous = np.full(row_count, -1, dtype=np.int64)
    offset = 0
    prior_single = -1
    for _, tokens, _, _ in steps:
        if offset >= row_count:
            break
        if tokens == 1:
            if prior_single >= 0:
                previous[offset] = prior_single
            prior_single = offset
        else:
            prior_single = -1
        offset += tokens
    return previous


def _merge_prefetch(parts: list[dict[str, int | float]]) -> dict[str, int | float]:
    issued = sum(int(part["issued"]) for part in parts)
    useful = sum(int(part["useful"]) for part in parts)
    misses = sum(int(part["demand_misses"]) for part in parts)
    return {
        "issued": issued,
        "useful": useful,
        "demand_misses": misses,
        "precision": useful / issued if issued else 0.0,
        "recall": useful / misses if misses else 0.0,
    }


def _score_predictor(
    routed: np.ndarray,
    steps: list[tuple[float, int, int, int]],
    splits: dict[str, dict[str, np.ndarray]],
    allocation: dict[int, int],
    *,
    experts: int,
    limits: tuple[int, ...],
    strategy: str,
) -> dict[str, object]:
    if strategy not in {"adjacent_layer", "previous_token"}:
        raise ValueError(f"unsupported predictor strategy: {strategy}")
    previous = _previous_single_indices(steps, len(routed))
    per_limit_segment: dict[int, dict[str, list[dict[str, int | float]]]] = {
        limit: {segment: [] for segment in splits} for limit in limits
    }
    for segment, split in splits.items():
        train_indices = split["train"]
        train_set = set(int(item) for item in train_indices)
        allowed = set(int(item) for item in np.concatenate([train_indices, split["held_out"]]))
        for layer, capacity in allocation.items():
            if strategy == "adjacent_layer":
                source_layer = layer - 1
                if source_layer < 0:
                    continue
                train_target_indices = train_indices
                train_source_indices = train_indices
                target_indices = split["held_out"]
                source_indices = target_indices
            else:
                source_layer = layer
                train_target_indices = np.array(
                    [idx for idx in train_indices if previous[idx] in train_set],
                    dtype=np.int64,
                )
                train_source_indices = previous[train_target_indices]
                target_indices = np.array(
                    [idx for idx in split["held_out"] if previous[idx] in allowed],
                    dtype=np.int64,
                )
                source_indices = previous[target_indices]
            transition = train_transition(
                routed[train_source_indices, source_layer, :],
                routed[train_target_indices, layer, :],
                experts=experts,
            )
            scores = score_prefetch_limits(
                routed[source_indices, source_layer, :],
                routed[target_indices, layer, :],
                transition,
                capacity=capacity,
                limits=limits,
                warm_target_rows=routed[train_indices, layer, :],
            )
            for limit in limits:
                per_limit_segment[limit][segment].append(scores[limit])

    output_limits: dict[str, object] = {}
    for limit in limits:
        per_segment = {
            segment: _merge_prefetch(parts)
            for segment, parts in per_limit_segment[limit].items()
        }
        output_limits[str(limit)] = {
            "aggregate": _merge_prefetch(list(per_segment.values())),
            "per_segment": per_segment,
        }
    evaluation_scope = (
        "all held-out decode rows"
        if strategy == "adjacent_layer"
        else "unambiguous consecutive single-token decode steps; batched C4/C8 steps excluded"
    )
    return {
        "training_scope": "that workload segment's chronological training prefix only",
        "evaluation_scope": evaluation_scope,
        "limits": output_limits,
    }


def _run_experiments(
    routed: np.ndarray,
    steps: list[tuple[float, int, int, int]],
    slot_map: dict[str, object],
    *,
    marks: Path,
    train_fraction: float,
    curve_sample: int,
    minimum: int,
    maximum: int,
    step: int,
    limits: tuple[int, ...],
    experts: int,
) -> dict[str, object]:
    splits = build_segment_splits(
        steps,
        row_count=len(routed),
        marks_path=marks,
        train_fraction=train_fraction,
    )
    allocation = {int(layer): int(capacity) for layer, capacity in slot_map["per_layer"].items()}
    if sum(allocation.values()) != int(slot_map["total"]):
        raise ValueError("slot map total does not equal per-layer capacities")
    segments = {
        name: {
            "train_tokens": len(split["train"]),
            "held_out_tokens": len(split["held_out"]),
        }
        for name, split in splits.items()
    }
    frozen = {
        policy: _score_policy(routed, splits, allocation, policy)
        for policy in POLICIES
    }
    all_train = np.sort(np.concatenate([split["train"] for split in splits.values()]))
    optimized: dict[str, object] = {}
    for policy in POLICIES:
        candidate, curves = _derive_allocation(
            routed,
            all_train,
            sorted(allocation),
            policy,
            total=int(slot_map["total"]),
            minimum=minimum,
            maximum=maximum,
            step=step,
            curve_sample=curve_sample,
        )
        scored = _score_policy(routed, splits, candidate, policy)
        scored["allocation"] = {str(layer): capacity for layer, capacity in candidate.items()}
        scored["tuning_mean_hit_rate"] = float(
            np.mean([curves[layer][candidate[layer]] for layer in candidate])
        )
        optimized[policy] = scored

    baseline = float(frozen["lru"]["aggregate"]["hit_rate"])
    candidates = {
        **{f"frozen:{policy}": value for policy, value in frozen.items()},
        **{f"optimized:{policy}": value for policy, value in optimized.items()},
    }
    best_name, best_value = max(
        candidates.items(),
        key=lambda item: float(item[1]["aggregate"]["hit_rate"]),
    )
    best_hit = float(best_value["aggregate"]["hit_rate"])
    tools_segment = "tools_low" if "tools_low" in splits else next(iter(splits))
    tools_baseline = float(frozen["lru"]["per_segment"][tools_segment]["hit_rate"])
    tools_best = float(best_value["per_segment"][tools_segment]["hit_rate"])
    cache_gate = {
        "threshold_points": 5.0,
        "baseline": "frozen:lru",
        "baseline_hit_rate": baseline,
        "best_candidate": best_name,
        "best_hit_rate": best_hit,
        "gain_points": (best_hit - baseline) * 100.0,
        "agent_segment": tools_segment,
        "agent_gain_points": (tools_best - tools_baseline) * 100.0,
        "pass": (best_hit - baseline) >= 0.05 and (tools_best - tools_baseline) >= 0.05,
    }

    adjacent = _score_predictor(
        routed,
        steps,
        splits,
        allocation,
        experts=experts,
        limits=limits,
        strategy="adjacent_layer",
    )
    previous = _score_predictor(
        routed,
        steps,
        splits,
        allocation,
        experts=experts,
        limits=limits,
        strategy="previous_token",
    )
    predictor_candidates = []
    for strategy, result in (("adjacent_layer", adjacent), ("previous_token", previous)):
        for limit, value in result["limits"].items():
            predictor_candidates.append((float(value["aggregate"]["precision"]), strategy, limit, value))
    precision, strategy, limit, value = max(predictor_candidates)
    predictor_gate = {
        "threshold_precision": 0.5,
        "best_strategy": strategy,
        "best_limit": int(limit),
        "precision": precision,
        "recall": float(value["aggregate"]["recall"]),
        "pass": precision >= 0.5,
    }
    return {
        "segments": segments,
        "cache_policies": {
            "frozen_allocation": frozen,
            "optimized_allocation": optimized,
            "gate": cache_gate,
        },
        "prefetch": {
            "adjacent_layer": adjacent,
            "previous_token": previous,
            "gate": predictor_gate,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", required=True, type=Path)
    parser.add_argument("--slot-map", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--marks", type=Path)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--curve-sample", type=int, default=6000)
    parser.add_argument("--minimum", type=int)
    parser.add_argument("--maximum", type=int)
    parser.add_argument("--step", type=int)
    parser.add_argument("--predict-limits", default="1,2,4,8")
    args = parser.parse_args()

    routed, tags, steps = load_trace(args.trace_dir)
    slot_map = json.loads(args.slot_map.read_text())
    valid = routed[routed >= 0]
    experts = int(valid.max()) + 1 if valid.size else 0
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "input": {
            "tokens": int(len(routed)),
            "decode_tokens": int(np.count_nonzero(tags == 0)),
            "prefill_tokens": int(np.count_nonzero(tags == 1)),
            "layers": int(routed.shape[1]),
            "topk": int(routed.shape[2]),
            "experts": experts,
            "steps": len(steps),
        },
    }
    if args.marks:
        limits = tuple(sorted({int(item) for item in args.predict_limits.split(",") if item}))
        minimum = args.minimum if args.minimum is not None else int(slot_map.get("min", min(slot_map["per_layer"].values())))
        maximum = args.maximum if args.maximum is not None else int(slot_map.get("max", max(slot_map["per_layer"].values())))
        step = args.step if args.step is not None else int(slot_map.get("step", 16))
        payload.update(
            _run_experiments(
                routed,
                steps,
                slot_map,
                marks=args.marks,
                train_fraction=args.train_fraction,
                curve_sample=args.curve_sample,
                minimum=minimum,
                maximum=maximum,
                step=step,
                limits=limits,
                experts=experts,
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["input"], sort_keys=True))
    if "cache_policies" in payload:
        print("CACHE_GATE " + json.dumps(payload["cache_policies"]["gate"], sort_keys=True))
        print("PREFETCH_GATE " + json.dumps(payload["prefetch"]["gate"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
