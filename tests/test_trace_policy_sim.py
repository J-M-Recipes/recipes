import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "recipes/dgx-station-gb300/glm-5.3-nvfp4-uva-slot-cache/scripts/trace_policy_sim.py"
SPEC = importlib.util.spec_from_file_location("trace_policy_sim", SCRIPT)
assert SPEC and SPEC.loader
SIM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SIM)


def _write_trace(tmp_path: Path, rows: np.ndarray, step_sizes: list[int]) -> Path:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    rows.astype(np.int16).tofile(trace_dir / "trace-1.i16")
    with (trace_dir / "trace-1.steps").open("w") as f:
        offset = 0
        for i, size in enumerate(step_sizes, start=1):
            assert offset + size <= len(rows)
            f.write(f"{float(i):.3f} {size} {rows.shape[1]} {rows.shape[2]}\n")
            offset += size
    assert offset == len(rows)
    return trace_dir


def _run(trace_dir: Path, slot_map: Path, output: Path, *extra: str):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--trace-dir",
            str(trace_dir),
            "--slot-map",
            str(slot_map),
            "--output",
            str(output),
            *extra,
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_cli_reports_exact_trace_shape_and_decode_classification(tmp_path):
    rows = np.arange(12 * 2 * 2, dtype=np.int16).reshape(12, 2, 2) % 4
    trace_dir = _write_trace(tmp_path, rows, [9, 1, 1, 1])
    slot_map = tmp_path / "slots.json"
    slot_map.write_text(json.dumps({"total": 4, "per_layer": {"0": 2, "1": 2}}))
    output = tmp_path / "result.json"

    result = _run(trace_dir, slot_map, output)

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text())
    assert payload["schema"] == "glm53-slotcache-offline-sim-v1"
    assert payload["input"] == {
        "tokens": 12,
        "decode_tokens": 3,
        "prefill_tokens": 9,
        "layers": 2,
        "topk": 2,
        "experts": 4,
        "steps": 4,
    }


def test_lfu_retains_a_frequent_expert_that_lru_evicts():
    rows = np.array([[0], [1], [0], [2], [1], [0]], dtype=np.int16)

    lru = SIM.simulate_cache(rows, capacity=2, policy="lru")
    lfu = SIM.simulate_cache(rows, capacity=2, policy="lfu")

    assert lru == {"hits": 1, "misses": 5, "accesses": 6, "hit_rate": 1 / 6}
    assert lfu == {"hits": 2, "misses": 4, "accesses": 6, "hit_rate": 2 / 6}


def test_static_and_hybrid_score_only_held_out_rows():
    train = np.array([[0], [0], [1], [2]], dtype=np.int16)
    held_out = np.array([[0], [1], [3], [0]], dtype=np.int16)

    static = SIM.simulate_cache(held_out, capacity=2, policy="static", train_rows=train)
    hybrid = SIM.simulate_cache(
        held_out,
        capacity=2,
        policy="hybrid",
        train_rows=train,
        pin_fraction=0.5,
    )

    assert static == {"hits": 3, "misses": 1, "accesses": 4, "hit_rate": 0.75}
    assert hybrid == {"hits": 2, "misses": 2, "accesses": 4, "hit_rate": 0.5}


def test_segment_splits_exclude_prefill_and_preserve_chronology(tmp_path):
    steps = [
        (1.0, 9, 2, 2),
        (2.0, 1, 2, 2),
        (3.0, 1, 2, 2),
        (4.0, 9, 2, 2),
        (5.0, 1, 2, 2),
        (6.0, 1, 2, 2),
    ]
    marks = tmp_path / "marks.txt"
    marks.write_text("MARK first 3.0\nMARK second 6.0\n")

    splits = SIM.build_segment_splits(steps, row_count=22, marks_path=marks, train_fraction=0.5)

    np.testing.assert_array_equal(splits["first"]["train"], np.array([9]))
    np.testing.assert_array_equal(splits["first"]["held_out"], np.array([10]))
    np.testing.assert_array_equal(splits["second"]["train"], np.array([20]))
    np.testing.assert_array_equal(splits["second"]["held_out"], np.array([21]))


def test_greedy_allocator_spends_the_exact_budget_on_largest_marginal_gain():
    curves = {
        3: {1: 0.5, 2: 0.9, 3: 1.0},
        4: {1: 0.4, 2: 0.6, 3: 0.95},
    }

    allocation = SIM.greedy_allocate(curves, total=4, minimum=1, maximum=3, step=1)

    assert allocation == {3: 2, 4: 2}
    assert sum(allocation.values()) == 4


def test_transition_predictor_ranks_learned_targets_and_excludes_resident_rows():
    source = np.array([[0, -1], [0, -1], [1, -1], [0, 1]], dtype=np.int16)
    target = np.array([[2, -1], [2, -1], [3, -1], [2, 3]], dtype=np.int16)

    transition = SIM.train_transition(source, target, experts=4)

    assert SIM.rank_predictions(transition, np.array([0]), resident=set(), limit=2) == [2, 3]
    assert SIM.rank_predictions(transition, np.array([1]), resident=set(), limit=2) == [3, 2]
    assert SIM.rank_predictions(transition, np.array([0]), resident={2}, limit=2) == [3]


def test_prefetch_score_counts_only_nonresident_issued_rows_and_true_misses():
    transition = np.zeros((4, 4), dtype=np.float64)
    transition[0, 2] = 1.0
    transition[1, 3] = 1.0
    source = np.array([[0], [1], [0]], dtype=np.int16)
    target = np.array([[2], [3], [1]], dtype=np.int16)
    warm_target = np.array([[0], [1]], dtype=np.int16)

    result = SIM.score_prefetch(
        source,
        target,
        transition,
        capacity=2,
        limit=1,
        warm_target_rows=warm_target,
    )

    assert result == {
        "issued": 2,
        "useful": 2,
        "demand_misses": 3,
        "precision": 1.0,
        "recall": 2 / 3,
    }


def test_cli_runs_held_out_policy_and_adjacent_prefetch_experiments(tmp_path):
    rows = np.zeros((15, 2, 1), dtype=np.int16)
    rows[9:, 0, 0] = [0, 0, 1, 0, 1, 0]
    rows[9:, 1, 0] = [2, 2, 3, 2, 3, 2]
    trace_dir = _write_trace(tmp_path, rows, [9, 1, 1, 1, 1, 1, 1])
    marks = trace_dir / "marks.txt"
    marks.write_text("MARK seg 7.0\n")
    slot_map = tmp_path / "slots.json"
    slot_map.write_text(
        json.dumps(
            {
                "total": 2,
                "min": 1,
                "max": 1,
                "step": 1,
                "per_layer": {"0": 1, "1": 1},
            }
        )
    )
    output = tmp_path / "result.json"

    result = _run(
        trace_dir,
        slot_map,
        output,
        "--marks",
        str(marks),
        "--train-fraction",
        "0.5",
        "--predict-limits",
        "1",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text())
    assert payload["segments"] == {"seg": {"train_tokens": 3, "held_out_tokens": 3}}
    assert payload["cache_policies"]["frozen_allocation"]["lru"]["aggregate"]["hit_rate"] == 0.0
    assert payload["cache_policies"]["frozen_allocation"]["static"]["aggregate"]["hit_rate"] == 2 / 3
    adjacent = payload["prefetch"]["adjacent_layer"]["limits"]["1"]["aggregate"]
    assert adjacent["precision"] == 1.0
    assert adjacent["recall"] == 1.0


def test_predictor_training_is_segment_local():
    routed = np.zeros((8, 2, 1), dtype=np.int16)
    routed[:, 0, 0] = 0
    routed[:, 1, 0] = [2, 2, 4, 2, 3, 3, 4, 3]
    splits = {
        "a": {"train": np.array([0, 1, 2]), "held_out": np.array([3])},
        "b": {"train": np.array([4, 5, 6]), "held_out": np.array([7])},
    }
    steps = [(float(i), 1, 2, 1) for i in range(8)]

    result = SIM._score_predictor(
        routed,
        steps,
        splits,
        {1: 1},
        experts=5,
        limits=(1,),
        strategy="adjacent_layer",
    )

    assert result["limits"]["1"]["per_segment"]["a"]["precision"] == 1.0
    assert result["limits"]["1"]["per_segment"]["b"]["precision"] == 1.0
    assert result["limits"]["1"]["aggregate"]["precision"] == 1.0
