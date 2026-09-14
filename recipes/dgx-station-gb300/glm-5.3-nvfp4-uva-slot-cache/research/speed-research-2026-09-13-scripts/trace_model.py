#!/usr/bin/env python3
"""Trace replay, slot-curve, prefetch and calibrated cost model for GLM-5.3 GB300.

Pure local Python + numpy. Writes:
  findings/b-trace-model.md
  findings/scripts/out/trace_model_results.json
  findings/scripts/out/stack_hists.npz (cache)
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import statistics
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("/Users/jamesmeadlock/hermes/glm53-speed-research-20260913")
TRACE_DIR = ROOT / "station/trace/r1-base"
FINDINGS = ROOT / "findings"
OUTDIR = FINDINGS / "scripts/out"
RECIPE_RESULTS = Path(
    "/Users/jamesmeadlock/hermes/jm-recipes/recipes/recipes/dgx-station-gb300/"
    "glm-5.3-nvfp4-uva-slot-cache/results"
)
CTX_SUMMARY = RECIPE_RESULTS / "2026-09-13-context-slot-curve/SUMMARY.json"
K2_SUMMARY = RECIPE_RESULTS / "2026-09-13-k2-fair-gate-needle-trace/SUMMARY.json"
E1_BUCKETS = RECIPE_RESULTS / "2026-09-08-e1-v2-live/analysis/corrected-buckets.json"
K2_LOG = ROOT / "station/k2gate.log.gz"

N_LAYERS_ATTN = 78
N_MOE_LAYERS = 75
TOPK = 8
EXPERTS = 256
DEFAULT_ACCEPT_K1 = 1.80  # used for context-curve step conversion; receipt 512K acceptance is malformed.
DEFAULT_ACCEPT_K2 = 2.08
C2C_BW_GB_S_LEDGER = 358.2


def pct(values, q):
    if not values:
        return None
    a = sorted(values)
    if len(a) == 1:
        return a[0]
    pos = (len(a) - 1) * q / 100.0
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi:
        return a[lo]
    return a[lo] * (hi - pos) + a[hi] * (pos - lo)


def safe_mean(values):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(statistics.mean(values)) if values else None


def load_trace():
    steps = []
    files = sorted(TRACE_DIR.glob("trace-*.i16"))
    arrays = []
    for f in files:
        st_path = f.with_suffix(".steps")
        st = []
        with open(st_path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                a, b, c, e = line.split()
                st.append((float(a), int(b), int(c), int(e)))
        L, K = st[0][2], st[0][3]
        arr = np.fromfile(f, dtype=np.int16)
        n = arr.size // (L * K)
        arrays.append(arr[: n * L * K].reshape(n, L, K))
        steps.extend(st)
    X = np.concatenate(arrays, axis=0)
    ntag = sum(nt for _, nt, _, _ in steps)
    X = X[:ntag]
    tags = np.concatenate([np.full(nt, 0 if nt <= 8 else 1, dtype=np.int8) for _, nt, _, _ in steps])[: X.shape[0]]
    Xd = X[tags == 0]
    # Dense layers 0-2 have a constant dummy expert id in this capture; require >8 unique ids.
    moe = [l for l in range(X.shape[1]) if (X[:, l, :] >= 0).any() and len(np.unique(Xd[:, l, :])) > 8]
    E = int(Xd[Xd >= 0].max()) + 1
    return X, Xd, moe, E, X.shape[1], X.shape[2], int((tags == 0).sum()), int((tags == 1).sum())


def load_slots(name="slots-7360-ctx256k.json"):
    data = json.load(open(TRACE_DIR / name))
    return {int(k): int(v) for k, v in data["per_layer"].items()}


def lru_stack_hist_for_layer(values: np.ndarray, E: int = EXPERTS) -> np.ndarray:
    """Histogram of LRU recency ranks for an infinite 256-entry recency stack.

    hist[r] = accesses whose previous occurrence has recency rank r (0=MRU).
    hist[E] = cold/never-seen accesses. LRU hit count at capacity C is sum(hist[:C]).
    """
    hist = np.zeros(E + 1, dtype=np.int64)
    order: list[int] = []  # MRU first, all seen experts retained; E<=256.
    seen = [False] * E
    for raw in values.reshape(-1):
        e = int(raw)
        if e < 0:
            continue
        if seen[e]:
            idx = order.index(e)  # C loop over <=256 ints; fast enough for 43M accesses.
            hist[idx] += 1
            if idx:
                order.pop(idx)
                order.insert(0, e)
        else:
            hist[E] += 1
            seen[e] = True
            order.insert(0, e)
    return hist


def compute_or_load_stack_hists(Xd: np.ndarray, moe: list[int], E: int, refresh=False):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cache = OUTDIR / "stack_hists.npz"
    if cache.exists() and not refresh:
        npz = np.load(cache, allow_pickle=True)
        layers = [int(x) for x in npz["layers"].tolist()]
        hists = {int(l): npz[f"hist_{l}"] for l in layers}
        counts = {int(l): npz[f"counts_{l}"] for l in layers}
        return hists, counts
    hists = {}
    counts = {}
    for i, l in enumerate(moe, 1):
        print(f"stack histogram layer {l} ({i}/{len(moe)})", flush=True)
        Y = Xd[:, l, :]
        hists[l] = lru_stack_hist_for_layer(Y, E)
        y = Y[Y >= 0]
        counts[l] = np.bincount(y, minlength=E).astype(np.int64)
    payload = {"layers": np.array(moe, dtype=np.int16)}
    for l in moe:
        payload[f"hist_{l}"] = hists[l]
        payload[f"counts_{l}"] = counts[l]
    np.savez_compressed(cache, **payload)
    return hists, counts


def hit_from_hist(hist: np.ndarray, cap: int) -> float:
    cap = max(0, min(int(cap), len(hist) - 1))
    total = int(hist.sum())
    return float(hist[:cap].sum() / total) if total else 0.0


def static_hit_from_counts(counts: np.ndarray, cap: int) -> float:
    cap = max(0, min(int(cap), len(counts)))
    total = int(counts.sum())
    if total == 0:
        return 0.0
    return float(np.sort(counts)[::-1][:cap].sum() / total)


def capacities_for_budget(base_slots: dict[int, int], layers: list[int], total: int, variant: str) -> dict[int, int]:
    max_cap = EXPERTS
    if variant == "proportional":
        base_total = sum(base_slots[l] for l in layers)
        raw = {l: base_slots[l] * total / base_total for l in layers}
    elif variant == "flat":
        raw = {l: total / len(layers) for l in layers}
    else:
        raise ValueError(variant)
    caps = {l: max(0, min(max_cap, int(math.floor(raw[l])))) for l in layers}
    current = sum(caps.values())
    # Largest remainder allocation, respecting cap.
    if current < total:
        order = sorted(layers, key=lambda l: (raw[l] - math.floor(raw[l])), reverse=True)
        idx = 0
        while current < total and any(caps[l] < max_cap for l in layers):
            l = order[idx % len(order)]
            if caps[l] < max_cap:
                caps[l] += 1
                current += 1
            idx += 1
    elif current > total:
        order = sorted(layers, key=lambda l: (raw[l] - math.floor(raw[l])))
        idx = 0
        while current > total and any(caps[l] > 0 for l in layers):
            l = order[idx % len(order)]
            if caps[l] > 0:
                caps[l] -= 1
                current -= 1
            idx += 1
    return caps


def evaluate_caps(hists, counts, caps: dict[int, int], layers: list[int]):
    per = []
    for l in layers:
        cap = caps[l]
        h = hit_from_hist(hists[l], cap)
        s = static_hit_from_counts(counts[l], cap)
        per.append((l, cap, h, s))
    mean_lru = float(statistics.mean(x[2] for x in per))
    mean_static = float(statistics.mean(x[3] for x in per))
    return {
        "total_slots": int(sum(caps.values())),
        "mean_lru_hit": mean_lru,
        "mean_static_hit": mean_static,
        "misses_per_layer_step_lru": TOPK * (1.0 - mean_lru),
        "misses_per_layer_step_static": TOPK * (1.0 - mean_static),
        "per_layer": [{"layer": l, "S": cap, "lru_hit": h, "static_hit": s} for l, cap, h, s in per],
    }


def reproduce_policy_table(Xd, moe, base_slots):
    from collections import OrderedDict

    def lru_hit(Y, cap):
        cache = OrderedDict(); hits = tot = 0
        for row in Y:
            for e0 in row:
                e = int(e0)
                if e < 0: continue
                tot += 1
                if e in cache:
                    cache.move_to_end(e); hits += 1
                else:
                    cache[e] = 1
                    if len(cache) > cap:
                        cache.popitem(last=False)
        return hits / tot if tot else 0.0

    def hybrid_hit(Y, cap, pinned):
        pin = set(int(x) for x in pinned); lcap = cap - len(pin)
        cache = OrderedDict(); hits = tot = 0
        for row in Y:
            for e0 in row:
                e = int(e0)
                if e < 0: continue
                tot += 1
                if e in pin:
                    hits += 1; continue
                if e in cache:
                    cache.move_to_end(e); hits += 1
                else:
                    cache[e] = 1
                    if lcap > 0 and len(cache) > lcap:
                        cache.popitem(last=False)
                    elif lcap <= 0:
                        cache.clear()
        return hits / tot if tot else 0.0

    sub = max(1, Xd.shape[0] // 8000)
    rows = {"layer": [], "S": [], "static": [], "lru": [], "hyb50": [], "hyb75": [], "xlayer": [], "temporal": []}
    for idx, l in enumerate(moe):
        S = int(base_slots[l])
        Y = Xd[:, l, :]
        Ys = Y[::sub]
        cnt = Counter(int(e) for e in Y[Y >= 0].ravel().tolist())
        tot = sum(cnt.values())
        order = [e for e, _ in cnt.most_common()]
        static = sum(cnt[e] for e in order[:S]) / tot
        lru = lru_hit(Ys, S)
        h50 = hybrid_hit(Ys, S, order[: S // 2])
        h75 = hybrid_hit(Ys, S, order[: int(S * 0.75)])
        nxt = moe[idx + 1] if idx + 1 < len(moe) else None
        if nxt is not None:
            Z = Xd[:, nxt, :]
            xl = float(np.mean([len(set(a[a >= 0]) & set(b[b >= 0])) / max(1, int((b >= 0).sum())) for a, b in zip(Y[::sub], Z[::sub])]))
        else:
            xl = float("nan")
        tm = float(np.mean([len(set(a[a >= 0]) & set(b[b >= 0])) / max(1, int((b >= 0).sum())) for a, b in zip(Y[:-1:sub], Y[1::sub])]))
        for k, v in zip(rows, (l, S, static, lru, h50, h75, xl, tm)):
            rows[k].append(v)
    def m(k): return float(np.nanmean(rows[k]))
    summary = {
        "decode_tokens": int(Xd.shape[0]),
        "moe_layers": len(moe),
        "total_slots": int(sum(rows["S"])),
        "mean_static_oracle": m("static"),
        "mean_lru": m("lru"),
        "mean_hybrid_pin50": m("hyb50"),
        "mean_hybrid_pin75": m("hyb75"),
        "mean_crosslayer_overlap": m("xlayer"),
        "mean_temporal_overlap": m("temporal"),
        "random_overlap_baseline": TOPK / EXPERTS,
        "misses_per_layer_step": {
            "static": TOPK * (1 - m("static")),
            "lru": TOPK * (1 - m("lru")),
            "hyb75": TOPK * (1 - m("hyb75")),
        },
    }
    return {"summary": summary, "per_layer": rows, "subsample": sub}


def prefetch_metrics(Xd, moe, base_slots):
    # Demand misses under normal LRU are the bytes an oracle one-step prefetch can move off critical path.
    layers_out = []
    total_baseline_misses = 0
    total_accesses = 0
    total_first_step_misses = 0
    prev_pred_attempts = 0
    prev_pred_hit_anyway = 0
    prev_pred_not_requested = 0
    prev_pred_covers_actual_misses = 0
    cache_aware_prev_prefetch_bytes = 0
    for l in moe:
        cap = int(base_slots[l])
        cache = OrderedDict()
        baseline_misses = 0
        accesses = 0
        first_step_misses = 0
        pred_attempts = 0
        pred_hit_anyway = 0
        pred_not_requested = 0
        pred_covers = 0
        cache_aware_bytes = 0
        T = Xd.shape[0]
        for t in range(T):
            req = [int(e) for e in Xd[t, l, :] if int(e) >= 0]
            # Demand process current step.
            step_misses = 0
            for e in req:
                accesses += 1
                if e in cache:
                    cache.move_to_end(e)
                else:
                    baseline_misses += 1
                    step_misses += 1
                    cache[e] = 1
                    if len(cache) > cap:
                        cache.popitem(last=False)
            if t == 0:
                first_step_misses += step_misses
            if t + 1 < T:
                next_req = [int(e) for e in Xd[t + 1, l, :] if int(e) >= 0]
                next_set = set(next_req)
                # Misses that would happen at t+1 if no prefetch is inserted after t.
                next_miss = {e for e in next_set if e not in cache}
                pred = set(req)  # previous-step same-layer predictor
                pred_attempts += len(pred)
                covered = len(next_miss & pred)
                pred_covers += covered
                hit_anyway = len((pred & next_set) - next_miss)
                pred_hit_anyway += hit_anyway
                pred_not_requested += len(pred - next_set)
                # A sane prefetcher would check residency before copying; previous-step requests are resident.
                cache_aware_bytes += len([e for e in pred if e not in cache])
        total_baseline_misses += baseline_misses
        total_accesses += accesses
        total_first_step_misses += first_step_misses
        prev_pred_attempts += pred_attempts
        prev_pred_hit_anyway += pred_hit_anyway
        prev_pred_not_requested += pred_not_requested
        prev_pred_covers_actual_misses += pred_covers
        cache_aware_prev_prefetch_bytes += cache_aware_bytes
        layers_out.append({
            "layer": l,
            "baseline_misses_per_step": baseline_misses / T,
            "perfect_remaining_sync_per_step": first_step_misses / T,
            "prev_pred_attempts_per_step": pred_attempts / max(1, T - 1),
            "prev_pred_covers_actual_misses_per_step": pred_covers / max(1, T - 1),
            "prev_pred_hit_anyway_per_step": pred_hit_anyway / max(1, T - 1),
            "prev_pred_not_requested_per_step": pred_not_requested / max(1, T - 1),
            "cache_aware_prev_prefetch_bytes_per_step": cache_aware_bytes / max(1, T - 1),
        })
    T = Xd.shape[0]
    nlayer_steps = T * len(moe)
    nlayer_pred_steps = (T - 1) * len(moe)
    replay_misses_after_first = max(1, total_baseline_misses - total_first_step_misses)
    return {
        "baseline_hit": 1 - total_baseline_misses / total_accesses,
        "baseline_misses_per_layer_step": total_baseline_misses / nlayer_steps,
        "perfect_remaining_sync_misses_per_layer_step": total_first_step_misses / nlayer_steps,
        "perfect_prefetch_bytes_misses_per_layer_step": (total_baseline_misses - total_first_step_misses) / nlayer_pred_steps,
        "perfect_prefetch_coverage_fraction_of_misses": (total_baseline_misses - total_first_step_misses) / max(1, total_baseline_misses),
        "prev_predictor_blind_prefetches_per_layer_step": prev_pred_attempts / nlayer_pred_steps,
        "prev_predictor_covers_actual_misses_per_layer_step": prev_pred_covers_actual_misses / nlayer_pred_steps,
        "prev_predictor_coverage_fraction_of_next_misses": prev_pred_covers_actual_misses / replay_misses_after_first,
        "prev_predictor_hit_anyway_per_layer_step": prev_pred_hit_anyway / nlayer_pred_steps,
        "prev_predictor_not_requested_per_layer_step": prev_pred_not_requested / nlayer_pred_steps,
        "prev_predictor_cache_aware_prefetches_per_layer_step": cache_aware_prev_prefetch_bytes / nlayer_pred_steps,
        "prev_predictor_hit_anyway_fraction_of_blind_prefetches": prev_pred_hit_anyway / max(1, prev_pred_attempts),
        "prev_predictor_not_requested_fraction_of_blind_prefetches": prev_pred_not_requested / max(1, prev_pred_attempts),
        "prev_predictor_no_miss_value_fraction_of_blind_prefetches": 1.0 - (prev_pred_covers_actual_misses / max(1, prev_pred_attempts)),
        "per_layer": layers_out,
    }


def parse_slot_bytes_from_log(lines):
    vals = []
    pat = re.compile(r"SLOT_CACHE cache built .*: S=(\d+) E=\d+ slot bytes=([0-9.]+) GB")
    for l in lines:
        m = pat.search(l)
        if m:
            S = int(m.group(1)); gb = float(m.group(2))
            if S:
                vals.append(gb * 1e9 / S)
    if not vals:
        return None
    return {
        "n": len(vals),
        "mean_bytes": safe_mean(vals),
        "median_bytes": statistics.median(vals),
        "min_bytes": min(vals),
        "max_bytes": max(vals),
        "mean_MB_decimal": safe_mean(vals) / 1e6,
        "mean_MiB": safe_mean(vals) / (1024**2),
    }


def parse_k2_log():
    with gzip.open(K2_LOG, "rt", errors="replace") as f:
        lines = list(f)
    stats_re = re.compile(
        r"SLOT_CACHE STATS window (\d+)s: steps/layer=(\d+) misses/step/layer=([0-9.]+) HIT=([0-9.]+) "
        r"best=(\d+):([0-9.]+) worst=(\d+):([0-9.]+) median_layer_hit=([0-9.]+)"
    )
    thr_re = re.compile(
        r"INFO (\d\d-\d\d \d\d:\d\d:\d\d).*Avg prompt throughput: ([0-9.]+) tokens/s, "
        r"Avg generation throughput: ([0-9.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs, "
        r"GPU KV cache usage: ([0-9.]+)%, Prefix cache hit rate: ([0-9.]+)%"
    )
    spec_re = re.compile(
        r"INFO (\d\d-\d\d \d\d:\d\d:\d\d).*SpecDecoding metrics: Mean acceptance length: ([0-9.]+), "
        r"Accepted throughput: ([0-9.]+) tokens/s, Drafted throughput: ([0-9.]+) tokens/s, Accepted: (\d+) tokens, Drafted: (\d+) tokens, "
        r"Per-position acceptance rate: ([0-9.]+), ([0-9.]+), Avg Draft acceptance rate: ([0-9.]+)%"
    )
    throughputs = []
    specs = []
    windows = []
    last_stat_idx = -1
    for i, l in enumerate(lines):
        mt = thr_re.search(l)
        if mt:
            throughputs.append({
                "idx": i, "ts": mt.group(1), "prompt_tps": float(mt.group(2)), "gen_tps": float(mt.group(3)),
                "running": int(mt.group(4)), "waiting": int(mt.group(5)), "kv_usage_pct": float(mt.group(6)),
                "prefix_hit_pct": float(mt.group(7)),
            })
        ms = spec_re.search(l)
        if ms:
            specs.append({
                "idx": i, "ts": ms.group(1), "mean_acceptance_length": float(ms.group(2)),
                "accepted_tps": float(ms.group(3)), "drafted_tps": float(ms.group(4)),
                "accepted": int(ms.group(5)), "drafted": int(ms.group(6)),
                "pos_accept_1": float(ms.group(7)), "pos_accept_2": float(ms.group(8)), "avg_draft_accept_pct": float(ms.group(9)),
            })
        m = stats_re.search(l)
        if m:
            sec = int(m.group(1)); steps = int(m.group(2))
            nearby_thr = [x for x in throughputs if last_stat_idx < x["idx"] <= i]
            nearby_spec = [x for x in specs if last_stat_idx < x["idx"] <= i]
            if not nearby_thr:
                nearby_thr = [x for x in throughputs if x["idx"] < i][-2:]
            if not nearby_spec:
                nearby_spec = [x for x in specs if x["idx"] < i][-2:]
            gen = safe_mean([x["gen_tps"] for x in nearby_thr])
            kvu = safe_mean([x["kv_usage_pct"] for x in nearby_thr])
            pref = safe_mean([x["prefix_hit_pct"] for x in nearby_thr])
            accept = safe_mean([x["mean_acceptance_length"] for x in nearby_spec])
            windows.append({
                "idx": i, "window_sec": sec, "steps_per_layer": steps,
                "step_ms_from_steps": sec * 1000.0 / steps if steps else None,
                "misses_per_layer_step": float(m.group(3)), "hit": float(m.group(4)),
                "best_layer": int(m.group(5)), "best_hit": float(m.group(6)),
                "worst_layer": int(m.group(7)), "worst_hit": float(m.group(8)),
                "median_layer_hit": float(m.group(9)),
                "gen_tps_avg": gen, "kv_usage_pct_avg": kvu, "prefix_hit_pct_avg": pref,
                "mean_acceptance_length_avg": accept,
                "step_ms_from_accept_gen": (accept * 1000.0 / gen) if accept and gen else None,
                "estimated_context_tokens": (kvu / 100.0 * 262144.0) if kvu is not None else None,
            })
            last_stat_idx = i
    # Keep active decode windows: stats exists and adjacent generation throughput is non-trivial.
    active = [w for w in windows if (w.get("gen_tps_avg") or 0) >= 20 and w["steps_per_layer"] > 50]
    def dist(vals):
        vals = [v for v in vals if v is not None]
        return {"n": len(vals), "mean": safe_mean(vals), "median": statistics.median(vals) if vals else None,
                "p05": pct(vals, 5), "p25": pct(vals, 25), "p75": pct(vals, 75), "p95": pct(vals, 95),
                "min": min(vals) if vals else None, "max": max(vals) if vals else None}
    first_q = active[: max(1, len(active)//4)]
    last_q = active[-max(1, len(active)//4):]
    return {
        "line_count": len(lines),
        "throughput_count": len(throughputs),
        "spec_count": len(specs),
        "window_count": len(windows),
        "active_window_count": len(active),
        "slot_bytes": parse_slot_bytes_from_log(lines),
        "distributions": {
            "hit": dist([w["hit"] for w in active]),
            "misses_per_layer_step": dist([w["misses_per_layer_step"] for w in active]),
            "gen_tps": dist([w["gen_tps_avg"] for w in active]),
            "step_ms_from_steps": dist([w["step_ms_from_steps"] for w in active]),
            "mean_acceptance_length": dist([w["mean_acceptance_length_avg"] for w in active]),
            "kv_usage_pct": dist([w["kv_usage_pct_avg"] for w in active]),
            "estimated_context_tokens": dist([w["estimated_context_tokens"] for w in active]),
            "best_worst_spread": dist([w["best_hit"] - w["worst_hit"] for w in active]),
            "median_layer_hit": dist([w["median_layer_hit"] for w in active]),
        },
        "first_quartile": {
            "n": len(first_q),
            "hit_mean": safe_mean([w["hit"] for w in first_q]),
            "miss_mean": safe_mean([w["misses_per_layer_step"] for w in first_q]),
            "gen_tps_mean": safe_mean([w["gen_tps_avg"] for w in first_q]),
            "kv_usage_mean": safe_mean([w["kv_usage_pct_avg"] for w in first_q]),
        },
        "last_quartile": {
            "n": len(last_q),
            "hit_mean": safe_mean([w["hit"] for w in last_q]),
            "miss_mean": safe_mean([w["misses_per_layer_step"] for w in last_q]),
            "gen_tps_mean": safe_mean([w["gen_tps_avg"] for w in last_q]),
            "kv_usage_mean": safe_mean([w["kv_usage_pct_avg"] for w in last_q]),
        },
        "windows": active,
    }


def fit_linear(points, features):
    # points: list dict with y and feature keys. features excludes intercept.
    y = np.array([p["step_ms"] for p in points], dtype=float)
    X = np.ones((len(points), 1 + len(features)), dtype=float)
    for j, f in enumerate(features, start=1):
        X[:, j] = np.array([p.get(f, 0.0) for p in points], dtype=float)
    coef, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) if len(y) > 1 else 0.0
    r2 = 1 - ss_res / ss_tot if ss_tot else 1.0
    rmse = math.sqrt(ss_res / max(1, len(y) - len(coef))) if len(y) > len(coef) else math.sqrt(ss_res / max(1, len(y)))
    names = ["intercept"] + features
    return {
        "features": features,
        "coef": {names[i]: float(coef[i]) for i in range(len(names))},
        "n": len(points), "rank": int(rank), "r2": r2, "rmse_ms": rmse,
        "points": [{**p, "pred_ms": float(pred[i]), "resid_ms": float(y[i] - pred[i])} for i, p in enumerate(points)],
    }


def cost_model(k2log):
    ctx = json.load(open(CTX_SUMMARY))
    points = []
    # Matched K=1 context curve. Use 1.8 accepted tokens/step per task note; 512K summary acceptance=0.993 is contaminated.
    for label, slots, ctx_tokens in [("256K", 7360, 262144), ("512K", 5792, 524288), ("1M", 2672, 1048576)]:
        tok_s = float(ctx[label]["decode_median"])
        miss = float(ctx[label]["misses_mean"])
        step_ms = DEFAULT_ACCEPT_K1 * 1000.0 / tok_s
        points.append({"name": f"context-{label}-K1", "kind": "K1-context", "step_ms": step_ms, "misses": miss,
                       "k2": 0.0, "context_tokens": ctx_tokens, "tok_s": tok_s, "accept": DEFAULT_ACCEPT_K1})
    # E1 v2 decode wall from FACTS/receipts note: 40.1 ms/step at ~4.4 misses. Corrected aggregate GPU file has broader attribution; keep FACTS point.
    points.append({"name": "E1-v2-nsight-K1", "kind": "K1-profile", "step_ms": 40.1, "misses": 4.4, "k2": 0.0,
                   "context_tokens": 524288, "tok_s": DEFAULT_ACCEPT_K1 * 1000 / 40.1, "accept": DEFAULT_ACCEPT_K1})
    # Historical Sep 9 matched 512K points from the lane recap. These are clean decode probes, not the Sep 13 contaminated 512K acceptance counters.
    points.append({"name": "sep9-512K-K1", "kind": "K1-sep9", "step_ms": DEFAULT_ACCEPT_K1 * 1000.0 / 45.747,
                   "misses": 4.236, "k2": 0.0, "context_tokens": 524288, "tok_s": 45.747, "accept": DEFAULT_ACCEPT_K1})
    points.append({"name": "sep9-512K-K2", "kind": "K2-sep9", "step_ms": 2.39 * 1000.0 / 47.127,
                   "misses": 6.02, "k2": 1.0, "context_tokens": 524288, "tok_s": 47.127, "accept": 2.39,
                   "stable_decode": 1.0, "clean_decode": 1.0})
    # K2 active windows from log. For speed modeling, prefer accepted-tokens/generation throughput when available;
    # raw 20s slot-cache step counters include startup/idle/prompt gaps in bad windows.
    for i, w in enumerate(k2log["windows"]):
        step_ms = w.get("step_ms_from_accept_gen") or w.get("step_ms_from_steps")
        if step_ms is None:
            continue
        gen = w.get("gen_tps_avg") or 0.0
        accept = w.get("mean_acceptance_length_avg") or DEFAULT_ACCEPT_K2
        miss = w["misses_per_layer_step"]
        # Broad non-startup filter kept for diagnostics.
        stable = (
            35.0 <= gen <= 60.0
            and 35.0 <= step_ms <= 70.0
            and (w.get("kv_usage_pct_avg") or 0.0) >= 20.0
        )
        # Strict decode-only fit requested here: agent-traffic K2 windows in the observed good band,
        # with long-context KV residency and sane accepted-token accounting.
        clean = (
            41.0 <= gen <= 53.0
            and 4.55 <= miss <= 5.10
            and 2.20 <= accept <= 2.80
            and 40.0 <= step_ms <= 60.0
            and (w.get("kv_usage_pct_avg") or 0.0) >= 20.0
        )
        points.append({"name": f"k2log-{i:03d}", "kind": "K2-live", "step_ms": step_ms,
                       "misses": miss, "k2": 1.0, "stable_decode": 1.0 if stable else 0.0,
                       "clean_decode": 1.0 if clean else 0.0,
                       "context_tokens": w.get("estimated_context_tokens") or 0.0,
                       "tok_s": gen, "accept": accept})
    for p in points:
        p["ctx256"] = (p.get("context_tokens") or 0.0) / 262144.0
    fit_simple = fit_linear(points, ["misses"])
    fit_k2 = fit_linear(points, ["misses", "k2"])
    fit_ctx = fit_linear(points, ["misses", "k2", "ctx256"])
    broad_points = [p for p in points if p["k2"] == 0.0 or p.get("stable_decode") == 1.0]
    clean_points = [p for p in points if p["k2"] == 0.0 or p.get("clean_decode") == 1.0]
    task_band_points = [p for p in points if p["k2"] == 0.0 or p.get("kind") == "K2-sep9" or (
        p.get("kind") == "K2-live" and 41.0 <= p.get("tok_s", 0.0) <= 53.0
        and 4.55 <= p.get("misses", 0.0) <= 5.10 and 2.20 <= p.get("accept", 0.0) <= 2.80
        and 40.0 <= p.get("step_ms", 0.0) <= 60.0
    )]
    fit_broad_k2 = fit_linear(broad_points, ["misses", "k2"])
    fit_broad_ctx = fit_linear(broad_points, ["misses", "k2", "ctx256"])
    fit_clean_k2 = fit_linear(clean_points, ["misses", "k2"])
    fit_clean_ctx = fit_linear(clean_points, ["misses", "k2", "ctx256"])
    fit_task_band_k2 = fit_linear(task_band_points, ["misses", "k2"])
    # K1-only fit for daily K=1 lever estimates.
    k1_points = [p for p in points if p["k2"] == 0.0]
    fit_k1 = fit_linear(k1_points, ["misses"])
    # Effective C2C bandwidth from copy bucket.
    slot_bytes = k2log.get("slot_bytes") or {}
    per_slot_bytes = slot_bytes.get("mean_bytes") or (19.0e6)
    bytes_per_miss_layer_step = per_slot_bytes * N_MOE_LAYERS
    bw_from_e1 = (4.4 * bytes_per_miss_layer_step) / (18.0 / 1000.0)  # bytes/s; 18 ms is the C2C-byte portion of E1 row-copy bucket.
    copy_ms_per_miss = bytes_per_miss_layer_step / (bw_from_e1 / 1000.0)
    clean_slope = fit_clean_k2["coef"].get("misses")
    bw_from_clean_slope = bytes_per_miss_layer_step / (clean_slope / 1000.0) if clean_slope else None
    copy_ms_low_400 = bytes_per_miss_layer_step / (400.0e9 / 1000.0)
    copy_ms_high_fit = max(fit_k1["coef"].get("misses", copy_ms_per_miss), fit_clean_ctx["coef"].get("misses", copy_ms_per_miss))
    return {
        "points": points,
        "fit_all_simple": fit_simple,
        "fit_all_with_k2_intercept": fit_k2,
        "fit_all_with_context": fit_ctx,
        "fit_broad_decode_with_k2_intercept": fit_broad_k2,
        "fit_broad_decode_with_context": fit_broad_ctx,
        "fit_task_band_with_k2_intercept": fit_task_band_k2,
        "fit_clean_decode_with_k2_intercept": fit_clean_k2,
        "fit_clean_decode_with_context": fit_clean_ctx,
        # Back-compatible names now point at the strict clean decode fit.
        "fit_stable_decode_with_k2_intercept": fit_clean_k2,
        "fit_stable_decode_with_context": fit_clean_ctx,
        "fit_k1_only": fit_k1,
        "per_slot_bytes_used": per_slot_bytes,
        "bytes_per_miss_layer_step": bytes_per_miss_layer_step,
        "copy_ms_per_miss_from_e1": copy_ms_per_miss,
        "copy_ms_per_miss_band": {"low": copy_ms_low_400, "central": copy_ms_per_miss, "high": copy_ms_high_fit},
        "effective_c2c_GB_s_from_e1": bw_from_e1 / 1e9,
        "effective_c2c_GB_s_from_clean_fit": bw_from_clean_slope / 1e9 if bw_from_clean_slope else None,
        "ledger_c2c_GB_s": C2C_BW_GB_S_LEDGER,
    }


def expert_slot_byte_sanity():
    elems_w13 = 2 * 2048 * 6144
    elems_w2 = 2048 * 6144
    raw_bytes = (elems_w13 + elems_w2) * 0.5
    scales_per_16 = (elems_w13 + elems_w2) / 16.0
    scales_per_32 = (elems_w13 + elems_w2) / 32.0
    empirical_bytes = 24 * 1024**3 / 1568
    return {
        "raw_nvfp4_bytes": raw_bytes,
        "raw_nvfp4_MB": raw_bytes / 1e6,
        "raw_nvfp4_MiB": raw_bytes / 1024**2,
        "plus_scale_per16_bytes": raw_bytes + scales_per_16,
        "plus_scale_per16_MB": (raw_bytes + scales_per_16) / 1e6,
        "plus_scale_per32_bytes": raw_bytes + scales_per_32,
        "plus_scale_per32_MB": (raw_bytes + scales_per_32) / 1e6,
        "empirical_24GiB_1568_bytes": empirical_bytes,
        "empirical_24GiB_1568_MB": empirical_bytes / 1e6,
        "slots_per_GiB_empirical": 1568 / 24,
        "slots_from_24GiB_raw": 24 * 1024**3 / raw_bytes,
        "slots_from_24GiB_scale16": 24 * 1024**3 / (raw_bytes + scales_per_16),
        "slots_from_24GiB_scale32": 24 * 1024**3 / (raw_bytes + scales_per_32),
    }


def kv_offload_costs(slot_curve, model, contexts=(30000, 60000, 150000, 256000)):
    # Use actual-ish C2C bandwidth from E1; fallback ledger.
    bw_GB_s = model["effective_c2c_GB_s_from_e1"] or C2C_BW_GB_S_LEDGER
    bytes_per_ms = bw_GB_s * 1e9 / 1000.0
    positions_k1 = 2
    full_indexer_layers = 20
    compressed_entry_bytes = (512 + 64) * 2  # task-specified bf16 compressed KV entry
    indexer_fp8_entry_bytes = 132  # actual vLLM default indexer cache: 128 fp8 + 4 scale bytes
    indexer_mxfp4_entry_bytes = 68
    indexer_bf16_entry_bytes = 128 * 2
    # Slot-gain miss delta from proportional 7360 -> 8928.
    prop = {r["total_slots"]: r for r in slot_curve["proportional"]}
    base = prop[7360]
    plus = prop[8928]
    miss_delta = base["misses_per_layer_step_lru"] - plus["misses_per_layer_step_lru"]
    b = model["copy_ms_per_miss_from_e1"]
    slot_save_ms = miss_delta * b
    out = []
    for ctx in contexts:
        selected_entries = min(2048, math.ceil(ctx / 4))  # all requested contexts here hit 2048 cap.
        selected_bytes = positions_k1 * N_LAYERS_ATTN * selected_entries * compressed_entry_bytes
        selected_ms = selected_bytes / bytes_per_ms
        comp_len = math.ceil(ctx / 4)
        idx_fp8_ms = positions_k1 * full_indexer_layers * comp_len * indexer_fp8_entry_bytes / bytes_per_ms
        idx_mxfp4_ms = positions_k1 * full_indexer_layers * comp_len * indexer_mxfp4_entry_bytes / bytes_per_ms
        idx_bf16_compressed_ms = positions_k1 * full_indexer_layers * comp_len * indexer_bf16_entry_bytes / bytes_per_ms
        idx_bf16_uncompressed_ms = positions_k1 * full_indexer_layers * ctx * indexer_bf16_entry_bytes / bytes_per_ms
        actualish_ms = selected_ms + idx_fp8_ms
        fp4_ms = selected_ms + idx_mxfp4_ms
        pessimistic_compressed_bf16_ms = selected_ms + idx_bf16_compressed_ms
        pessimistic_uncompressed_bf16_ms = selected_ms + idx_bf16_uncompressed_ms
        out.append({
            "context_tokens": ctx,
            "selected_entries": selected_entries,
            "slot_miss_delta": miss_delta,
            "slot_save_ms": slot_save_ms,
            "selected_kv_read_ms": selected_ms,
            "indexer_scan_ms_fp8_compressed_actual_vllm": idx_fp8_ms,
            "indexer_scan_ms_mxfp4_compressed": idx_mxfp4_ms,
            "indexer_scan_ms_bf16_compressed_bound": idx_bf16_compressed_ms,
            "indexer_scan_ms_bf16_uncompressed_bound": idx_bf16_uncompressed_ms,
            "kv_cost_ms_actual_fp8_indexer": actualish_ms,
            "kv_cost_ms_mxfp4_indexer": fp4_ms,
            "kv_cost_ms_bf16_compressed_bound": pessimistic_compressed_bf16_ms,
            "kv_cost_ms_bf16_uncompressed_bound": pessimistic_uncompressed_bf16_ms,
            "net_ms_actual_fp8_indexer": slot_save_ms - actualish_ms,
            "net_ms_mxfp4_indexer": slot_save_ms - fp4_ms,
            "net_ms_bf16_compressed_bound": slot_save_ms - pessimistic_compressed_bf16_ms,
            "net_ms_bf16_uncompressed_bound": slot_save_ms - pessimistic_uncompressed_bf16_ms,
        })
    return {"bw_GB_s": bw_GB_s, "positions_per_step": positions_k1, "full_indexer_layers": full_indexer_layers, "rows": out}


def tok_s_from_step(step_ms, accept=DEFAULT_ACCEPT_K1):
    return accept * 1000.0 / step_ms


def build_lever_table(results):
    ctx = json.load(open(CTX_SUMMARY))
    baseline_tok = float(ctx["256K"]["decode_median"])
    baseline_step = DEFAULT_ACCEPT_K1 * 1000 / baseline_tok
    cm = results["cost_model"]
    b = cm["copy_ms_per_miss_from_e1"]
    band = cm.get("copy_ms_per_miss_band") or {"low": b, "central": b, "high": b}
    slot_curve = results["slot_curve"]
    prop = {r["total_slots"]: r for r in slot_curve["proportional"]}
    miss_7360 = prop[7360]["misses_per_layer_step_lru"]
    miss_8928 = prop[8928]["misses_per_layer_step_lru"]
    sim_delta_1568 = miss_7360 - miss_8928
    live_k1_miss = float(ctx["256K"]["misses_mean"])
    hideable_compute_ms = 9.41 + 5.38  # E1 v2 dense GEMM + routed MoE GEMM per full step.

    def tok_for_save(save_ms: float, accept: float = DEFAULT_ACCEPT_K1) -> float:
        return tok_s_from_step(max(1e-6, baseline_step - save_ms), accept=accept)

    def fmt_trip(vals, nd=2, signed=True):
        out = []
        for v in vals:
            out.append(f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}")
        return " / ".join(out)

    def add_ranges(row, save_triplet):
        toks = [tok_for_save(s) for s in save_triplet]
        deltas = [t - baseline_tok for t in toks]
        row["ms_delta_range"] = fmt_trip(save_triplet)
        row["pred_tok_s_range"] = fmt_trip(toks, signed=False)
        row["delta_tok_s_range"] = fmt_trip(deltas)
        row["ms_delta"] = float(save_triplet[1])
        row["pred_tok_s"] = float(toks[1])
        row["delta_tok_s"] = float(deltas[1])
        return row

    levers = []
    pf = results["prefetch"]
    perfect_copy_trip = [live_k1_miss * band["low"], live_k1_miss * band["central"], live_k1_miss * band["high"]]
    perfect_save_trip = [min(x, hideable_compute_ms) for x in perfect_copy_trip]
    levers.append(add_ranges({
        "lever": "perfect one-step known-route prefetch (upper bound)",
        "miss_delta": live_k1_miss,
        "miss_delta_range": f"{live_k1_miss:.3f} / {live_k1_miss:.3f} / {live_k1_miss:.3f}",
        "mechanism_assumption": "Route for step t+1 is known early enough to issue all currently missing same-layer expert fills during step t; only dense+MoE compute (14.79 ms/step) is counted as safely hideable.",
        "note": f"copy time range {perfect_copy_trip[0]:.2f}/{perfect_copy_trip[1]:.2f}/{perfect_copy_trip[2]:.2f} ms; hideable cap {hideable_compute_ms:.2f} ms; perfect coverage {pf['perfect_prefetch_coverage_fraction_of_misses']:.5f} after the first trace step.",
    }, perfect_save_trip))

    levers.append(add_ranges({
        "lever": "previous-step same-layer predictor prefetch",
        "miss_delta": pf["prev_predictor_covers_actual_misses_per_layer_step"],
        "miss_delta_range": "0.000 / 0.000 / 0.000",
        "mechanism_assumption": "Predict step t+1 with step t's same-layer expert set under the live LRU cache state.",
        "note": f"direct miss-coverage fraction {pf['prev_predictor_coverage_fraction_of_next_misses']:.6f}; blind prefetch attempts {pf['prev_predictor_blind_prefetches_per_layer_step']:.2f}/layer-step, {pf['prev_predictor_not_requested_fraction_of_blind_prefetches']:.1%} not requested and {pf['prev_predictor_hit_anyway_fraction_of_blind_prefetches']:.1%} already-hit; no miss-value waste {pf['prev_predictor_no_miss_value_fraction_of_blind_prefetches']:.1%}.",
    }, [0.0, 0.0, 0.0]))

    kv_rows = {r["context_tokens"]: r for r in results["kv_offload"]["rows"]}
    for c in [30000, 60000, 150000, 256000]:
        row = kv_rows[c]
        save_trip = [row["net_ms_bf16_uncompressed_bound"], row["net_ms_actual_fp8_indexer"], row["net_ms_mxfp4_indexer"]]
        levers.append(add_ranges({
            "lever": f"KV offload frees +1,568 expert slots @ {c//1000}K ctx",
            "miss_delta": sim_delta_1568,
            "miss_delta_range": f"{sim_delta_1568:.3f} / {sim_delta_1568:.3f} / {sim_delta_1568:.3f}",
            "mechanism_assumption": "Move 24 GiB KV to Grace and spend the HBM on the 8,928-slot proportional map; decode reads sparse DSA selected KV over C2C plus the compressed indexer cache.",
            "note": f"slot save {row['slot_save_ms']:.2f} ms; central fp8-indexer KV cost {row['kv_cost_ms_actual_fp8_indexer']:.2f} ms; range low=bf16 uncompressed scan, high=MXFP4 indexer.",
        }, save_trip))

    levers.append(add_ranges({
        "lever": "masked-row-copy launch coalescing / scale-fuse",
        "miss_delta": 0.0,
        "miss_delta_range": "0.000 / 0.000 / 0.000",
        "mechanism_assumption": "Remove only launch/empty/scale bookkeeping around masked row copies; C2C bytes and expert misses are unchanged.",
        "note": "E1 v2 row-copy bucket says ~18 ms/step is bytes and only ~2 ms/step is launch/empty overhead, so 2 ms is an upper bound, not a planning central case.",
    }, [0.0, 1.0, 2.0]))

    # K2 threshold.
    k2 = json.load(open(K2_SUMMARY))["speed"]
    k1_gate = k2["k1_256k"]["c1_median_tok_s"]
    k2_gate = k2["k2_256k"]["c1_median_tok_s"]
    target = k1_gate * 1.05
    accept_low, accept_central, accept_high = 2.20, 2.39, 2.80
    def needed_ms(acc):
        return acc * 1000.0 * (1.0 / k2_gate - 1.0 / target)
    needed_ms_trip = [needed_ms(accept_low), needed_ms(accept_central), needed_ms(accept_high)]
    needed_miss_trip = [needed_ms_trip[0] / band["high"], needed_ms_trip[1] / band["central"], needed_ms_trip[2] / band["low"]]
    levers.append({
        "lever": "K=2 re-gate threshold after any miss reducer",
        "miss_delta": needed_miss_trip[1],
        "miss_delta_range": fmt_trip(needed_miss_trip, nd=3, signed=False),
        "ms_delta": needed_ms_trip[1],
        "ms_delta_range": fmt_trip(needed_ms_trip),
        "pred_tok_s": target,
        "pred_tok_s_range": f"{k2_gate:.2f} now / {target:.2f} bar / >{target:.2f} pass",
        "delta_tok_s": target - baseline_tok,
        "delta_tok_s_range": f"{k2_gate - baseline_tok:+.2f} now / {target - baseline_tok:+.2f} bar / mechanism-dependent",
        "mechanism_assumption": "K2 quality remains equivalent and acceptance stays in the 2.2–2.8 accepted-token/step band; only miss-copy critical-path time changes.",
        "note": f"Need {needed_miss_trip[0]:.3f}/{needed_miss_trip[1]:.3f}/{needed_miss_trip[2]:.3f} fewer misses/layer-step (low/central/high uncertainty) to make {k2_gate:.2f} tok/s clear the +5% bar {target:.2f} tok/s.",
    })
    return {
        "baseline_tok_s": baseline_tok,
        "baseline_step_ms": baseline_step,
        "hideable_compute_ms": hideable_compute_ms,
        "copy_ms_per_miss_band": band,
        "k2_threshold": {"k1_gate_tok_s": k1_gate, "k2_gate_tok_s": k2_gate, "target_tok_s": target,
                         "needed_ms_range": needed_ms_trip, "needed_miss_range": needed_miss_trip},
        "levers": levers,
    }


def fmt(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        if math.isnan(x): return "nan"
        return f"{x:.{nd}f}"
    return str(x)


def write_markdown(results):
    FINDINGS.mkdir(parents=True, exist_ok=True)
    p = FINDINGS / "b-trace-model.md"
    pol = results["policy_reproduction"]["summary"]
    blog = {
        "mean_static_oracle": 0.6325043065112578,
        "mean_lru": 0.7193080206695125,
        "mean_hybrid_pin50": 0.7095347487456002,
        "mean_hybrid_pin75": 0.6973041638583091,
        "mean_crosslayer_overlap": 0.03103652686611572,
        "mean_temporal_overlap": 0.2705852617389351,
        "miss_lru": 2.2455358346439,
    }
    cm = results["cost_model"]
    k2log = results["k2_log"]
    lever = results["lever_table"]
    lines = []
    lines.append("# GLM-5.3 GB300 slot-cache speed levers — trace simulation + calibrated cost model")
    lines.append("")
    lines.append("Generated by `findings/scripts/trace_model.py` from local trace/log/receipt files. Numbers are tagged as **measured** when they come from receipts/logs and **modeled** when they are derived from the replay/cost model.")
    lines.append("")
    lines.append("## Executive findings")
    lines.append("")
    lines.append(f"- **Top lever is one-step/known-route prefetch**, but only in the perfect/lookahead form: measured K1 256K daily misses ({json.load(open(CTX_SUMMARY))['256K']['misses_mean']:.2f}/layer-step) correspond to ~{json.load(open(CTX_SUMMARY))['256K']['misses_mean']*cm['copy_ms_per_miss_from_e1']:.1f} ms/step of C2C row-copy time, inside the conservative 14.79 ms/step dense+MoE hide window. Range after the slope/hide cap: {lever['levers'][0]['delta_tok_s_range']} Δtok/s. This is an upper bound, not an implementation claim.")
    lines.append(f"- **Previous-step temporal prefetch is not a lever against LRU.** The frozen trace temporal overlap is {pol['mean_temporal_overlap']:.4f}, but direct replay gives {results['prefetch']['prev_predictor_coverage_fraction_of_next_misses']:.6f} coverage of actual next-step misses: those previous-step experts are already resident. Blind DMA would copy ~{results['prefetch']['prev_predictor_blind_prefetches_per_layer_step']:.2f} experts/layer-step; {results['prefetch']['prev_predictor_not_requested_fraction_of_blind_prefetches']:.1%} are not requested next step and {results['prefetch']['prev_predictor_hit_anyway_fraction_of_blind_prefetches']:.1%} are already hits.")
    lines.append("- **KV offload for slots is marginal at 256K daily.** +1,568 slots saves about " + f"{results['kv_offload']['rows'][-1]['slot_save_ms']:.2f} ms/step in miss copies, but decode KV over C2C costs ~{results['kv_offload']['rows'][-1]['kv_cost_ms_actual_fp8_indexer']:.2f} ms/step with the actual vLLM fp8 compressed-indexer model. At 30K/60K it is a small central win; at 150K/256K it is wash-to-loss unless the indexer path is cheaper than modeled.")
    lines.append(f"- **K=2 needs only {lever['k2_threshold']['needed_miss_range'][0]:.3f}–{lever['k2_threshold']['needed_miss_range'][2]:.3f} fewer misses/layer-step** (central {lever['k2_threshold']['needed_miss_range'][1]:.3f}, ≈{lever['k2_threshold']['needed_ms_range'][1]:.2f} ms/step) to clear the +5% bar if acceptance stays like the fair gate.")
    lines.append("")

    lines.append("## 1. Reproduced trace policy table")
    lines.append("")
    lines.append("Replay uses the recipe analyzer semantics: decode tokens only, MoE layers 3–77, per-layer `slots-7360-ctx256k.json`, and the same `T//8000` subsample for LRU/hybrid/cross/temporal metrics.")
    lines.append("")
    lines.append("| metric | reproduced | blog/receipt | diff |")
    lines.append("|---|---:|---:|---:|")
    rows = [
        ("static oracle hit", pol["mean_static_oracle"], blog["mean_static_oracle"]),
        ("LRU hit", pol["mean_lru"], blog["mean_lru"]),
        ("hybrid pin-50 hit", pol["mean_hybrid_pin50"], blog["mean_hybrid_pin50"]),
        ("hybrid pin-75 hit", pol["mean_hybrid_pin75"], blog["mean_hybrid_pin75"]),
        ("temporal overlap", pol["mean_temporal_overlap"], blog["mean_temporal_overlap"]),
        ("cross-layer overlap", pol["mean_crosslayer_overlap"], blog["mean_crosslayer_overlap"]),
        ("LRU misses/layer-step", pol["misses_per_layer_step"]["lru"], blog["miss_lru"]),
    ]
    for name, got, exp in rows:
        lines.append(f"| {name} | {got:.6f} | {exp:.6f} | {got-exp:+.6f} |")
    lines.append("")
    lines.append(f"Trace shape: {pol['decode_tokens']:,} decode tokens × {pol['moe_layers']} MoE layers × top-{TOPK}; total slots {pol['total_slots']:,}. Cross-layer overlap remains chance-like (random baseline {pol['random_overlap_baseline']:.5f}).")
    lines.append("")

    lines.append("## 2. One-step-ahead prefetch bounds")
    lines.append("")
    pf = results["prefetch"]
    lines.append("| variant | synchronous misses/layer-step | async/prefetch experts/layer-step | coverage / waste | modeled 256K K1 Δtok/s range |")
    lines.append("|---|---:|---:|---|---:|")
    lines.append(f"| no prefetch LRU baseline (trace) | {pf['baseline_misses_per_layer_step']:.3f} | 0 | exact full-trace replay at 7,360 slots; hit {pf['baseline_hit']:.4f} | 0 |")
    lines.append(f"| perfect one-step lookahead | {pf['perfect_remaining_sync_misses_per_layer_step']:.4f} | {pf['perfect_prefetch_bytes_misses_per_layer_step']:.3f} | covers {pf['perfect_prefetch_coverage_fraction_of_misses']:.5f} of misses after first step; same bytes, hidden if overlap succeeds | {lever['levers'][0]['delta_tok_s_range']} |")
    lines.append(f"| previous-step predictor, cache-aware | {pf['baseline_misses_per_layer_step']:.3f} | {pf['prev_predictor_cache_aware_prefetches_per_layer_step']:.4f} | covers {pf['prev_predictor_coverage_fraction_of_next_misses']:.6f} of actual next-step misses; no-op under LRU | +0.00 / +0.00 / +0.00 |")
    lines.append(f"| previous-step predictor, blind DMA | {pf['baseline_misses_per_layer_step']:.3f} | {pf['prev_predictor_blind_prefetches_per_layer_step']:.3f} | {pf['prev_predictor_hit_anyway_per_layer_step']:.3f} hit-anyway + {pf['prev_predictor_not_requested_per_layer_step']:.3f} not requested; {pf['prev_predictor_no_miss_value_fraction_of_blind_prefetches']:.1%} has no miss-reduction value | ≤0 |")
    lines.append("")
    lines.append("Interpretation: the 0.2706 temporal overlap is real, but it is almost entirely **hit-anyway** under LRU. It is evidence for demand-cache locality, not evidence that copying the previous step's set reduces future misses. K=2/live route-lookahead is different because it can know the actual next verify-step routes before demand.")
    lines.append("")
    lines.append("K=2 live windows have a median ~" + f"{k2log['distributions']['misses_per_layer_step']['median']:.2f} misses/layer-step and {k2log['distributions']['step_ms_from_steps']['median']:.1f} ms/step. The same perfect-prefetch upper bound is larger in ms/step for K=2 because there are more verify positions and more misses; previous-step prefetch remains a no-op.")
    lines.append("")

    lines.append("## 3. Slot-budget curve")
    lines.append("")
    lines.append("Curves below use an exact full-trace stack-distance replay so marginal slot gains are not distorted by the recipe analyzer's speed subsample. That is why 7,360 slots shows 0.802 exact LRU hit here while the reproduced blog table above shows 0.719 on the `T//8000` subsample. Lever sizing is finally calibrated to measured live misses, not the exact replay miss level.")
    lines.append("")
    for variant in ["proportional", "flat"]:
        lines.append(f"### {variant} capacity shape")
        lines.append("")
        lines.append("| total slots | LRU hit | misses/layer-step | static hit | marginal hit / +1k slots |")
        lines.append("|---:|---:|---:|---:|---:|")
        prev = None
        for r in results["slot_curve"][variant]:
            if prev is None:
                marg = "—"
            else:
                dh = r["mean_lru_hit"] - prev["mean_lru_hit"]
                ds = r["total_slots"] - prev["total_slots"]
                marg = f"{(dh / ds * 1000):.4f}"
            lines.append(f"| {r['total_slots']:,} | {r['mean_lru_hit']:.4f} | {r['misses_per_layer_step_lru']:.3f} | {r['mean_static_hit']:.4f} | {marg} |")
            prev = r
        lines.append("")
    sanity = results["slot_byte_sanity"]
    lines.append("Slot-byte sanity: model dimensions give raw NVFP4 expert bytes " + f"{sanity['raw_nvfp4_MB']:.1f} MB; with 1 scale byte per 32 or 16 weights this is {sanity['plus_scale_per32_MB']:.1f}–{sanity['plus_scale_per16_MB']:.1f} MB. The empirical +1,568 slots/24 GiB assumption implies {sanity['empirical_24GiB_1568_MB']:.1f} MB/slot, so +1,568 is optimistic relative to a full weight+scale accounting; I keep it because it is the measured context-lane tradeoff requested in the task.")
    lines.append("")

    lines.append("## 4. KV-offload decode cost")
    lines.append("")
    lines.append("vLLM source check: decode does **not** gather full KV for sparse attention. `DeepseekV4FlashMLAAttention._forward_decode` passes `extra_indices_in_kvcache=topk_indices` and `extra_topk_length` to `flash_mla_with_kvcache`; the C4A indexer separately scans a quantized index-key cache (`SparseAttnIndexer`, default FP8 128B+4B scale per compressed entry, MXFP4 68B opt-in). Thus I model both actual-ish compressed/indexed reads and a pessimistic bf16 scan bound.")
    lines.append("")
    lines.append("| context | slot save ms | selected KV C2C ms | fp8 indexer scan ms | net ms fp8 | net ms MXFP4 | net ms bf16-compressed | net ms bf16-uncompressed | verdict |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in results["kv_offload"]["rows"]:
        verdict = "small win" if r["net_ms_actual_fp8_indexer"] > 0.3 else ("wash" if r["net_ms_actual_fp8_indexer"] > -0.3 else "loss")
        lines.append(f"| {r['context_tokens']:,} | {r['slot_save_ms']:.2f} | {r['selected_kv_read_ms']:.2f} | {r['indexer_scan_ms_fp8_compressed_actual_vllm']:.2f} | {r['net_ms_actual_fp8_indexer']:+.2f} | {r['net_ms_mxfp4_indexer']:+.2f} | {r['net_ms_bf16_compressed_bound']:+.2f} | {r['net_ms_bf16_uncompressed_bound']:+.2f} | {verdict} |")
    lines.append("")

    lines.append("## 5. Calibrated cost model")
    lines.append("")
    lines.append(f"Per-slot bytes from live log cache-build lines: mean {k2log['slot_bytes']['mean_MB_decimal']:.2f} MB ({k2log['slot_bytes']['n']} layers parsed). One `misses/layer-step` therefore moves ~{cm['bytes_per_miss_layer_step']/1e9:.2f} GB across all 75 MoE layers. E1 row-copy bucket implies effective C2C bandwidth {cm['effective_c2c_GB_s_from_e1']:.1f} GB/s, matching ledger C2C probes ({cm['ledger_c2c_GB_s']:.1f} GB/s); the strict clean-window slope implies {cm['effective_c2c_GB_s_from_clean_fit']:.1f} GB/s. Physical copy slope used for central lever sizing is {cm['copy_ms_per_miss_from_e1']:.2f} ms per misses/layer-step; uncertainty band {cm['copy_ms_per_miss_band']['low']:.2f}/{cm['copy_ms_per_miss_band']['central']:.2f}/{cm['copy_ms_per_miss_band']['high']:.2f} ms.")
    lines.append("")
    for fit_name, label in [("fit_k1_only", "K1-only measured decode points"), ("fit_task_band_with_k2_intercept", "K1 + K2 task-band windows (no KV>=20 filter)"), ("fit_clean_decode_with_k2_intercept", "K1 + strict clean K2 decode windows"), ("fit_clean_decode_with_context", "K1 + strict clean K2 + context")]:
        fit = cm[fit_name]
        co = fit["coef"]
        co_s = ", ".join(f"{k}={v:.3f}" for k, v in co.items())
        lines.append(f"- {label}: n={fit['n']}, R²={fit['r2']:.3f}, RMSE={fit['rmse_ms']:.2f} ms, {co_s}.")
    allfit = cm["fit_all_with_k2_intercept"]
    broad = cm["fit_broad_decode_with_k2_intercept"]
    lines.append(f"- Broad non-startup windows: n={broad['n']}, R²={broad['r2']:.3f}, RMSE={broad['rmse_ms']:.2f} ms; useful as a residual check, not for lever sizing.")
    lines.append(f"- All parsed active K2 windows without warmup/filtering: n={allfit['n']}, R²={allfit['r2']:.3f}, RMSE={allfit['rmse_ms']:.2f} ms; this is the warning fit because startup/idle/prompt gaps and acceptance variance swamp the miss signal.")
    lines.append("")
    clean = cm["fit_clean_decode_with_k2_intercept"]
    sep9_res = [p['resid_ms'] for p in clean['points'] if p['name'] == 'sep9-512K-K2']
    live_res = [p['resid_ms'] for p in clean['points'] if p.get('kind') == 'K2-live']
    lines.append("Residual structure: K1 context/profile points are within ~1–2 ms except the 1M context point; the historical Sep 9 K2 point sits " + (f"{sep9_res[0]:+.2f} ms" if sep9_res else "n/a") + " from the strict fit; clean K2 live residuals span " + (f"{min(live_res):+.2f} to {max(live_res):+.2f} ms" if live_res else "n/a") + ". The remaining structure tracks acceptance/window composition more than context length, so I do not use a separate context coefficient for central estimates.")
    lines.append("")

    lines.append("## 6. Lever table at 256K-K1 daily profile")
    lines.append("")
    lines.append(f"Baseline: **measured** 256K K1 median {lever['baseline_tok_s']:.2f} tok/s, **modeled** {lever['baseline_step_ms']:.2f} ms/step at {DEFAULT_ACCEPT_K1:.2f} accepted tokens/step. Ranges are low / central / high; central uses the E1 physical C2C slope, not the noisy all-window regression.")
    lines.append("")
    lines.append("| lever | miss Δ/layer-step range | modeled ms Δ range | predicted tok/s range | Δtok/s range | mechanism assumption / caveat |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for row in lever["levers"]:
        note = row["note"]
        assumption = row.get("mechanism_assumption", "")
        miss_s = row.get('miss_delta_range') or f"{row['miss_delta']:.3f}"
        ms_s = row.get('ms_delta_range') or f"{row['ms_delta']:+.2f}"
        pred_s = row.get('pred_tok_s_range') or f"{row['pred_tok_s']:.2f}"
        delta_s = row.get('delta_tok_s_range') or f"{row['delta_tok_s']:+.2f}"
        lines.append(f"| {row['lever']} | {miss_s} | {ms_s} | {pred_s} | {delta_s} | {assumption} {note} |")
    lines.append("")

    lines.append("## 7. k2gate.log real-agent-traffic telemetry")
    lines.append("")
    d = k2log["distributions"]
    lines.append(f"Parsed {k2log['active_window_count']} active SLOT_CACHE windows, {k2log['throughput_count']} throughput lines, {k2log['spec_count']} speculative-decode metric lines. The log does not contain TTFT/request-latency fields; only prompt/generation throughput series are present.")
    lines.append("")
    lines.append("| metric | median | p05 | p95 | min | max |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for key, label in [("hit", "SLOT_CACHE hit"), ("misses_per_layer_step", "misses/layer-step"), ("gen_tps", "generation tok/s"), ("step_ms_from_steps", "step ms from steps"), ("mean_acceptance_length", "mean accept length"), ("kv_usage_pct", "GPU KV usage %"), ("best_worst_spread", "best-worst layer hit spread")]:
        x = d[key]
        lines.append(f"| {label} | {fmt(x['median'])} | {fmt(x['p05'])} | {fmt(x['p95'])} | {fmt(x['min'])} | {fmt(x['max'])} |")
    lines.append("")
    fq = k2log["first_quartile"]; lq = k2log["last_quartile"]
    lines.append(f"Session drift: first quartile hit {fq['hit_mean']:.3f}, misses {fq['miss_mean']:.2f}, KV usage {fq['kv_usage_mean']:.1f}%; last quartile hit {lq['hit_mean']:.3f}, misses {lq['miss_mean']:.2f}, KV usage {lq['kv_usage_mean']:.1f}%. The tail includes degradation to hit {d['hit']['min']:.3f}/misses {d['misses_per_layer_step']['max']:.2f} as long prompts grew context.")
    lines.append("")
    lines.append("Contrast: frozen probe replay LRU at 7,360 slots is hit 0.719 / 2.25 misses, K1 context-lane live is hit 0.61 / 3.12 misses, and real agent K2 traffic is median hit " + f"{d['hit']['median']:.3f} / {d['misses_per_layer_step']['median']:.2f} misses. Agent traffic is harder because it includes K=2 verify positions, long prompts, and live serving effects not present in decode-only replay.")
    lines.append("")

    lines.append("## 8. Frozen gate designs")
    lines.append("")
    lines.append("### Gate A — route-lookahead prefetch prototype (top lever)")
    lines.append("")
    lines.append("- One axis: enable the prefetch mechanism only (same image/tag, same 256K geometry, same `slots-7360-ctx256k.json`, same K as control). No slot-budget, K, prefix-cache, or routing-policy change in the same window.")
    lines.append("- Implementation receipts must prove the mechanism is using known step t+1 routes, not the previous-step predictor: log prefetch issued, demand-miss avoided, hit-anyway, not-requested, and eviction-caused-miss counters per layer. Target counter: previous-step coverage stays 0; route-lookahead demand-miss coverage is the measured value.")
    lines.append("- Speed gate: existing `2026-09-13-k2-fair-gate-needle-trace/scripts/speed_reps.py`, paired 8 prompts × 512 tokens × 3 scored reps (+ discarded warm) against the incumbent; bootstrap paired CI. Pass for a mechanical K1 prefetch candidate = lower CI >0 and mean ≥+3%; promotion-style bar = mean ≥+5% with no prompt-class loss cluster.")
    lines.append("- Hygiene/quality: greedy self-repeat 20/20 via `greedy_equiv.py`; routing-equivalence / teacher-forced margin check if prefetch changes ordering or CUDA graph capture; task ladder 200 with no regression; needle ladder only if context/KV path is touched.")
    lines.append("- Receipts: speed JSON/log, SLOT_CACHE + prefetch counters, metrics snapshot, sanitized inspect, early hook logs proving `EXACT_PIN` and `SLOT_CACHE`, bootstrap CI JSON/notebook. Stop-not-rm the candidate; no :30003 restore/promotion.")
    lines.append("")
    lines.append("### Gate B — K=2 re-gate after a miss reducer is frozen")
    lines.append("")
    lines.append("- One axis: change only `num_speculative_tokens` K=1→K=2 on top of the already-frozen miss reducer (for example Gate A prefetch). Same image, same slot map, same KV budget, same hook hashes, same prompt set.")
    lines.append("- Pre-registered threshold: K2 must clear +5% over the matched K1 control on the same setup. The model says this requires about " + f"{lever['k2_threshold']['needed_miss_range'][0]:.3f}–{lever['k2_threshold']['needed_miss_range'][2]:.3f} fewer misses/layer-step versus the current K2 gate if acceptance remains 2.2–2.8 accepted tokens/step.")
    lines.append("- Speed/telemetry: paired `speed_reps.py` 8×512×3, bootstrap CI, per-prompt class table, per-class acceptance, misses/layer-step, hit rate, prefetch coverage/waste, and step-ms derived both from accepted/gen and SLOT_CACHE step counters.")
    lines.append("- Quality: greedy self-repeat 20/20, cross-lane greedy-equivalence noted as a kernel-shape gate not a quality gate, teacher-forced divergence margins if text differs, and task ladder 200 with no loss cluster. Keep the Sep 13 margin method/receipts as the reference.")
    lines.append("- Receipts and cleanup: sanitized inspect/env (no API key), launch line, hook hashes, speed/margins/task outputs, metrics, logs; stop-not-rm. If the miss reducer is not frozen, do not run this gate because it mixes axes.")
    lines.append("")
    lines.append("KV-offload-for-slots is not frozen as a top 256K speed lever here: the central model is small positive only at 30K/60K and wash-to-loss at 150K/256K. If James wants long-context capacity anyway, it needs its own one-axis capacity gate, not a hidden part of Gate A or B.")
    lines.append("")

    lines.append("## Changes in this continuation")
    lines.append("")
    lines.append("- Refit the decode cost model on strict clean K2 windows, added the Sep 9 K1/K2 512K points, and kept the all-window regression only as a warning about prefill/startup/idle contamination.")
    lines.append("- Recomputed the previous-step same-layer prefetch value directly against the exact LRU trace state: coverage of actual next-step misses is 0.000000; blind previous-step DMA is 100% waste for miss reduction.")
    lines.append("- Replaced point estimates in the lever table with low/central/high ranges, added the K=2 re-gate threshold range, and froze gates only for route-lookahead prefetch plus K=2 after a miss reducer.")
    lines.append("")

    lines.append("## Reproducibility")
    lines.append("")
    lines.append("```bash")
    lines.append("cd /Users/jamesmeadlock/hermes/glm53-speed-research-20260913")
    lines.append("python3 findings/scripts/trace_model.py --write-report")
    lines.append("```")
    lines.append("")
    lines.append("The script caches stack-distance histograms in `findings/scripts/out/stack_hists.npz`; delete it or pass `--refresh-cache` to recompute from `trace-361.i16`.")
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-report", action="store_true")
    ap.add_argument("--refresh-cache", action="store_true")
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print("loading trace", flush=True)
    X, Xd, moe, E, L, K, dec_tokens, prefill_tokens = load_trace()
    base_slots = load_slots("slots-7360-ctx256k.json")
    print(f"trace loaded: decode={dec_tokens} prefill={prefill_tokens} layers={L} moe={len(moe)} topk={K} experts={E}", flush=True)

    policy = reproduce_policy_table(Xd, moe, base_slots)
    print("policy reproduced", policy["summary"], flush=True)

    hists, counts = compute_or_load_stack_hists(Xd, moe, E, refresh=args.refresh_cache)
    budgets = [2672, 5792, 7360, 8928, 10000, 12000]
    slot_curve = {"proportional": [], "flat": []}
    for variant in slot_curve:
        for total in budgets:
            caps = capacities_for_budget(base_slots, moe, total, variant)
            slot_curve[variant].append(evaluate_caps(hists, counts, caps, moe))
    print("slot curve done", flush=True)

    pf = prefetch_metrics(Xd, moe, base_slots)
    print("prefetch done", flush=True)

    k2log = parse_k2_log()
    print("k2 log parsed", k2log["active_window_count"], flush=True)

    cm = cost_model(k2log)
    sanity = expert_slot_byte_sanity()
    kv = kv_offload_costs(slot_curve, cm)

    results = {
        "paths": {
            "trace_dir": str(TRACE_DIR), "k2_log": str(K2_LOG), "ctx_summary": str(CTX_SUMMARY),
            "k2_summary": str(K2_SUMMARY), "e1_buckets": str(E1_BUCKETS),
        },
        "trace": {"decode_tokens": dec_tokens, "prefill_tokens": prefill_tokens, "layers": L, "moe_layers": len(moe), "topk": K, "experts": E},
        "policy_reproduction": policy,
        "slot_curve": slot_curve,
        "prefetch": pf,
        "k2_log": k2log,
        "cost_model": cm,
        "slot_byte_sanity": sanity,
        "kv_offload": kv,
    }
    results["lever_table"] = build_lever_table(results)
    if args.write_report:
        results["findings_file"] = write_markdown(results)
    out = OUTDIR / "trace_model_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else int(o) if isinstance(o, np.integer) else str(o))
    print(f"wrote {out}")
    if args.write_report:
        print(f"wrote {results['findings_file']}")


if __name__ == "__main__":
    main()
