"""E4 adaptive remap helpers. Imported by pin_hot_experts_hook (same dir).

Hot path (E4b D1): graph-safe snap[layer, :T].copy_(ids) — no ring index, no
cursor. Side thread samples the snap every N host steps (execute_model ticks),
updates per-layer EWMA, queues swaps. Swaps run on the main stream between
steps (GPUModelRunner.execute_model, and layer-0 eager _invoke).

D0 is not viable: DeepseekV4Model.topk_indices_buffer is Indexer attention
scratch [max_batched, index_topk], shared across layers, not MoE routed ids.

Ordering (bit-exact): for each swap, copy hot-row -> HBM staging, cold-row ->
hot slot, staging -> cold slot, then flip two entries of the device row_map.
Never flip a layer whose rows are mid-copy. Row-map ADDRESS is fixed (CUDA
graph); contents change between replays.
"""
from __future__ import annotations

import json
import os
import threading
import time

import torch

# Bound at load by pin_hot_experts_hook.bind_adaptive(ns)
H = None


def bind(ns):
    global H
    H = ns


def _log(msg: str) -> None:
    H._LOG(f"ADAPT {msg}")


def _counter() -> str:
    c = getattr(H, "COUNTER", "d1")
    if c == "d0":
        return "d1"
    return c


def init_ring(device) -> None:
    """Preallocate snap (D1) and optional ring. Illegal during capture."""
    if H._ring_ready:
        return
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("PIN_HOT adaptive ring alloc during capture")
    R, Tm, K = H.RING_R, H.RING_TMAX, H.TOPK
    H._tvals = torch.arange(Tm + 1, dtype=torch.int32, device=device)
    H._layer_ids = torch.arange(H.N_MOE_LAYERS, dtype=torch.int32, device=device)
    H._last_ids = torch.full(
        (H.N_MOE_LAYERS, Tm, K), -1, dtype=torch.int32, device=device
    )
    H._last_t = torch.zeros(H.N_MOE_LAYERS, dtype=torch.int32, device=device)
    mode = _counter()
    if mode == "ring":
        H._ring = torch.full((R, Tm, K), -1, dtype=torch.int32, device=device)
        H._ring_t = torch.zeros(R, dtype=torch.int32, device=device)
        H._ring_layer = torch.full((R,), -1, dtype=torch.int32, device=device)
        H._cursor = torch.zeros(1, dtype=torch.int64, device=device)
        H._slot = torch.zeros(1, dtype=torch.int64, device=device)
        H._idpad = torch.full((1, Tm, K), -1, dtype=torch.int32, device=device)
    else:
        H._ring = None
        H._ring_t = None
        H._ring_layer = None
        H._cursor = torch.zeros(1, dtype=torch.int64, device=device)
        H._slot = None
        H._idpad = None
    torch.cuda.synchronize()
    H._ring_ready = True
    bytes_ = H._last_ids.numel() * 4 + H._last_t.numel() * 4
    if H._ring is not None:
        bytes_ += (
            H._ring.numel() * 4
            + H._ring_t.numel() * 4
            + H._ring_layer.numel() * 4
        )
    _log(
        f"counter={mode} TMAX={Tm} K={K} layers={H.N_MOE_LAYERS} "
        f"device_bytes={bytes_} ({bytes_ / 1024**2:.2f} MiB) "
        f"d0=indexer-scratch-not-moe"
    )


def alloc_staging(st) -> dict:
    """One HBM row per leading-E tensor. Reused sequentially (same stream)."""
    out = {}
    n = 0
    for key, t in st.hot.items():
        if t is None:
            continue
        row = torch.empty(t.shape[1:], dtype=t.dtype, device=t.device)
        out[key] = row
        n += int(row.numel() * row.element_size())
    H._staging = out
    H._staging_bytes = n
    _log(f"staging one-row HBM {n / 1024**2:.2f} MiB keys={list(out)}")
    return out


def ring_push(layer_idx: int, ids: torch.Tensor) -> None:
    """Graph-safe record of topk_ids.

    D1 (default): snap[layer, :T].copy_(ids) + last_t[layer] = T.
    No cursor, no index_copy_. T is a Python int (fixed per captured graph).
    """
    if not H._ring_ready or ids is None:
        return
    if isinstance(ids, (tuple, list)):
        ids = ids[0]
    if not torch.is_tensor(ids):
        return
    T = ids.shape[0]
    K = ids.shape[-1]
    if T > H.RING_TMAX or T == 0:
        return
    if layer_idx < 0 or layer_idx >= H.N_MOE_LAYERS:
        return
    ids32 = ids if ids.dtype == torch.int32 else ids.to(dtype=torch.int32)
    mode = _counter()
    if mode != "ring":
        H._last_ids[layer_idx, :T, :K].copy_(ids32)
        H._last_t[layer_idx : layer_idx + 1].copy_(H._tvals[T : T + 1])
        return
    H._idpad[0, :T, :K].copy_(ids32)
    H._slot.copy_(H._cursor)
    H._cursor.add_(1)
    H._slot.remainder_(H.RING_R)
    H._ring.index_copy_(0, H._slot, H._idpad)
    H._ring_t.index_copy_(0, H._slot, H._tvals[T : T + 1])
    H._ring_layer.index_copy_(0, H._slot, H._layer_ids[layer_idx : layer_idx + 1])


def swap_pair(st, hot_eid: int, cold_eid: int, staging=None, misordered: bool = False):
    """Swap currently-hot expert hot_eid with currently-cold cold_eid.

    Default order: row copies, then row-map flip (bit-exact vs full-E).
    misordered=True flips the map first — used by the dry-run to prove the
    check is real (must DIVERGE).
    """
    if staging is None:
        staging = H._staging
    if staging is None:
        raise RuntimeError("PIN_HOT swap_pair: staging not allocated")
    rm = st.row_map
    hr = int(rm[hot_eid].item())
    cr_enc = int(rm[cold_eid].item())
    if hr < 0:
        raise RuntimeError(f"swap_pair: hot_eid={hot_eid} not hot (map={hr})")
    if cr_enc >= 0:
        raise RuntimeError(f"swap_pair: cold_eid={cold_eid} not cold (map={cr_enc})")
    cr = -cr_enc - 1
    return swap_rows_known(st, hot_eid, cold_eid, hr, cr, staging, misordered=misordered)


def swap_rows_known(
    st, hot_eid, cold_eid, hot_row, cold_row, staging, misordered: bool = False
):
    rm = st.row_map
    if misordered:
        # Test hook: flip the map while rows stay put. Must diverge vs full-E.
        rm[hot_eid] = -(cold_row + 1)
        rm[cold_eid] = hot_row
        return {
            "hot_eid": hot_eid,
            "cold_eid": cold_eid,
            "hot_row": hot_row,
            "cold_row": cold_row,
            "misordered": True,
        }
    for key, ht in st.hot.items():
        ct = st.cold.get(key)
        stg = staging.get(key) if staging is not None else None
        if ct is None or stg is None:
            continue
        stg.copy_(ht[hot_row])
        ht[hot_row].copy_(ct[cold_row])
        ct[cold_row].copy_(stg)
    rm[hot_eid] = -(cold_row + 1)
    rm[cold_eid] = hot_row
    if st.hot_eids is not None:
        st.hot_eids[hot_row] = cold_eid
        st.cold_eids[cold_row] = hot_eid
    return {"hot_eid": hot_eid, "cold_eid": cold_eid, "hot_row": hot_row, "cold_row": cold_row}


def apply_pending_swaps() -> int:
    """Enqueue copies + map flips on the current (main) stream. Eager only."""
    if torch.cuda.is_current_stream_capturing():
        return 0
    with H._swap_lock:
        pending = H._pending_swaps
        H._pending_swaps = []
    if not pending:
        return 0
    n = 0
    for item in pending:
        st = H._PIN_BY_IDX.get(item["layer"])
        if st is None or st.row_map is None:
            continue
        swap_rows_known(
            st,
            item["out_id"],
            item["in_id"],
            item["hot_row"],
            item["cold_row"],
            H._staging,
            misordered=False,
        )
        n += 1
        H._n_swaps += 1
        _log_swap(item)
    return n


def _log_swap(item: dict) -> None:
    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "step": item.get("step"),
        "layer": item["layer"],
        "in_id": item["in_id"],
        "out_id": item["out_id"],
        "ewma_in": item.get("ewma_in"),
        "ewma_out": item.get("ewma_out"),
        "ewma_coolest_hot": item.get("ewma_coolest_hot"),
        "n_swaps": H._n_swaps,
    }
    try:
        os.makedirs(os.path.dirname(H.ADAPTIVE_LOG) or ".", exist_ok=True)
        with open(H.ADAPTIVE_LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError as e:
        _log(f"swap log error {e!r}")
    _log(
        f"swap L{item['layer']} in={item['in_id']} out={item['out_id']} "
        f"ewma_in={item.get('ewma_in'):.3f} ewma_out={item.get('ewma_out'):.3f} "
        f"n={H._n_swaps}"
    )


def _open_log():
    try:
        os.makedirs(os.path.dirname(H.ADAPTIVE_LOG) or ".", exist_ok=True)
        with open(H.ADAPTIVE_LOG, "a") as f:
            f.write(
                json.dumps(
                    {
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "event": "adaptive_start",
                        "freeze": H.ADAPTIVE_FREEZE,
                        "counter": _counter(),
                        "ring": H.ADAPTIVE_RING,
                        "sample": H.ADAPTIVE_SAMPLE,
                        "drain_n": H.DRAIN_N,
                        "halflife": H.EWMA_HALFLIFE,
                        "hyst": H.SWAP_HYST,
                        "min_abs": H.SWAP_MIN_ABS,
                        "swaps_per_drain": H.SWAPS_PER_DRAIN,
                        "staging_miB": round(H._staging_bytes / 1024**2, 3),
                    }
                )
                + "\n"
            )
    except OSError as e:
        _log(f"log open error {e!r}")


def _apply_ewma(counts, steps: float) -> None:
    decay = 0.5 ** (steps / H.EWMA_HALFLIFE)
    if H._ewma is None:
        H._ewma = {i: [0.0] * H.E for i in H._PIN_BY_IDX}
    for L, bucket in counts.items():
        ew = H._ewma.setdefault(L, [0.0] * H.E)
        for e in range(H.E):
            ew[e] = ew[e] * decay + bucket[e]


def _queue_swaps_from_ewma() -> None:
    if H.ADAPTIVE_FREEZE:
        return
    queued = []
    for L, st in H._PIN_BY_IDX.items():
        ew = H._ewma.get(L)
        if ew is None or st.hot_eids is None:
            continue
        hot = st.hot_eids.detach().to("cpu").tolist()
        cold = st.cold_eids.detach().to("cpu").tolist()
        hot_vals = [(ew[int(e)], int(e), r) for r, e in enumerate(hot)]
        if not hot_vals:
            continue
        coolest = min(hot_vals)
        thr = max(coolest[0] * H.SWAP_HYST, H.SWAP_MIN_ABS)
        cands = []
        for r, e in enumerate(cold):
            e = int(e)
            v = ew[e]
            if v >= thr and v > coolest[0]:
                cands.append((v, e, r))
        cands.sort(reverse=True)
        hot_vals.sort()
        used_hot = set()
        used_cold = set()
        for i, (v, e_in, r_c) in enumerate(cands[: H.SWAPS_PER_DRAIN]):
            out = None
            for hv, e_out, r_h in hot_vals:
                if e_out in used_hot:
                    continue
                if v >= max(hv * H.SWAP_HYST, H.SWAP_MIN_ABS):
                    out = (hv, e_out, r_h)
                    break
            if out is None:
                continue
            used_hot.add(out[1])
            used_cold.add(e_in)
            queued.append(
                {
                    "layer": L,
                    "in_id": e_in,
                    "out_id": out[1],
                    "hot_row": out[2],
                    "cold_row": r_c,
                    "ewma_in": v,
                    "ewma_out": out[0],
                    "ewma_coolest_hot": coolest[0],
                    "step": int(H._drain_steps),
                }
            )
    if queued:
        with H._swap_lock:
            H._pending_swaps.extend(queued)
        _log(
            f"queued {len(queued)} swaps freeze={H.ADAPTIVE_FREEZE} "
            f"steps={H._drain_steps:.0f}"
        )


def drain_and_decide() -> None:
    """Side-thread: D2H snap (D1) or ring, bincount, EWMA, queue swaps."""
    if not H._ring_ready:
        return
    mode = _counter()
    if mode != "ring":
        t_cpu = H._last_t.to("cpu")
        ids_cpu = H._last_ids.to("cpu")
        host_now = int(H._host_steps)
        delta = max(1, host_now - int(H._seen_cursor))
        H._seen_cursor = host_now
        H._drain_steps += delta
        counts = {i: [0] * H.E for i in H._PIN_BY_IDX}
        for L in list(counts):
            T = int(t_cpu[L].item())
            if T <= 0:
                continue
            T = min(T, H.RING_TMAX)
            row = ids_cpu[L, :T].reshape(-1).tolist()
            bucket = counts[L]
            for e in row:
                if 0 <= e < H.E:
                    bucket[e] += 1
        _apply_ewma(counts, float(delta))
        _queue_swaps_from_ewma()
        return

    cur = int(H._cursor.item())
    seen = H._seen_cursor
    if cur <= seen:
        return
    n = cur - seen
    if n > H.RING_R:
        _log(f"ring overrun lost={n - H.RING_R} cur={cur} seen={seen}")
        n = H.RING_R
        seen = cur - H.RING_R
    n_layers = max(1, len(H._PIN_BY_IDX))
    steps = max(1, n / n_layers)
    H._drain_steps += steps
    start = seen % H.RING_R
    if start + n <= H.RING_R:
        idx = list(range(start, start + n))
    else:
        idx = list(range(start, H.RING_R)) + list(range(0, (start + n) % H.RING_R))
    t_cpu = H._ring_t[idx].to("cpu")
    l_cpu = H._ring_layer[idx].to("cpu")
    ids_cpu = H._ring[idx].to("cpu")
    H._seen_cursor = cur
    counts = {i: [0] * H.E for i in H._PIN_BY_IDX}
    for i in range(len(idx)):
        L = int(l_cpu[i].item())
        T = int(t_cpu[i].item())
        if L not in counts or T <= 0:
            continue
        row = ids_cpu[i, :T].reshape(-1).tolist()
        bucket = counts[L]
        for e in row:
            if 0 <= e < H.E:
                bucket[e] += 1
    _apply_ewma(counts, steps)
    _queue_swaps_from_ewma()


def _worker():
    go = os.environ.get("PIN_ADAPTIVE_GO", "<box-home>/pin-hot-experts/e4/GO")
    _log(f"worker parked until GO file {go}")
    while not os.path.exists(go):
        time.sleep(0.5)
    _log(
        f"GO seen freeze={H.ADAPTIVE_FREEZE} drain_n={H.DRAIN_N} "
        f"halflife={H.EWMA_HALFLIFE} counter={_counter()}"
    )
    last = 0
    while True:
        try:
            time.sleep(0.05)
            if not H._ring_ready:
                continue
            mode = _counter()
            if mode != "ring":
                steps = int(H._host_steps)
                if (steps - last) < H.DRAIN_N:
                    continue
                drain_and_decide()
                last = int(H._seen_cursor)
            else:
                cur = int(H._cursor.item())
                n_layers = max(1, len(H._PIN_BY_IDX) or H.N_MOE_LAYERS)
                if (cur - last) < H.DRAIN_N * n_layers:
                    continue
                drain_and_decide()
                last = H._seen_cursor
            if (not H._execute_wrapped) and (not H.ADAPTIVE_FREEZE):
                dev = H._last_ids.device
                s = torch.cuda.default_stream(dev)
                with torch.cuda.stream(s):
                    apply_pending_swaps()
        except Exception as e:
            _log(f"worker error {e!r}")
            time.sleep(1.0)


def ensure_worker() -> None:
    """Idempotent: start the parked drain thread (it waits on the GO file)."""
    if H._worker_on or not H._adaptive_started:
        return
    t = threading.Thread(target=_worker, daemon=True, name="pin-hot-adapt")
    t.start()
    H._worker_on = True
    _log("worker thread parked (GO file)")


def start_adaptive(states) -> None:
    if H._adaptive_started:
        return
    if not states:
        return
    dev = states[0].hot["w1"].device
    for st in states:
        H._PIN_BY_IDX[st.idx] = st
    init_ring(dev)
    alloc_staging(states[0])
    H._ewma = {st.idx: [0.0] * H.E for st in states}
    H._host_steps = 0
    H._seen_cursor = 0
    if H._cursor is not None:
        H._cursor.zero_()
    torch.cuda.synchronize()
    _open_log()
    H._adaptive_started = True
    _log(
        f"buffers ready layers={len(states)} freeze={H.ADAPTIVE_FREEZE} "
        f"counter={_counter()} staging_miB={H._staging_bytes / 1024**2:.2f} "
        f"worker parked on GO file. "
        f"ordering=row-copies then map-flip on main stream between steps."
    )
    ensure_worker()


def resolve_state(mod, w1):
    st = getattr(mod, "_pin", None)
    if st is None or st.row_map is None:
        st = H._PIN_BY_W1.get(w1.data_ptr() if torch.is_tensor(w1) else None)
        if st is not None:
            mod._pin = st
    return st
