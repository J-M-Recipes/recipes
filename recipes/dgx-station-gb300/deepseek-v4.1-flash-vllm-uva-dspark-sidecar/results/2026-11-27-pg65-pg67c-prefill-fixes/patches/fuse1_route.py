"""fuse1_route.py.DRAFT -> lands as fastpath/hook/fuse1_route.py   (W12 FusEager, candidate F1a; DESK DRAFT, never run on a GPU)

What it replaces (mega_peer_hook._forward, T > FUSED_MAX cold path, src/pg66-tree/fastpath/hook/mega_peer_hook.py:222-225):

    valid = topk_ids >= 0
    rm = st.row_map[topk_ids.clamp_min(0).long()]
    hot_ids  = torch.where(valid & (rm >= 0), rm, -1).to(topk_ids.dtype)
    cold_ids = torch.where(valid & (rm < 0), -rm - 1, -1).to(torch.int32)

That is 12 CUDA elementwise/index launches per MoE layer (ge, clamp_min, index, ge, and, where, to, lt, and, neg, sub, where; `.long()` and the
final `.to(int32)` are no-ops for int64 topk_ids / int32 cold) and it is a pure function of ONE integer per route and the per-layer 384-entry row map.
So it is a table lookup.  Two 385-entry tables built once per layer at finalize:

    hot_tab [E+1]  (dtype = topk_ids.dtype) : hot_tab[e]  = row[e]       if row[e] >= 0 else -1 ;  hot_tab[E]  = -1
    cold_tab[E+1]  (int32)                   : cold_tab[e] = -row[e] - 1  if row[e] <  0 else -1 ;  cold_tab[E] = -1

and the lookup is   idx = topk_ids.clamp_min(-1);  hot_ids = hot_tab[idx];  cold_ids = cold_tab[idx]     (3 launches).
`clamp_min(-1)` sends EVERY negative id (the original's `valid = ids >= 0` treats all negatives alike) to -1, which advanced indexing wraps to slot E,
the sentinel -1/-1.  Domain in which it is bit-identical to the original: every integer id <= E-1 (exhaustively tested -12..E-1).  Differs ONLY for
id >= E, where the original indexes row_map out of bounds (device assert / undefined) and the table silently maps id == E to -1: that input is outside
the gate's contract either way.

Safety rails (all fail-open to the ORIGINAL code path):
  * never used while a CUDA graph is being captured (captured graphs stay byte-identical => A12 graph inventory / graph bytes unchanged by construction);
  * dtype of topk_ids not in {int64, int32} -> None -> original path;
  * first eager call per layer runs a self-test on the device against the original formula (all ids -3..E-1); mismatch -> path disabled + logged.
"""
from __future__ import annotations

import sys

import torch

E = 384


class RouteTables:
    __slots__ = ("pairs", "checked", "ok", "layer")

    def __init__(self, pairs, layer):
        self.pairs = pairs      # {torch.dtype: (hot_tab, cold_tab)}
        self.checked = False
        self.ok = False
        self.layer = layer


def build_tables(row, device, layer=-1, num_experts: int = E) -> RouteTables:
    """row: python list (len E) of ColdState.row: >=0 hot row, else -(cold row + 1).  Built on CPU, copied (loader-thread safe, like row_map)."""
    assert len(row) == num_experts
    r = torch.tensor(list(row) + [0], dtype=torch.int64)
    neg1 = torch.full_like(r, -1)
    hot = torch.where(r >= 0, r, neg1)
    cold = torch.where(r < 0, -r - 1, neg1)
    hot[num_experts] = -1
    cold[num_experts] = -1
    pairs = {
        torch.int64: (hot.to(torch.int64).to(device), cold.to(torch.int32).to(device)),
        torch.int32: (hot.to(torch.int32).to(device), cold.to(torch.int32).to(device)),
    }
    return RouteTables(pairs, layer)


def split_reference(row_map: torch.Tensor, topk_ids: torch.Tensor):
    """The ORIGINAL four lines of mega_peer_hook._forward (copied verbatim) -- the oracle."""
    valid = topk_ids >= 0
    rm = row_map[topk_ids.clamp_min(0).long()]
    hot_ids = torch.where(valid & (rm >= 0), rm, -1).to(topk_ids.dtype)
    cold_ids = torch.where(valid & (rm < 0), -rm - 1, -1).to(torch.int32)
    return hot_ids, cold_ids


def _lookup(pair, topk_ids):
    idx = topk_ids.clamp_min(-1)
    return pair[0][idx], pair[1][idx]


def _selftest(tabs: RouteTables, row_map: torch.Tensor, num_experts: int = E) -> bool:
    dev = row_map.device
    ok = True
    for dt, pair in tabs.pairs.items():
        ids = torch.arange(-3, num_experts, dtype=dt, device=dev).reshape(-1, 1).expand(-1, 6).contiguous()
        h0, c0 = split_reference(row_map, ids)
        h1, c1 = _lookup(pair, ids)
        ok = ok and h0.dtype == h1.dtype and c0.dtype == c1.dtype and bool(torch.equal(h0, h1)) and bool(torch.equal(c0, c1))
    return ok


def route_split(tabs: RouteTables | None, row_map: torch.Tensor, topk_ids: torch.Tensor):
    """(hot_ids, cold_ids) bit-identical to split_reference(row_map, topk_ids), or None -> caller runs the original lines.  EAGER ONLY (the caller
    guards torch.cuda.is_current_stream_capturing())."""
    if tabs is None:
        return None
    pair = tabs.pairs.get(topk_ids.dtype)
    if pair is None:
        return None
    if not tabs.checked:
        tabs.checked = True
        try:
            tabs.ok = _selftest(tabs, row_map)
        except Exception as e:  # fail open
            sys.stderr.write(f"MEGA_PEER FUSE1 selftest error layer {tabs.layer}: {e!r}\n")
            tabs.ok = False
        sys.stderr.write(f"MEGA_PEER FUSE1 layer {tabs.layer}: selftest {'OK' if tabs.ok else 'FAILED -> original path'}\n"
                         if tabs.layer in (0, 20, 21, 39) or not tabs.ok else "")
    if not tabs.ok:
        return None
    return _lookup(pair, topk_ids)
