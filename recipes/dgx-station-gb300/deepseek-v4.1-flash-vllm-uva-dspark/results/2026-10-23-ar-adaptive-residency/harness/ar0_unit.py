#!/usr/bin/env python3
"""AR0 unit tests for the v26ar hook (atomic counter + guarded swaps).

Runs inside vllm/vllm-openai:deepseekv41-flash-0909 on one quiet GPU. Never
loads the model. Writes a JSON verdict to $AR0_OUT.

T1 split+count equivalence (T = 1, 6, 24, 48, 96; with -1 lanes):
   hot/cold ids identical to the base kernel and the torch reference; counts
   equal torch.bincount; flag off -> no counting.
T2 CUDA graph: 40 layer launches captured, 50 replays -> counts = 50 x bincount
   per layer; flag toggled between replays takes effect without re-capture.
T3 queue policy (CPU): one clear winner queues exactly one swap with the
   expected rows; frozen queues nothing; busy layer is skipped.
T4 guarded apply on GPU: rows move, row_map flips, mirror updates; replaying
   the same (now stale) item is refused and counted.
T5 drain: deltas, EWMA, cold share, telemetry line.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import threading
import time

import numpy as np
import torch

HOOK = os.environ.get("PIN_HOOK", "/w/pin_hot_experts_hook.py")
ROWMAP = os.environ.get("PIN_ROWMAP", "/w/rowmap-static-v1.json")
OUT = os.environ.get("AR0_OUT", "/out/ar0-unit.json")
os.environ.setdefault("PIN_COUNTER", "atomic")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


R: dict = {"hook": HOOK, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "tests": {}}


def record(name, ok, **kw):
    R["tests"][name] = {"ok": bool(ok), **kw}
    print(f"[ar0] {name}: {'PASS' if ok else 'FAIL'} {kw if kw else ''}", flush=True)


pin = load(HOOK, "pin_hot_experts_hook")
dev = torch.device("cuda", 0)
torch.cuda.set_device(dev)
E, L, K = pin.E, pin.N_MOE_LAYERS, pin.TOPK
rowmap = pin.load_rowmap(ROWMAP)
rms = [pin.build_row_map_tensor(rowmap[i]["hot"], rowmap[i]["cold"], dev) for i in range(L)]
pin._cnt = torch.zeros((L, E), dtype=torch.int32, device=dev)
pin._cnt_on = torch.ones(1, dtype=torch.int32, device=dev)
g = torch.Generator(device="cpu").manual_seed(20261023)


def make_ids(T):
    ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(T)]).to(torch.int32)
    if T >= 6:
        ids[1, 2] = -1
        ids[T - 1, 0] = -1
    return ids.to(dev)


# ---- T1 ---------------------------------------------------------------------
t1 = []
for T in (1, 6, 24, 48, 96):
    ids = make_ids(T)
    rm = rms[3]
    h0, c0 = pin.split_ids(ids, rm)
    pin._cnt.zero_()
    h1, c1 = pin.split_ids(ids, rm, pin._cnt[3])
    torch.cuda.synchronize()
    ref = torch.bincount(ids[ids >= 0].long(), minlength=E).to(torch.int32)
    same = torch.equal(h0, h1) and torch.equal(c0, c1)
    ht, ct = pin.split_ids_torch(ids, rm)
    same_ref = torch.equal(ht, h1) and torch.equal(ct, c1)
    cnt_ok = torch.equal(pin._cnt[3], ref) and int(pin._cnt.sum()) == int(ref.sum())
    pin._cnt_on.zero_()
    pin.split_ids(ids, rm, pin._cnt[3])
    torch.cuda.synchronize()
    off_ok = torch.equal(pin._cnt[3], ref)
    pin._cnt_on.fill_(1)
    t1.append({"T": T, "same_as_base": same, "same_as_torch": same_ref,
               "counts_ok": cnt_ok, "flag_off_ok": off_ok})
record("T1_split_count_equivalence", all(all(v for k, v in r.items() if k != "T") for r in t1), rows=t1)

# ---- T2 ---------------------------------------------------------------------
T = 6
ids_l = [make_ids(T) for _ in range(L)]
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(2):
        for i in range(L):
            pin.split_ids(ids_l[i], rms[i], pin._cnt[i])
torch.cuda.current_stream().wait_stream(s)
torch.cuda.synchronize()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    outs = [pin.split_ids(ids_l[i], rms[i], pin._cnt[i]) for i in range(L)]
pin._cnt.zero_()
torch.cuda.synchronize()
N = 50
for _ in range(N):
    graph.replay()
torch.cuda.synchronize()
ref = torch.stack([torch.bincount(x[x >= 0].long(), minlength=E).to(torch.int32) for x in ids_l])
g_ok = torch.equal(pin._cnt, ref * N)
pin._cnt_on.zero_()
for _ in range(10):
    graph.replay()
torch.cuda.synchronize()
g_off = torch.equal(pin._cnt, ref * N)
pin._cnt_on.fill_(1)
graph.replay()
torch.cuda.synchronize()
g_on = torch.equal(pin._cnt, ref * (N + 1))
h_ok = all(torch.equal(outs[i][0], pin.split_ids_torch(ids_l[i], rms[i])[0]) for i in range(L))
record("T2_graph_counts", g_ok and g_off and g_on and h_ok,
       counts_eq_50x=g_ok, flag_off_frozen=g_off, flag_on_resumes=g_on, graph_outputs_ok=h_ok)

# ---- T3 ---------------------------------------------------------------------
e4 = pin._e4()
pin.ADAPTIVE_FREEZE = False
pin._pending_swaps = []
pin._drain_steps = 0
hot0 = list(rowmap[0]["hot"])
cold0 = list(rowmap[0]["cold"])
pin._ar_hot_cpu = {0: list(hot0)}
pin._ar_cold_cpu = {0: list(cold0)}
pin._ar_ewma = np.zeros((L, E), dtype=np.float64)
pin._ar_ewma[0, hot0] = 10.0
pin._ar_ewma[0, hot0[7]] = 1.0
pin._ar_ewma[0, cold0[3]] = 50.0
pin.SWAP_HYST, pin.SWAP_MIN_ABS, pin.SWAPS_PER_DRAIN = 1.5, 8.0, 2
n = e4._queue_atomic()
q = list(pin._pending_swaps)
exp = {"layer": 0, "in_id": cold0[3], "out_id": hot0[7], "hot_row": 7, "cold_row": 3}
one_ok = n == 1 and all(q[0][k] == v for k, v in exp.items())
n_busy = e4._queue_atomic()  # layer 0 is busy (pending) -> nothing new
pin._pending_swaps = []
pin.ADAPTIVE_FREEZE = True
n_frozen = e4._queue_atomic()
pin.ADAPTIVE_FREEZE = False
record("T3_queue_policy", one_ok and n_busy == 0 and n_frozen == 0,
       queued=n, item=q[0] if q else None, busy_queued=n_busy, frozen_queued=n_frozen)

# ---- T4 ---------------------------------------------------------------------
D = 64
st = pin.PinState(0, "fake.layer0")
st.row_map = rms[0].clone()
st.hot = {"w1": torch.arange(295 * D, dtype=torch.float32, device=dev).view(295, D)}
st.cold = {"w1": (torch.arange(89 * D, dtype=torch.float32, device=dev).view(89, D) + 1e6)}
st.hot_eids = torch.tensor(hot0, dtype=torch.int32, device=dev)
st.cold_eids = torch.tensor(cold0, dtype=torch.int32, device=dev)
pin._PIN_BY_IDX = {0: st}
pin._staging = {"w1": torch.empty(D, dtype=torch.float32, device=dev)}
pin._ar_stale = 0
pin._n_swaps = 0
tmpd = tempfile.mkdtemp()
pin.ADAPTIVE_LOG = os.path.join(tmpd, "swaps.jsonl")
hot_row7 = st.hot["w1"][7].clone()
cold_row3 = st.cold["w1"][3].clone()
pin._pending_swaps = [dict(q[0])]
applied = e4.apply_pending_swaps()
torch.cuda.synchronize()
rows_ok = torch.equal(st.hot["w1"][7], cold_row3) and torch.equal(st.cold["w1"][3], hot_row7)
rm_cpu = st.row_map.cpu().tolist()
map_ok = rm_cpu[cold0[3]] == 7 and rm_cpu[hot0[7]] == -(3 + 1)
mir_ok = pin._ar_hot_cpu[0][7] == cold0[3] and pin._ar_cold_cpu[0][3] == hot0[7]
eid_ok = int(st.hot_eids[7]) == cold0[3] and int(st.cold_eids[3]) == hot0[7]
pin._pending_swaps = [dict(q[0])]
applied2 = e4.apply_pending_swaps()
stale_ok = applied2 == 0 and pin._ar_stale == 1
pin.SWAPS_PER_BARRIER = 1
pin._pending_swaps = [dict(q[0]), dict(q[0]), dict(q[0])]
e4.apply_pending_swaps()
cap_ok = len(pin._pending_swaps) == 2
pin._pending_swaps = []
record("T4_guarded_apply", applied == 1 and rows_ok and map_ok and mir_ok and eid_ok and stale_ok and cap_ok,
       applied=applied, rows_moved=rows_ok, map_flipped=map_ok, mirror=mir_ok,
       eids=eid_ok, stale_refused=stale_ok, barrier_cap=cap_ok)

# ---- T5 ---------------------------------------------------------------------
pin.TELEMETRY = os.path.join(tmpd, "telemetry.jsonl")
pin._cnt.zero_()
torch.cuda.synchronize()
e4._atomic_init([st], dev)
pin._host_steps = 0
pin._seen_cursor = 0
e4._atomic_baseline("unit")
cur_cold = pin._ar_cold_cpu[0]
cur_hot = pin._ar_hot_cpu[0]
pin._cnt[0, cur_hot[:10]] += 9   # 90 hot selections
pin._cnt[0, cur_cold[:2]] += 5   # 10 cold selections
torch.cuda.synchronize()
pin._host_steps = 64
pin.ADAPTIVE_FREEZE = True
e4._drain_atomic()
lines = open(pin.TELEMETRY).read().strip().splitlines()
last = json.loads(lines[-1])
drain_ok = (last["sel"] == 100 and last["cold"] == 10 and abs(last["cold_share"] - 0.1) < 1e-9
            and last["dsteps"] == 64 and abs(pin._ar_ewma[0, cur_cold[0]] - 5.0) < 1e-9)
record("T5_drain_telemetry", drain_ok, telemetry=last)

# ---- T6 (v26ar.1) restore_map --------------------------------------------
if hasattr(e4, "_queue_restore"):
    pin.ADAPTIVE_FREEZE = True
    pin._pending_swaps = []
    pin.SWAPS_PER_BARRIER = 0
    moved_before = e4.hot_set_diff_from_boot()  # boot = post-T4 state, so 0
    orig = {}
    for rr, e in enumerate(pin._ar_hot_cpu[0]):
        orig[e] = st.hot["w1"][rr].clone()
    for rr, e in enumerate(pin._ar_cold_cpu[0]):
        orig[e] = st.cold["w1"][rr].clone()
    # move 3 experts away from the boot hot set with real swaps
    hot = pin._ar_hot_cpu[0]; cold = pin._ar_cold_cpu[0]
    items = [{"layer": 0, "in_id": cold[k], "out_id": hot[k + 10], "hot_row": k + 10,
              "cold_row": k, "ewma_in": 9.0, "ewma_out": 1.0, "step": 0} for k in range(3)]
    pin._pending_swaps = items
    e4.apply_pending_swaps()
    moved_mid = e4.hot_set_diff_from_boot()
    nq = e4._queue_restore()
    e4.apply_pending_swaps()
    torch.cuda.synchronize()
    moved_after = e4.hot_set_diff_from_boot()
    rm = st.row_map.cpu().tolist()
    consistent = all(rm[e] >= 0 for e in pin._ar_boot_hot[0]) and all(rm[e] < 0 for e in pin._ar_boot_cold[0])
    rows_ok = (all(torch.equal(st.hot["w1"][rr], orig[e]) for rr, e in enumerate(pin._ar_hot_cpu[0]))
               and all(torch.equal(st.cold["w1"][rr], orig[e]) for rr, e in enumerate(pin._ar_cold_cpu[0])))
    rm_rows_ok = all(rm[e] == rr for rr, e in enumerate(pin._ar_hot_cpu[0])) and \
        all(rm[e] == -(rr + 1) for rr, e in enumerate(pin._ar_cold_cpu[0]))
    record("T6_restore_map", moved_before == 0 and moved_mid == 3 and nq == 3 and moved_after == 0
           and consistent and rows_ok and rm_rows_ok,
           moved_before=moved_before, moved_mid=moved_mid, queued=nq, moved_after=moved_after,
           map_consistent=consistent, weights_follow_experts=rows_ok, row_map_matches_mirror=rm_rows_ok)

R["pass"] = all(t["ok"] for t in R["tests"].values())
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(R, f, indent=2, default=str)
print(f"[ar0] VERDICT {'PASS' if R['pass'] else 'FAIL'} -> {OUT}", flush=True)
sys.exit(0 if R["pass"] else 1)
