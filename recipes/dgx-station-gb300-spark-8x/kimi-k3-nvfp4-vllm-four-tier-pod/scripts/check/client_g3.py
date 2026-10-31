#!/usr/bin/env python3
"""G3 / A4 — Station -> one Spark hop cost, end to end, on the v1 worker.

Modes (all on one RC connection, one request in flight, 7,424 B each way):
  forced  : for k = 1..16, requests whose 16 ids contain exactly k held by this Spark (rest held elsewhere)
  replay  : real decode top-16 vectors from the Oct-3 routing trace; sent only when this Spark holds >= 1
Every reply is checked: req_id echoed, bank_tag matches the plan, k_used and the ids_used mask equal what
the plan predicts, y finite and non-zero.

Usage: client_g3.py SPARK_FABRIC_IP PORT TARGET [iters_per_k] [replay_limit]
Runs on the Station (python with numpy; libk3hop.so next to this file).
"""
from __future__ import annotations

import json, os, struct, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k3proto import HDR, HIDDEN, MSG_BYTES, REP, REQ, SHUTDOWN_ID, ST_OK, TOPK, Conn  # noqa: E402

PEER, PORT, TARGET = sys.argv[1], int(sys.argv[2]), sys.argv[3]
ITERS = int(sys.argv[4]) if len(sys.argv) > 4 else 500
REPLAY = int(sys.argv[5]) if len(sys.argv) > 5 else 20000
PLAN = os.environ.get("PLAN", "/tmp/plan-uniform-62.json")
TRACE = os.environ.get("TRACE", "/models/kimi-k3/hist/k3_routing_trace.bin")
DEV = os.environ.get("DEV", "mlx5_0")
OUT = os.environ.get("OUT", f"$K3_WORK/g3-{TARGET}.json")

plan = json.load(open(PLAN))
held = {int(L): set(v) for L, v in plan["targets"][TARGET]["layers"].items()}
layers = plan["moe_layers"]
rng = np.random.default_rng(20261030)

conn = Conn.connect(PEER, PORT, DEV, 3)
req = conn.view(0, MSG_BYTES)
x = rng.standard_normal(HIDDEN).astype(np.float32)
xb = (x.view(np.uint32) >> 16).astype(np.uint16)          # bf16 bits
np.frombuffer(req, dtype=np.uint8)[HDR:HDR + HIDDEN * 2] = xb.view(np.uint8)
rid = 0
bank_tag = None


def call(layer, gids, ws):
    global rid, bank_tag
    rid += 1
    REQ.pack_into(req, 0, rid, layer, 1, 0, 0, *gids, *ws)
    t0 = time.perf_counter_ns()
    conn.send(0, MSG_BYTES)
    s = conn.poll(2_000_000)
    t1 = time.perf_counter_ns()
    if s < 0:
        raise RuntimeError(f"timeout req {rid}")
    rv = conn.view(s, MSG_BYTES)
    r = REP.unpack_from(rv, 0)
    y = np.frombuffer(rv, dtype=np.uint16, count=HIDDEN, offset=HDR).copy()
    conn.post_recv(s)
    rq, status, k_used, mask, tag, t_recv, t_x, t_k, t_kd, t_rep, kns = r
    assert rq == rid, (rq, rid)
    if bank_tag is None:
        bank_tag = tag
    assert tag == bank_tag
    want_mask = sum(1 << j for j, g in enumerate(gids) if g in held[layer])
    assert mask == want_mask and k_used == bin(want_mask).count("1"), (layer, mask, want_mask, k_used)
    yf = (y.astype(np.uint32) << 16).view(np.float32)
    ok = status == ST_OK and np.isfinite(yf).all() and float(np.abs(yf).mean()) > 0
    return {"rtt_us": (t1 - t0) / 1e3, "k": k_used, "ok": bool(ok),
            "worker_us": (t_rep - t_recv) / 1e3, "kernel_us": kns / 1e3, "pre_us": (t_k - t_recv) / 1e3,
            "post_us": (t_rep - t_kd) / 1e3}


def summarize(rows):
    a = np.array([r["rtt_us"] for r in rows]); w = np.array([r["worker_us"] for r in rows])
    kk = np.array([r["kernel_us"] for r in rows])
    return {"n": len(rows), "rtt_p50": round(float(np.percentile(a, 50)), 1),
            "rtt_p95": round(float(np.percentile(a, 95)), 1), "rtt_p99": round(float(np.percentile(a, 99)), 1),
            "worker_p50": round(float(np.percentile(w, 50)), 1), "kernel_p50": round(float(np.percentile(kk, 50)), 1),
            "net_and_client_p50": round(float(np.percentile(a - w, 50)), 1),
            "bad": int(sum(not r["ok"] for r in rows))}


res = {"target": TARGET, "peer": PEER, "msg_bytes": MSG_BYTES, "forced": {}, "replay": None}
# warm
for _ in range(50):
    L = int(rng.choice(layers)); h = sorted(held[L])
    call(L, [h[0]] + [-1] * 15, [1.0] + [0.0] * 15)

# forced k curve
for k in range(1, TOPK + 1):
    rows = []
    for _ in range(ITERS):
        L = int(rng.choice(layers))
        h = sorted(held[L]); nh = [g for g in range(896) if g not in held[L]]
        ids = list(rng.choice(h, k, replace=False)) + list(rng.choice(nh, TOPK - k, replace=False))
        rng.shuffle(ids)
        w = rng.dirichlet(np.ones(TOPK)).astype(np.float32).tolist()
        rows.append(call(L, [int(i) for i in ids], w))
    res["forced"][k] = summarize(rows)
    print("FORCED", k, json.dumps(res["forced"][k]), flush=True)

# replay real decode routing (trace: u8 layer | u8 is_prefill | u32 n | i16[n*16])
if os.path.exists(TRACE):
    data = open(TRACE, "rb").read()
    off = 0; rows = []; seen = 0; skipped = 0
    while off < len(data) and len(rows) < REPLAY:
        layer, pre, n = struct.unpack_from("<BBI", data, off); off += 6
        ids = np.frombuffer(data, dtype=np.int16, count=n * 16, offset=off).reshape(n, 16); off += n * 32
        if pre:
            continue
        for t in range(n):
            seen += 1
            g = [int(v) for v in ids[t]]
            if not any(v in held[layer] for v in g):
                skipped += 1
                continue
            rows.append(call(layer, g, [1.0 / 16] * 16))
    by_k = {}
    for r in rows:
        by_k.setdefault(r["k"], []).append(r)
    res["replay"] = {"decode_dispatches_seen": seen, "sent": len(rows), "not_held_here": skipped,
                     "p_this_spark_hit": round(len(rows) / max(seen, 1), 4),
                     "all": summarize(rows) if rows else None,
                     "by_k": {k: summarize(v) for k, v in sorted(by_k.items())}}
    print("REPLAY", json.dumps(res["replay"]), flush=True)

REQ.pack_into(req, 0, SHUTDOWN_ID, 0, 1, 0, 0, *([-1] * 16), *([0.0] * 16))
conn.send(0, 64)
conn.poll(2_000_000)
json.dump(res, open(OUT, "w"), indent=1)
print("WROTE", OUT)
