#!/usr/bin/env python3
"""Per-layer critical path for candidate placements, on real routed rows.
layer time = max(Station local, slowest worker); Station local = 15 + 3*k_hbm + 48*k_grace us (measured,
grace_vs_hbm_moe.py), Spark = 164.2 + 76.7 k (G3), RTX = measured k-curve (C1)."""
import json, struct, numpy as np
H1 = "/models/kimi-k3/hist/k3-trace-20261003-1213/k3_routing_hist.npz"
H2 = "/models/kimi-k3/hist/k3-ep-hist-20261002-0410/k3_routing_hist.final.npz"
TRACE = "/models/kimi-k3/hist/k3-trace-20261003-1213/k3_routing_trace.bin"
uni = json.load(open("/tmp/plan-4tier-v1.json")); layers = uni["moe_layers"]
hist = sum(np.load(p)["prefill"] + np.load(p)["decode"] for p in (H1, H2)).astype(float)
RTX = [0, 405.2, 408.3, 413.1, 414.8, 440, 455, 463, 471.4, 500, 530, 565, 598.4, 620, 650, 670, 693.1]
HBM_PER_LAYER = 60          # ~85 GiB of packed station weights fit beside everything else

def own_uniform():
    o = {}
    for L in layers:
        a = np.full(896, -1, np.int8)
        for i in range(8): a[uni["targets"][f"spark{i}"]["layers"][str(L)]] = i
        a[uni["targets"]["rtx0"]["layers"][str(L)]] = 8
        st = sorted(uni["station"]["layers"][str(L)], key=lambda g: -hist[L][g])
        a[st[:HBM_PER_LAYER]] = -1   # station HBM (hot within the station set, if HBM were chosen by heat)
        a[st[HBM_PER_LAYER:]] = -2   # station Grace
        o[L] = a
    return o

def own_ranked(n_station, n_hbm, rtx_cold=True, uniform_grace_hbm=False):
    o = {}
    for L in layers:
        order = list(np.argsort(-hist[L], kind="stable"))
        a = np.full(896, -1, np.int8)
        st, rest = order[:n_station], order[n_station:]
        a[st[:n_hbm]] = -1; a[st[n_hbm:]] = -2
        if rtx_cold:
            a[rest[-56:]] = 8; mid = rest[:-56]
        else:
            mid = rest
        for i in range(8): a[mid[i::8]] = i
        o[L] = a
    return o

def sim(o, rows):
    t = []; parts = []
    for L, r in rows:
        w = o[L][r]
        st = 15 + 3 * int((w == -1).sum()) + 48 * int((w == -2).sum())
        rem = 0.0
        for s in range(8):
            k = int((w == s).sum())
            if k: rem = max(rem, 164.2 + 76.7 * k)
        k = int((w == 8).sum())
        if k: rem = max(rem, RTX[k])
        t.append(max(st, rem)); parts.append((st, rem))
    t = np.array(t); p = np.array(parts)
    return t.mean(), p[:, 0].mean(), p[:, 1].mean()

data = open(TRACE, "rb").read(); rows = []; off = 0
while off < len(data):
    L, pre, n = struct.unpack_from("<BBI", data, off); off += 6
    v = np.frombuffer(data, np.int16, n * 16, off).reshape(n, 16).astype(np.int64); off += n * 32
    rows += [(L, r) for r in v]
rng = np.random.default_rng(0); rows = [rows[i] for i in rng.choice(len(rows), 30000, replace=False)]
print("uniform today (station Grace uses vLLM layer-order offload ~ 5/6 Grace):")
o = own_uniform()
for L in layers:   # emulate today: no heat-aware HBM choice -> 1/6 of station experts HBM at random
    w = o[L]; st = np.where(w < 0)[0]; w[st] = -2; w[st[::6]] = -1
print("  uniform-v1      : layer %.0f us (station %.0f, remote %.0f) -> %.1f ms/token" % ((*sim(o, rows),) + (sim(o, rows)[0] * 92 / 1e3,)))
for n_st in (200, 250, 300, 344):
    for hbm in (60,):
        m = sim(own_ranked(n_st, hbm), rows)
        print(f"  hot{n_st:>3} hbm{hbm} rtxcold: layer {m[0]:.0f} us (station {m[1]:.0f}, remote {m[2]:.0f}) -> {m[0] * 92 / 1e3:.1f} ms/token")
