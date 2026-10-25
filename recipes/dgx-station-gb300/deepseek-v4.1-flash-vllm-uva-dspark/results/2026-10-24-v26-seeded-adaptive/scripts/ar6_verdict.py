#!/usr/bin/env python3
"""ar6_verdict.py — Card AR6 v2 verdict (AR6-SPEC-v2.md §Gates, frozen 2026-10-24). Pure stdlib (no numpy).

Usage:
  ar6_verdict.py <R>                     -> <R>/verdict.json + human table on stdout
  ar6_verdict.py <R> --pick-winner       -> "WINNER=<B1|B2|B3|NONE>"  (runner: third-domain cell + G4 target)
  ar6_verdict.py <R> --b3-check          -> "B3=RUN ..." | "B3=SKIP ..."  (B2 G3 churn sub-gate vs same-boot B0)
  ar6_verdict.py --legacy-run1 <run1dir> [--out file]
        sanity/degraded mode on AR2 run1 artifacts (telemetry.jsonl + shift-ar2-*.json, no phases.jsonl,
        no steps.bin, no chunk timestamps): computes what it can, marks the rest UNMEASURED with the reason.

Inputs (<R> = ~/dsv41/ar/results-cardAR6): phases.jsonl (runner + ar6_pass.py wall-clock boundaries),
boot0/ and bootA/ (= the containers' /ar: telemetry.jsonl, swaps.jsonl, steps.bin, shift-<tag>.json,
mixed-<tag>.json, parity-*.json, cmp-*.json), errcount-<boot>-<tag>.txt, gauntlet-<tag>/summary.json.

Instruments (registered definitions; DECISIONS in DEPLOY-AR6.md):
  t90_cold  PRIMARY. Telemetry drain records binned into 2 s WALL windows from the shift (phase t0); window share =
            sum(cold)/sum(sel). conv = pooled share over the phase's final CONV_S (60 s). peak = max window in the
            first 20 s. thr = conv + 0.1*(peak-conv). T90 = end of the first window that, together with the next
            non-empty window, is <= thr (hold 2 windows).
  t_2x      same, thr = 2*conv (spec-literal tail instrument; reported, feeds the coin-flip zone).
  t_1pct    same, thr = 0.01.
  t90_req   prompt-normalized per-request T90 (W3 w3_trajectories method): tok/s / that prompt's median over the
            final CONV_S; first request (end time) whose rolling-3 median >= 0.90 and stays >= 0.90 through 60 s.
  t90_w3    W3's 5-drain rolling instrument (reported only, for continuity with W3/sim numbers).
  churn     swaps/min from the telemetry cumulative 'swaps' counter over conv windows of the gated phases.
  G2        steps.bin host-step dt; swap-bearing = n_applied>0 or any swaps.jsonl apply_ns in [t_k, t_k+1);
            each step normalized by its 5 s bin's non-swap median; ratio = p95(swap)/p95(non-swap) pooled.
            Chunk-gap version (harness chunk_perf) reported alongside.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import statistics
import struct
import sys
from array import array
from pathlib import Path

WIN_S = 2.0
CONV_S = 60.0
PEAK_S = 20.0
HOLD = 2
BIN_S = 5.0
MDE_POOLED = 1.0      # W3: T90_cold MDE ~1.0 s pooled over 3 repeats (±50%)
MDE_SINGLE = 1.8
TAIL_ZONE = 0.40      # <2xconv ratio above this while the registered instrument halves -> coin-flip zone -> JAMES
G1_TOL = 0.02
G2_BAR = 1.25
G3_CHURN = 1.25
G3_CONV_COLD = 0.01
H1_MIXED = 1.20
H1_AGENT_TOL = 0.02
H1_STEADY_TOL = 0.02
CLASS_MIXED_FROZEN = 133.1   # spec: Arm 0 frozen class (run3 mixed-frozen-c1 agg post60, in-sample pools)
CLASS_AGENT = 195.7          # spec: agent C1 class (static frozen)
NOISE_IDS = {"agent-tool-1", "tool-weather-time"}  # spec G4: documented noise prompts
GATED = ("agent->jpde", "jpde->agent")
PREF = ("B1", "B2", "B3")    # winner preference = lowest queue rate first (spec: prefer B1 > B2/B3 > B4)
RATE = {"B0": 1.25, "B1": 2.5, "B2": 5.0, "B3": 5.0}


# ------------------------------------------------------------------ io
def J(p):
    p = Path(p)
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def jl(p):
    out = []
    p = Path(p)
    if not p.exists():
        return out
    for line in p.read_text(errors="replace").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except Exception:  # noqa: BLE001
                pass
    return out


class Boot:
    def __init__(self, d: Path):
        self.d = d
        T = jl(d / "telemetry.jsonl")
        self.tel_all = sorted([r for r in T if "ts" in r], key=lambda r: r["ts"])
        self.drains = [r for r in self.tel_all if not r.get("event") and r.get("dsteps") and r.get("sel")]
        self.drain_ts = [r["ts"] for r in self.drains]
        self.cum = [r for r in self.tel_all if "swaps" in r]
        self.cum_ts = [r["ts"] for r in self.cum]
        self.swaps = jl(d / "swaps.jsonl")
        self._steps = None

    def cum_at(self, t, key):
        i = bisect.bisect_right(self.cum_ts, t) - 1
        return self.cum[i].get(key) if i >= 0 else None

    def steps(self):
        """steps.bin -> (data rows [(hs, perf_ns, n_applied, pending)], perf->wall offset ns) or None."""
        if self._steps is not None:
            return self._steps
        p = self.d / "steps.bin"
        if not p.exists() or p.stat().st_size < 32:
            self._steps = False
            return False
        a = array("q")
        raw = p.read_bytes()
        raw = raw[: len(raw) - len(raw) % 32]
        a.frombytes(raw)
        if sys.byteorder != "little":
            a.byteswap()
        rows, offs = [], []
        for i in range(0, len(a), 4):
            hs, t, x, y = a[i], a[i + 1], a[i + 2], a[i + 3]
            if hs >= 0:
                rows.append((hs, t, x, y))
            elif hs == -2:
                offs.append(x - t)
        if not rows or not offs:
            self._steps = False
            return False
        self._steps = (rows, statistics.median(offs))
        return self._steps


# ------------------------------------------------------------------ instruments
def windows(boot: Boot, t0, t1, w=WIN_S):
    out = []
    n = int(math.ceil((t1 - t0) / w))
    i = bisect.bisect_left(boot.drain_ts, t0)
    for k in range(n):
        a, b = t0 + k * w, min(t0 + (k + 1) * w, t1)
        sel = cold = 0
        while i < len(boot.drains) and boot.drain_ts[i] < b:
            r = boot.drains[i]
            if boot.drain_ts[i] >= a:
                sel += r["sel"]
                cold += r["cold"]
            i += 1
        out.append((b - t0, (cold / sel) if sel else None, sel))
    return out


def pooled(boot: Boot, a, b):
    i = bisect.bisect_left(boot.drain_ts, a)
    j = bisect.bisect_left(boot.drain_ts, b)
    sel = sum(r["sel"] for r in boot.drains[i:j])
    cold = sum(r["cold"] for r in boot.drains[i:j])
    return (cold / sel) if sel else None


def first_hold(ws, thr):
    ne = [(t, s) for t, s, _ in ws if s is not None]
    for k in range(len(ne)):
        if all(ne[k + h][1] <= thr for h in range(HOLD) if k + h < len(ne)) and k + HOLD - 1 < len(ne):
            return round(ne[k][0], 2)
    return None


def t90_w3(boot: Boot, t0, t1, conv):
    i = bisect.bisect_left(boot.drain_ts, t0)
    j = bisect.bisect_left(boot.drain_ts, t1)
    D = boot.drains[i:j]
    if len(D) < 8 or conv is None:
        return None
    roll = []
    for k in range(len(D)):
        s = D[max(0, k - 4): k + 1]
        sel = sum(r["sel"] for r in s)
        roll.append((D[k]["ts"] - t0, sum(r["cold"] for r in s) / sel if sel else None))
    pk = max((v for t, v in roll if t <= PEAK_S and v is not None), default=None)
    if pk is None:
        return None
    thr = conv + 0.1 * (pk - conv)
    for k in range(len(roll) - 2):
        if all(roll[k + h][1] is not None and roll[k + h][1] <= thr for h in range(3)):
            return round(roll[k][0], 2)
    return None


def t90_req(ph, t0, t1):
    S = [s for s in ph.get("samples", []) if s.get("instrument_tok_s")]
    if len(S) < 6:
        return None
    conv = {}
    for s in S:
        if s["t_wall"] >= t1 - CONV_S:
            conv.setdefault(s["prompt_head"], []).append(s["instrument_tok_s"])
    conv = {k: statistics.median(v) for k, v in conv.items()}
    seq = [(s["t_wall"] - t0, s["instrument_tok_s"] / conv[s["prompt_head"]]) for s in S if s["prompt_head"] in conv]
    if len(seq) < 4:
        return None
    roll = [(seq[k][0], statistics.median([x for _, x in seq[max(0, k - 2): k + 1]])) for k in range(len(seq))]
    for k in range(len(roll)):
        if roll[k][1] >= 0.90 and all(v >= 0.90 for t, v in roll[k:] if t <= 60.0):
            return round(roll[k][0], 2)
    return None


def conv_tok(ph, t1):
    xs = [s["instrument_tok_s"] for s in ph.get("samples", []) if s.get("instrument_tok_s") and s["t_wall"] >= t1 - CONV_S]
    return xs


def shift_metrics(boot: Boot, ph, t0, t1):
    ws = windows(boot, t0, t1)
    conv = pooled(boot, t1 - CONV_S, t1)
    peak = max((s for t, s, _ in ws if t <= PEAK_S and s is not None), default=None)
    m = {"t0": t0, "t1": t1, "n_windows": sum(1 for _, s, _ in ws if s is not None), "conv_cold": conv, "peak_cold": peak,
         "t90_cold": None, "t_2x": None, "t_1pct": None, "t90_req": None, "t90_w3": None,
         "traj_2s": [None if s is None else round(s, 4) for _, s, _ in ws[:15]]}
    if conv is not None and peak is not None:
        m["t90_cold"] = first_hold(ws, conv + 0.1 * (peak - conv))
        m["t_2x"] = first_hold(ws, 2 * conv)
        m["t_1pct"] = first_hold(ws, 0.01)
        m["soft"] = (peak - conv) < 0.01
        m["t90_w3"] = t90_w3(boot, t0, t1, conv)
    if ph is not None:
        m["t90_req"] = t90_req(ph, t0, t1)
    a, b = t1 - CONV_S, t1
    s0, s1 = boot.cum_at(a, "swaps"), boot.cum_at(b, "swaps")
    m["churn_n"] = (s1 - s0) if (s0 is not None and s1 is not None) else None
    m["churn_min"] = (b - a) / 60.0
    st0, st1 = boot.cum_at(t0, "stale"), boot.cum_at(t1, "stale")
    m["stale"] = (st1 - st0) if (st0 is not None and st1 is not None) else None
    i = bisect.bisect_left(boot.drain_ts, a)
    j = bisect.bisect_left(boot.drain_ts, b)
    m["conv_sel"] = sum(r["sel"] for r in boot.drains[i:j])
    m["conv_cold_n"] = sum(r["cold"] for r in boot.drains[i:j])
    m["conv_tok"] = conv_tok(ph, t1) if ph is not None else []
    return m


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(math.ceil(q * len(xs))) - 1))]


def g2_cell(boot: Boot, spans, shift_docs):
    """spans: list of (t0,t1) wall. Returns dict or {'unmeasured': reason}."""
    st = boot.steps()
    if not st:
        return {"unmeasured": "no steps.bin (v26ar.2 ring absent)"}
    rows, off = st
    apply = sorted(int(s["apply_ns"]) for s in boot.swaps if s.get("apply_ns") is not None)
    if not apply:
        return {"unmeasured": "swaps.jsonl has no apply_ns"}
    perf = [r[1] for r in rows]
    norm_s, norm_n, excess, napp = [], [], [], 0
    nbins = 0
    for t0, t1 in spans:
        a = bisect.bisect_left(perf, int(t0 * 1e9 - off))
        b = bisect.bisect_left(perf, int(t1 * 1e9 - off))
        bins = {}
        for k in range(a, min(b, len(rows) - 1)):
            if rows[k + 1][0] != rows[k][0] + 1:
                continue
            dt = (rows[k + 1][1] - rows[k][1]) / 1e6
            if dt <= 0 or dt > 250:
                continue
            lo = bisect.bisect_left(apply, rows[k][1])
            hi = bisect.bisect_left(apply, rows[k + 1][1])
            n = max(rows[k][2], hi - lo)
            wall = (rows[k][1] + off) / 1e9
            bins.setdefault(int((wall - t0) // BIN_S), []).append((dt, n))
        for v in bins.values():
            non = [dt for dt, n in v if n == 0]
            sw = [(dt, n) for dt, n in v if n > 0]
            if len(non) < 20 or not sw:
                continue
            nbins += 1
            med = statistics.median(non)
            norm_n += [dt / med for dt in non]
            norm_s += [dt / med for dt, _ in sw]
            excess += [dt - med for dt, _ in sw]
            napp += sum(n for _, n in sw)
    if not norm_s:
        return {"unmeasured": "no matched 5 s bins with swap-bearing steps"}
    out = {"bins": nbins, "n_swap_steps": len(norm_s), "n_non_steps": len(norm_n)}
    for q in (0.90, 0.95, 0.99):
        out[f"p{int(q * 100)}_ratio"] = round(pct(norm_s, q) / pct(norm_n, q), 4)
    out["mean_ms_per_applied_swap"] = round(sum(excess) / napp, 4) if napp else None
    cms = [s["cuda_ms"] for s in boot.swaps if s.get("cuda_ms") is not None and s.get("apply_ts")
           and any(t0 <= s["apply_ts"] <= t1 for t0, t1 in spans)]
    out["cuda_ms_mean"] = round(statistics.mean(cms), 4) if cms else None
    out["cuda_ms_p95"] = round(pct(cms, 0.95), 4) if cms else None
    # chunk-gap corroboration (client inter-token)
    sw_wall = sorted((x + off) / 1e9 for x in apply)
    gs, gn = [], []
    for doc in shift_docs:
        ca = doc.get("clock_anchor") or {}
        if "perf" not in ca:
            continue
        cof = ca["wall"] - ca["perf"]
        for ph in doc.get("phases", {}).values():
            if not any(t0 <= ph.get("wall_start", -1) < t1 for t0, t1 in spans):
                continue
            bins = {}
            for s in ph.get("samples", []):
                c = s.get("chunk_perf") or []
                for i in range(len(c) - 1):
                    w0, w1 = c[i] + cof, c[i + 1] + cof
                    sb = bisect.bisect_left(sw_wall, w1) - bisect.bisect_left(sw_wall, w0) > 0
                    bins.setdefault(int((w0 - ph["wall_start"]) // BIN_S), []).append(((w1 - w0) * 1e3, sb))
            for v in bins.values():
                non = [g for g, sb in v if not sb]
                if len(non) < 20 or not any(sb for _, sb in v):
                    continue
                med = statistics.median(non)
                gn += [g / med for g in non]
                gs += [g / med for g, sb in v if sb]
    out["chunk_gap_p95_ratio"] = round(pct(gs, 0.95) / pct(gn, 0.95), 4) if gs and gn else None
    out["chunk_gap_n_swap"] = len(gs)
    return out


# ------------------------------------------------------------------ assembly
def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 3) if xs else None


def build_cells(R: Path, phases, boots, legacy=False):
    groups = {}
    for p in phases:
        if p.get("ev") != "phase":
            continue
        groups.setdefault((p["boot"], p["tag"]), []).append(p)
    cells = {}
    for (b, tag), P in sorted(groups.items()):
        boot = boots.get(b)
        if boot is None:
            continue
        P.sort(key=lambda r: r["idx"])
        doc = J(boot.d / P[0]["file"]) or {}
        cell, rep = P[0]["cell"], P[0]["rep"]
        third = any(p["domain"] == "third" for p in P)
        key = f"{b}:{cell}" + (":third" if third else "")
        C = cells.setdefault(key, {"boot": b, "cell": cell, "third": third, "reps": [], "docs": [],
                                   "knobs": (doc.get("ar6") or {}).get("shift_ctrl")})
        C["docs"].append(doc)
        rec = {"rep": rep, "tag": tag, "shifts": {}}
        for p in P:
            name = f"{p['prev']}->{p['domain']}"
            ph = (doc.get("phases") or {}).get(p["phase"])
            m = shift_metrics(boot, ph, p["t0"], p["t1"])
            m["phase"] = p["phase"]
            m["ctrl_lag_s"] = (round(p["ctrl_wall"] - p["t0"], 3) if p.get("ctrl_wall") else None)
            rec["shifts"][name] = m
        ec = R / f"errcount-{b}-{tag}.txt"
        rec["server_errors"] = int(ec.read_text().strip()) if ec.exists() and ec.read_text().strip().isdigit() else None
        C["reps"].append(rec)
    for key, C in cells.items():
        boot = boots[C["boot"]]
        names = sorted({n for r in C["reps"] for n in r["shifts"]})
        C["pooled"] = {}
        for n in names:
            ms = [r["shifts"][n] for r in C["reps"] if n in r["shifts"]]
            C["pooled"][n] = {k: mean([m[k] for m in ms]) for k in ("t90_cold", "t_2x", "t_1pct", "t90_req", "t90_w3")}
            C["pooled"][n]["per_rep_t90_cold"] = [m["t90_cold"] for m in ms]
            C["pooled"][n]["per_rep_t_2x"] = [m["t_2x"] for m in ms]
            C["pooled"][n]["n"] = len(ms)
        steady = [n for n in names if n in GATED or (legacy and n in ("boot->jpde",))] if not C["third"] else \
            [n for n in names if "third" in n]
        G = [r["shifts"][n] for r in C["reps"] for n in steady if n in r["shifts"]]
        ch = [m for m in G if m["churn_n"] is not None]
        C["churn_per_min"] = round(sum(m["churn_n"] for m in ch) / sum(m["churn_min"] for m in ch), 2) if ch else None
        sel = sum(m["conv_sel"] for m in G)
        C["conv_cold"] = round(sum(m["conv_cold_n"] for m in G) / sel, 5) if sel else None
        stl = [m["stale"] for m in G if m["stale"] is not None]
        C["stale"] = sum(stl) if stl else None
        errs = [r["server_errors"] for r in C["reps"] if r["server_errors"] is not None]
        C["server_errors"] = sum(errs) if errs else None
        c1 = {}
        for r in C["reps"]:
            for n, m in r["shifts"].items():
                dom = n.split("->")[1]
                if n in GATED or (legacy and n == "boot->jpde") or "third" in n:
                    c1.setdefault(dom, []).extend(m["conv_tok"])
        C["c1_conv"] = {d: round(statistics.median(v), 2) for d, v in c1.items() if v}
        spans = [(r["shifts"][n]["t0"], r["shifts"][n]["t1"]) for r in C["reps"] for n in steady if n in r["shifts"]]
        C["g2"] = g2_cell(boot, spans, C["docs"])
        for r in C["reps"]:
            for m in r["shifts"].values():
                m.pop("conv_tok", None)
        C.pop("docs")
    return cells


def ratio(a, b):
    return None if (a is None or b in (None, 0)) else round(a / b, 3)


def fmt(x, nd=1):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x))


def cell_gates(cells, boot, c):
    """Per-cell gates vs same-boot B0. Returns dict name->[status,text] and summary."""
    C, B0 = cells.get(f"{boot}:{c}"), cells.get(f"{boot}:B0")
    g, s = {}, {"cell": c}
    if C is None or B0 is None:
        why = f"{c} cell missing" if C is None else "same-boot B0 missing"
        for k in ("T90", "G1", "G2", "G3"):
            g[f"{k}_{c}"] = ["MISSING", why]
        s.update(reg_pass=False, status="MISSING")
        return g, s
    # T90 halving (registered) + tail instrument
    parts, reg, tails, marg = [], True, [], False
    mde = MDE_POOLED if min(C["pooled"].get(n, {}).get("n", 0) for n in GATED) >= 3 else MDE_SINGLE
    for n in GATED:
        a = C["pooled"].get(n, {}).get("t90_cold")
        b = B0["pooled"].get(n, {}).get("t90_cold")
        r = ratio(a, b)
        a2, b2 = C["pooled"].get(n, {}).get("t_2x"), B0["pooled"].get(n, {}).get("t_2x")
        r2 = ratio(a2, b2)
        tails.append(r2)
        ok = r is not None and r <= 0.5
        reg = reg and ok
        if a is not None and b is not None and abs(a - 0.5 * b) < mde:
            marg = True
        parts.append(f"{n}: T90_cold {fmt(a)} vs B0 {fmt(b)} s (×{fmt(r, 2)}); <2×conv {fmt(a2)} vs {fmt(b2)} (×{fmt(r2, 2)})")
    g[f"T90_{c}"] = ["PASS" if reg else "FAIL", "; ".join(parts) + f" — bar ≤0.50× on both gated shifts (registered 2 s-window cold-share T90); MDE {mde} s"]
    tail = max([t for t in tails if t is not None], default=None)
    s.update(reg_pass=reg, tail_ratio=tail, marginal=marg)
    # G1
    st, tx = "PASS", []
    for d in ("jpde", "agent"):
        a, b = C["c1_conv"].get(d), B0["c1_conv"].get(d)
        r = ratio(a, b)
        if r is None:
            st = "MISSING"
        elif r < 1 - G1_TOL and st != "MISSING":
            st = "FAIL"
        tx.append(f"{d} {fmt(a)} vs B0 {fmt(b)} tok/s ({'—' if r is None else f'{100 * (r - 1):+.2f}%'})")
    g[f"G1_{c}"] = [st, "; ".join(tx) + " — converged C1 (final 60 s, pooled reps), bar ≥ −2% vs same-boot B0"]
    # G2
    G2 = C["g2"]
    if "unmeasured" in G2:
        g[f"G2_{c}"] = ["UNMEASURED", G2["unmeasured"]]
    else:
        r = G2["p95_ratio"]
        g[f"G2_{c}"] = ["PASS" if r <= G2_BAR else "FAIL",
                        f"p95 swap/non-swap host-step ratio {r} (p90 {G2['p90_ratio']}, p99 {G2['p99_ratio']} reported) over "
                        f"{G2['bins']} matched 5 s bins, {G2['n_swap_steps']} swap steps; {G2['mean_ms_per_applied_swap']} ms/applied swap; "
                        f"cuda_ms mean {G2['cuda_ms_mean']}; chunk-gap p95 ratio {G2['chunk_gap_p95_ratio']} — bar ≤ {G2_BAR}"]
    # G3
    cr = ratio(C["churn_per_min"], B0["churn_per_min"])
    sub = {"churn": None if cr is None else cr <= G3_CHURN,
           "stale": None if C["stale"] is None else C["stale"] == 0,
           "errors": None if C["server_errors"] is None else C["server_errors"] == 0,
           "conv_cold": None if C["conv_cold"] is None else C["conv_cold"] < G3_CONV_COLD}
    st = "FAIL" if any(v is False for v in sub.values()) else ("UNMEASURED" if any(v is None for v in sub.values()) else "PASS")
    g[f"G3_{c}"] = [st, f"churn {fmt(C['churn_per_min'])}/min vs B0 {fmt(B0['churn_per_min'])} (×{fmt(cr, 2)}, bar ≤{G3_CHURN}); "
                        f"stale {C['stale']}; server errors {C['server_errors']}; conv cold {fmt(C['conv_cold'], 5)} (bar <0.01)"
                        + ("" if all(v is not None for v in sub.values()) else
                           f" — unmeasured: {[k for k, v in sub.items() if v is None]}")]
    s["g3_churn_ok"] = sub["churn"]
    greens = [g[f"G1_{c}"][0], g[f"G2_{c}"][0], g[f"G3_{c}"][0]]
    if not reg:
        s["status"] = "FAIL"
    elif "FAIL" in greens:
        s["status"] = "FAIL"
    elif "MISSING" in greens or "UNMEASURED" in greens:
        s["status"] = "UNMEASURED"
    elif marg or tail is None or tail > TAIL_ZONE:
        s["status"] = "JAMES"
    else:
        s["status"] = "PASS"
    s["greens"] = greens
    return g, s


def parity_misses(p):
    d = J(p)
    if not d:
        return None
    return {"exact": d.get("exact"), "n": d.get("n"), "ids": [m["id"] for m in d.get("first_misses", [])]}


def gauntlet_gate(R, tag):
    gt = J(R / f"gauntlet-{tag}" / "summary.json")
    if not gt:
        return ["MISSING", f"gauntlet-{tag}/summary.json missing"]
    ok = gt.get("success_count") == gt.get("task_count") == 12
    return ["PASS" if ok else "FAIL", f"{gt.get('success_count')}/{gt.get('task_count')} — bar 12/12"]


def verdict(R: Path, legacy_run1: Path | None = None):
    gates, notes = {}, []
    if legacy_run1:
        boots = {"run1": Boot(legacy_run1)}
        phases = []
        for f, cell in (("shift-ar2-adaptive.json", "B0"), ("shift-ar2-frozen.json", "FROZEN")):
            d = J(legacy_run1 / f)
            if not d:
                continue
            prev = "boot"
            for i, (k, ph) in enumerate(d["phases"].items()):
                dom = "jpde" if k.startswith("jpde") else "agent"
                phases.append({"ev": "phase", "boot": "run1", "cell": cell, "rep": 1, "tag": f.replace(".json", ""), "phase": k,
                               "domain": dom, "prev": prev, "idx": i, "t0": ph["wall_start"], "t1": ph["wall_end"], "file": f})
                prev = dom
        notes.append("LEGACY run1 mode: AR2 protocol (static boot, JP 180 s -> agent -> JP, adaptive from t=0 of jpde_1 with "
                     "EWMA baseline at GO), not the AR6 pass; jpde_2 stands in for agent->jpde (a RETURN shift, soft), "
                     "agent for jpde->agent. No same-boot comparison cells exist.")
    else:
        boots = {b: Boot(R / d) for b, d in (("0", "boot0"), ("A", "bootA")) if (R / d).exists()}
        phases = jl(R / "phases.jsonl")
    cells = build_cells(R, phases, boots, legacy=bool(legacy_run1))
    summ = {}
    if legacy_run1:
        for c in ("B1", "B2", "B3"):
            for k in ("T90", "G1", "G2", "G3"):
                gates[f"{k}_{c}"] = ["UNMEASURED", "no B cells in run1 (single adaptive shift run)"]
        B0 = cells.get("run1:B0")
        if B0:
            G2 = B0["g2"]
            gates["G2_B0"] = ["UNMEASURED", G2.get("unmeasured", "")] if "unmeasured" in G2 else ["REPORT", json.dumps(G2)]
            gates["G3_B0_absolute"] = ["REPORT", f"churn {B0['churn_per_min']}/min (W3 class 62–110); stale {B0['stale']}; "
                                                 f"server errors {B0['server_errors']} (no errcount in run1 → errors UNMEASURED); conv cold {B0['conv_cold']}"]
    else:
        for c in ("B1", "B2", "B3"):
            if f"A:{c}" not in cells and c == "B3":
                gates["T90_B3"] = ["REPORT", "B3 not run (conditional on B2 G3 churn fail; default skip)"]
                continue
            g, s = cell_gates(cells, "A", c)
            gates.update(g)
            summ[c] = s
        B0A = cells.get("A:B0")
        if B0A:
            sub = {"stale": B0A["stale"] == 0 if B0A["stale"] is not None else None,
                   "errors": B0A["server_errors"] == 0 if B0A["server_errors"] is not None else None,
                   "conv_cold": B0A["conv_cold"] < G3_CONV_COLD if B0A["conv_cold"] is not None else None}
            gates["G3_B0"] = ["REPORT", f"control cell (churn ratio 1 by definition): churn {B0A['churn_per_min']}/min; stale {B0A['stale']}; "
                                        f"errors {B0A['server_errors']}; conv cold {B0A['conv_cold']} — sub-checks {sub}"]
            G2 = B0A["g2"]
            gates["G2_B0"] = ["UNMEASURED", G2["unmeasured"]] if "unmeasured" in G2 else \
                ["PASS" if G2["p95_ratio"] <= G2_BAR else "FAIL", f"control cell p95 ratio {G2['p95_ratio']} (reported; feeds AR5)"]
    # ---- H2
    if legacy_run1:
        h2 = ["UNMEASURED", "no B1/B2/B3 cells and no same-boot B0 → H2 cannot be evaluated on run1"]
        winner = None
    else:
        winner, h2 = None, None
        for c in PREF:
            s = summ.get(c)
            if not s or s["status"] in ("FAIL", "MISSING"):
                continue
            if s["status"] == "PASS":
                winner, h2 = c, ["PASS", f"winner {c} (R={RATE[c]} swaps/step): halves cold-share T90 on both gated shifts, G1–G3 green, "
                                         f"<2×conv ratio {s['tail_ratio']} ≤ {TAIL_ZONE}"]
            elif s["status"] == "JAMES":
                winner, h2 = c, ["JAMES", f"JAMES DECIDES: {c} halves the registered instrument with G1–G3 green, but "
                                          + ("it is within the pooled MDE of the 0.5× bar" if s["marginal"] else "")
                                          + ("; " if s["marginal"] and (s["tail_ratio"] is None or s["tail_ratio"] > TAIL_ZONE) else "")
                                          + (f"the <2×conv instrument is in the coin-flip zone (×{s['tail_ratio']} > {TAIL_ZONE})"
                                             if (s["tail_ratio"] is None or s["tail_ratio"] > TAIL_ZONE) else "")
                                          + " — both instruments reported (spec §Gates)"]
            else:
                h2 = ["UNMEASURED", f"{c} halves the registered instrument but greens {s['greens']} include UNMEASURED/MISSING"]
                winner = c
            break
        if h2 is None:
            h2 = ["FAIL", "no B cell halves cold-share T90 vs same-boot B0 on both gated shifts with G1–G3 green: "
                          + "; ".join(f"{c} {summ[c]['status']}" for c in PREF if c in summ)]
    # ---- G4
    Ad = R / "bootA"
    a0d = R / "boot0"
    if legacy_run1:
        for k in ("G4_gauntlet_ArmA", "G4_parity_ArmA", "G4_gauntlet_winner", "G4_parity_winner"):
            gates[k] = ["UNMEASURED", "run1 has no AR6 G4 captures (parity-compare.json is AR2 pre/post only)"]
        pc = J(legacy_run1 / "parity-compare.json")
        if pc:
            gates["G4_parity_run1_prepost"] = ["REPORT", f"AR2 frozen vs post-swap parity {pc.get('exact')}/{pc.get('n')} "
                                                         f"misses {[m['id'] for m in pc.get('first_misses', [])]}"]
    else:
        nf = parity_misses(Ad / "cmp-ArmA-F1-vs-ArmA-F2.json")
        ns = parity_misses(a0d / "cmp-Arm0-S1-vs-Arm0-S2.json")
        noise = set(NOISE_IDS) | set((nf or {}).get("ids", [])) | set((ns or {}).get("ids", []))
        gates["G4_gauntlet_ArmA"] = gauntlet_gate(R, "ArmA")
        x = parity_misses(Ad / "cmp-Arm0-S1-vs-ArmA-F1.json")
        gates["G4_parity_ArmA"] = ["MISSING", "cmp-Arm0-S1-vs-ArmA-F1.json missing"] if x is None else \
            ["PASS" if set(x["ids"]) <= noise else "FAIL",
             f"static (Arm0-S1) vs seed (ArmA-F1) {x['exact']}/{x['n']} misses {x['ids']}; noise set {sorted(noise)} "
             f"(F1vsF2 {nf and nf['ids']}, S1vsS2 {ns and ns['ids']} + documented)"]
        if winner:
            gates["G4_gauntlet_winner"] = gauntlet_gate(R, winner)
            ms = [parity_misses(Ad / f"cmp-ArmA-F1-vs-{winner}-W{i}.json") for i in (1, 2)]
            if any(m is None for m in ms):
                gates["G4_parity_winner"] = ["MISSING", f"cmp-ArmA-F1-vs-{winner}-W1/W2 missing"]
            else:
                ok = all(set(m["ids"]) <= noise for m in ms)
                gates["G4_parity_winner"] = ["PASS" if ok else "FAIL",
                                             f"fixed seed (F1) vs {winner} live W1 {ms[0]['exact']}/{ms[0]['n']} {ms[0]['ids']}; "
                                             f"vs frozen-adapted W2 {ms[1]['exact']}/{ms[1]['n']} {ms[1]['ids']}; noise {sorted(noise)}"]
            if h2[0] == "PASS" and (gates["G4_gauntlet_winner"][0] != "PASS" or gates["G4_parity_winner"][0] != "PASS"):
                h2 = ["JAMES", h2[1] + f" — BUT G4 winner gauntlet {gates['G4_gauntlet_winner'][0]} / parity "
                                       f"{gates['G4_parity_winner'][0]}: not promotable as-is"]
        else:
            gates["G4_gauntlet_winner"] = ["REPORT", "no winner cell → winner G4 not run"]
    # ---- H1
    if legacy_run1:
        for k in ("H1_mixed", "H1_agent", "H1_adaptive_steady"):
            gates[k] = ["UNMEASURED", "no held-out mixed C1 on a seed boot in run1 (H1 needs Arm A + Arm 0 frozen cells)"]
        h1 = ["UNMEASURED", "H1 needs AR6 Boot 1 + Boot 2"]
    else:
        mA = J(Ad / "mixed-A-frozen-heldout-c1.json")
        m0 = J(a0d / "mixed-A0-frozen-heldout-c1.json")
        if not mA:
            gates["H1_mixed"] = ["MISSING", "mixed-A-frozen-heldout-c1.json missing"]
            gates["H1_agent"] = ["MISSING", "mixed-A-frozen-heldout-c1.json missing"]
        else:
            a = mA["agg_tok_s_post60"]
            ref, src = (m0["agg_tok_s_post60"], "same-window A0-frozen held-out") if m0 else (CLASS_MIXED_FROZEN, "CONSTANT class 133.1 (A0 missing)")
            r = ratio(a, ref)
            gates["H1_mixed"] = ["PASS" if r is not None and r >= H1_MIXED else "FAIL",
                                 f"A-frozen held-out mixed C1 agg post60 {a} vs {src} {ref} (×{r}; vs 133.1 class ×{ratio(a, CLASS_MIXED_FROZEN)}) — bar ≥ ×{H1_MIXED}"]
            aa = (mA["per_req"].get("agent") or {}).get("median_post60")
            ra, srca = ((m0["per_req"].get("agent") or {}).get("median_post60"), "same-window A0") if m0 else (CLASS_AGENT, "CONSTANT 195.7 class")
            r = ratio(aa, ra)
            gates["H1_agent"] = ["PASS" if r is not None and r >= 1 - H1_AGENT_TOL else "FAIL",
                                 f"A-frozen held-out agent C1 {aa} vs {srca} {ra} (×{r}; vs 195.7 class ×{ratio(aa, CLASS_AGENT)}) — bar ≥ −2%"]

            def first(m, dom):
                rows = sorted([x for x in (m or {}).get("rows", []) if x.get("dom") == dom and x.get("tok_s")], key=lambda x: x["t_start"])
                return rows[0]["tok_s"] if rows else None
            gates["H1_first_request"] = ["REPORT", f"first-request tok/s A-frozen jpde {first(mA, 'jpde')} agent {first(mA, 'agent')} | "
                                                   f"A0-frozen jpde {first(m0, 'jpde')} agent {first(m0, 'agent')}"]
        cA, c0 = cells.get("A:B0"), cells.get("0:B0")
        if not cA or not c0:
            gates["H1_adaptive_steady"] = ["MISSING", "A-adaptive (A:B0) or Arm 0 B0 cell missing"]
        else:
            st, tx = "PASS", []
            for d in ("jpde", "agent"):
                r = ratio(cA["c1_conv"].get(d), c0["c1_conv"].get(d))
                if r is None:
                    st = "MISSING"
                elif abs(r - 1) > H1_STEADY_TOL and st != "MISSING":
                    st = "FAIL"
                tx.append(f"{d} A {cA['c1_conv'].get(d)} vs Arm0 {c0['c1_conv'].get(d)} ({'—' if r is None else f'{100 * (r - 1):+.2f}%'})")
            gates["H1_adaptive_steady"] = [st, "; ".join(tx) + " — A-adaptive converged C1 matches B0 (Arm 0), bar |Δ| ≤ 2%"]
        hs = [gates[k][0] for k in ("H1_mixed", "H1_agent", "H1_adaptive_steady")]
        if all(x == "PASS" for x in hs):
            h1 = ["PASS", "seed map: held-out mixed ≥ +20%, agent within 2%, adaptive steady state matches B0"]
            if gates["G4_gauntlet_ArmA"][0] != "PASS" or gates["G4_parity_ArmA"][0] != "PASS":
                h1 = ["JAMES", h1[1] + f" — BUT G4 Arm A gauntlet {gates['G4_gauntlet_ArmA'][0]} / parity {gates['G4_parity_ArmA'][0]}"]
        elif "FAIL" in hs:
            h1 = ["FAIL", f"H1 sub-gates {dict(zip(('mixed', 'agent', 'steady'), hs))}"]
        else:
            h1 = ["UNMEASURED", f"H1 sub-gates {dict(zip(('mixed', 'agent', 'steady'), hs))}"]
    gates["H1"] = h1
    gates["H2"] = h2
    # third domain (non-gating)
    for k, C in cells.items():
        if C["third"]:
            gates[f"third_{C['cell']}"] = ["REPORT", "; ".join(f"{n}: T90_cold {fmt(v['t90_cold'])} s, <2×conv {fmt(v['t_2x'])}, ≤1% {fmt(v['t_1pct'])}"
                                                               for n, v in C["pooled"].items()) + " (non-gating, one pass)"]
    short = {"PASS": "PASS", "FAIL": "FAIL", "JAMES": "JAMES", "UNMEASURED": "UNM", "MISSING": "MISS"}
    if h2[0] == "JAMES" or h1[0] == "JAMES":
        res = "JAMES DECIDES"
    elif h1[0] == h2[0] == "PASS":
        res = "PASS"
    elif h1[0] == h2[0] == "FAIL":
        res = "FAIL"
    elif "UNMEASURED" in (h1[0], h2[0]) and not ("PASS" in (h1[0], h2[0]) or "FAIL" in (h1[0], h2[0])):
        res = "UNMEASURED"
    else:
        res = "MIXED"
    suffix = f"AR6-H1{short[h1[0]]}-H2{short[h2[0]]}" + (f"-{winner}" if winner and h2[0] in ("PASS", "JAMES") else "")
    unmeasured = sorted(k for k, v in gates.items() if v[0] == "UNMEASURED")
    out = {"card": "AR6 v2", "result": res, "suffix": suffix, "winner": winner, "gates": gates, "cells": cells,
           "unmeasured_gates": unmeasured, "notes": notes,
           "instrument": {"win_s": WIN_S, "conv_s": CONV_S, "peak_s": PEAK_S, "hold_windows": HOLD, "bin_s": BIN_S,
                          "mde_pooled_s": MDE_POOLED, "tail_zone": TAIL_ZONE}}
    return out, summ


def table(v):
    L = [f"Card AR6 v2 verdict: {v['result']}   suffix {v['suffix']}   winner {v['winner']}", ""]
    L.append(f"{'cell':<14}{'shift':<16}{'T90cold':>8}{'<2xconv':>8}{'<=1%':>7}{'T90req':>8}{'T90w3':>7}{'peak':>8}{'conv':>8}  reps T90cold")
    for k, C in v["cells"].items():
        for n, p in C["pooled"].items():
            reps = C["pooled"][n]["per_rep_t90_cold"]
            conv = mean([r["shifts"][n]["conv_cold"] for r in C["reps"] if n in r["shifts"]])
            peak = mean([r["shifts"][n]["peak_cold"] for r in C["reps"] if n in r["shifts"]])
            L.append(f"{k:<14}{n:<16}{fmt(p['t90_cold']):>8}{fmt(p['t_2x']):>8}{fmt(p['t_1pct']):>7}{fmt(p['t90_req']):>8}"
                     f"{fmt(p['t90_w3']):>7}{fmt(peak, 4):>8}{fmt(conv, 4):>8}  {reps}")
        L.append(f"{'':<14}churn {fmt(C['churn_per_min'])}/min  conv_cold {fmt(C['conv_cold'], 5)}  stale {C['stale']}  "
                 f"errors {C['server_errors']}  C1conv {C['c1_conv']}  G2 {C['g2'].get('p95_ratio', C['g2'].get('unmeasured'))}")
    L += ["", f"{'gate':<26}{'status':<12}detail"]
    for k, (st, tx) in v["gates"].items():
        L.append(f"{k:<26}{st:<12}{tx[:260]}")
    if v["unmeasured_gates"]:
        L += ["", "UNMEASURED: " + ", ".join(v["unmeasured_gates"])]
    for n in v["notes"]:
        L.append("NOTE: " + n)
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("R", nargs="?")
    ap.add_argument("--pick-winner", action="store_true")
    ap.add_argument("--b3-check", action="store_true")
    ap.add_argument("--legacy-run1")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.legacy_run1:
        R = Path(a.legacy_run1)
        v, _ = verdict(R, legacy_run1=R)
        out = Path(a.out) if a.out else None
    else:
        R = Path(a.R)
        v, summ = verdict(R)
        if a.pick_winner:
            w = "NONE"
            for c in PREF:
                s = summ.get(c)
                if s and s.get("reg_pass") and s.get("status") in ("PASS", "JAMES", "UNMEASURED") and \
                        "FAIL" not in (s.get("greens") or []):
                    w = c
                    break
            print(f"WINNER={w}")
            return 0
        if a.b3_check:
            s = summ.get("B2")
            ok = s.get("g3_churn_ok") if s else None
            g = v["gates"].get("G3_B2", ["MISSING", "no B2"])
            print(("B3=RUN" if ok is False else "B3=SKIP") + f" (B2 G3 churn ok={ok}; {g[1][:200]})")
            return 0
        out = Path(a.out) if a.out else R / "verdict.json"
    if out:
        out.write_text(json.dumps(v, indent=1, default=str) + "\n")
    print(table(v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
