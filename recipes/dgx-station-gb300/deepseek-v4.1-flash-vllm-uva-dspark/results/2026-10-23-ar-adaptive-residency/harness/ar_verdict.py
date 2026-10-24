#!/usr/bin/env python3
"""AR verdict: gates for AR0/AR1/AR2 from the run directory.

Usage: ar_verdict.py <run_dir>
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

RUN = Path(sys.argv[1])
POST_S = 60.0  # ignore the first 60 s of each phase (convergence window)


def load(name):
    p = RUN / name
    return json.loads(p.read_text()) if p.exists() else None


def tel():
    p = RUN / "telemetry.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def phase_stats(shift, T):
    out = {}
    if not shift:
        return out
    for name, ph in shift["phases"].items():
        s_all = [s["instrument_tok_s"] for s in ph["samples"] if s.get("instrument_tok_s")]
        s_post = [s["instrument_tok_s"] for s in ph["samples"]
                  if s.get("instrument_tok_s") and s.get("elapsed_s", 0) >= POST_S]
        w0, w1 = ph.get("wall_start"), ph.get("wall_end")
        sel = cold = 0
        sel_p = cold_p = 0
        if w0 and w1:
            for r in T:
                if r.get("dsteps") and w0 <= r["ts"] <= w1:
                    sel += r["sel"]
                    cold += r["cold"]
                    if r["ts"] >= w0 + POST_S:
                        sel_p += r["sel"]
                        cold_p += r["cold"]
        out[name] = {
            "n": len(s_all),
            "tok_s_median_all": round(statistics.median(s_all), 2) if s_all else None,
            "tok_s_median_post60": round(statistics.median(s_post), 2) if s_post else None,
            "cold_share_all": round(cold / sel, 4) if sel else None,
            "cold_share_post60": round(cold_p / sel_p, 4) if sel_p else None,
        }
    return out


def pct(a, b):
    return None if (a is None or b in (None, 0)) else round(100.0 * (a - b) / b, 2)


T = tel()
v = {"run": str(RUN)}
ar0 = load("ar0-unit.json")
v["AR0"] = {"pass": bool(ar0 and ar0.get("pass")),
            "tests": {k: t["ok"] for k, t in (ar0 or {}).get("tests", {}).items()}}
ar1 = load("ar1.json")
if ar1:
    ab = ar1["ab"]
    v["AR1"] = {
        "count_on_median": ab["median_on"],
        "count_off_median": ab["median_off"],
        "delta_pct_median": ab["delta_pct_median"],
        "paired_delta_pct_median": ab["paired_delta_pct_median"],
        "off_spread_pct": ab["spread_off_pct"],
        "same_tokens": ab["same_tokens"],
        "gate_overhead_le_0p5pct": (ab["paired_delta_pct_median"] is not None
                                    and ab["paired_delta_pct_median"] >= -0.5),
        "ttft": [{k: r[k] for k in ("words", "rep", "prompt_tokens", "ttft_s", "prefill_tok_s")}
                 for r in ar1["ttft"]["rows"]],
        "sanity": ar1["sanity"],
    }
fz = phase_stats(load("shift-ar2-frozen.json"), T)
ad = phase_stats(load("shift-ar2-adaptive.json"), T)
v["AR2_frozen"] = fz
v["AR2_adaptive"] = ad
cmp_ = {}
for name in ("jpde_1", "agent", "jpde_2"):
    f, a = fz.get(name, {}), ad.get(name, {})
    cmp_[name] = {
        "tok_s_post60_frozen": f.get("tok_s_median_post60"),
        "tok_s_post60_adaptive": a.get("tok_s_median_post60"),
        "tok_s_delta_pct": pct(a.get("tok_s_median_post60"), f.get("tok_s_median_post60")),
        "cold_post60_frozen": f.get("cold_share_post60"),
        "cold_post60_adaptive": a.get("cold_share_post60"),
    }
v["AR2_compare"] = cmp_
par = None
pc = RUN / "parity-compare.json"
if pc.exists():
    par = json.loads(pc.read_text())
v["AR2_parity"] = par
sw = RUN / "swaps.jsonl"
v["swaps_logged"] = sum(1 for _ in sw.open()) if sw.exists() else 0
last = [r for r in T if "swaps" in r]
v["telemetry_last"] = last[-1] if last else None

g = {}
g["AR0_unit"] = v["AR0"]["pass"]
g["AR1_overhead"] = v.get("AR1", {}).get("gate_overhead_le_0p5pct")
jp = [cmp_[n]["tok_s_delta_pct"] for n in ("jpde_1", "jpde_2") if cmp_[n]["tok_s_delta_pct"] is not None]
g["AR2_jpde_speed_ge_5pct"] = bool(jp) and min(jp) >= 5.0
ag = cmp_["agent"]["tok_s_delta_pct"]
g["AR2_agent_no_regress_3pct"] = ag is not None and ag >= -3.0
cf = [(cmp_[n]["cold_post60_frozen"], cmp_[n]["cold_post60_adaptive"]) for n in ("jpde_1", "jpde_2")]
g["AR2_jpde_cold_halved"] = all(f and a is not None and a <= 0.5 * f for f, a in cf)
g["AR2_parity_exact"] = bool(par) and par.get("exact") == par.get("n")
v["gates"] = g
v["verdict"] = "PASS" if all(g.values()) else "FAIL" if g.get("AR2_parity_exact") is False and par else "MIXED"
print(json.dumps(v, indent=1))
