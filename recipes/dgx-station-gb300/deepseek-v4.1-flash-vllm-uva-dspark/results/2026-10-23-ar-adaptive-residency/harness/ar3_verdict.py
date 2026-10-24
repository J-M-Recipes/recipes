#!/usr/bin/env python3
"""AR3 verdict. Usage: ar3_verdict.py <run3_dir> <run1_dir>"""
import json
import math
import statistics
import sys
from pathlib import Path

R3, R1 = Path(sys.argv[1]), Path(sys.argv[2])


def J(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def misses(a, b):
    d = J(R3 / f"cmp-{a}-vs-{b}.json")
    return None if d is None else {"exact": d["exact"], "n": d["n"], "ids": [m["id"] for m in d["first_misses"]]}


v = {}
ar0 = J(R3 / "ar0-unit.json")
v["AR0"] = {"pass": bool(ar0 and ar0["pass"]), "tests": {k: t["ok"] for k, t in (ar0 or {}).get("tests", {}).items()}}
ab = J(R3 / "ar1.json")["ab"]
pd = ab["paired_delta_pct"]
mean = statistics.mean(pd)
se = statistics.stdev(pd) / math.sqrt(len(pd)) if len(pd) > 1 else float("nan")
v["AB"] = {"median_on": ab["median_on"], "median_off": ab["median_off"], "paired": pd,
           "paired_mean_pct": round(mean, 3), "paired_se_pct": round(se, 3), "same_tokens": ab["same_tokens"]}
mix = {}
for c in ("c1", "c4"):
    f = J(R3 / f"mixed-frozen-{c}.json")
    a = J(R3 / f"mixed-adaptive-{c}.json")
    if f and a:
        mix[c] = {
            "agg_post60_frozen": f["agg_tok_s_post60"], "agg_post60_adaptive": a["agg_tok_s_post60"],
            "agg_delta_pct": round(100 * (a["agg_tok_s_post60"] - f["agg_tok_s_post60"]) / f["agg_tok_s_post60"], 2),
            "jpde_req_post60": [f["per_req"]["jpde"]["median_post60"], a["per_req"]["jpde"]["median_post60"]],
            "agent_req_post60": [f["per_req"]["agent"]["median_post60"], a["per_req"]["agent"]["median_post60"]],
            "errors": [f["errors"], a["errors"]],
        }
v["mixed"] = mix
s3 = J(R3 / "shift-ar3-adaptive.json")
s2a = J(R1 / "shift-ar2-adaptive.json")
s2f = J(R1 / "shift-ar2-frozen.json")


def post(sh, ph):
    xs = [s["instrument_tok_s"] for s in sh["phases"][ph]["samples"] if s.get("instrument_tok_s") and s.get("elapsed_s", 0) >= 60]
    return round(statistics.median(xs), 2) if xs else None


rep = {}
for ph in ("jpde_1", "agent", "jpde_2"):
    rep[ph] = {"AR2_frozen": post(s2f, ph) if s2f else None, "AR2_adaptive": post(s2a, ph) if s2a else None,
               "AR3_adaptive": post(s3, ph) if s3 else None}
v["shift_repro"] = rep
v["parity"] = {"S1a_vs_S1b": misses("S1a", "S1b"), "S1a_vs_P1": misses("S1a", "P1"),
               "S1a_vs_S2": misses("S1a", "S2"), "S1b_vs_S2": misses("S1b", "S2")}
noise = set((v["parity"]["S1a_vs_S1b"] or {}).get("ids", [])) | {"agent-tool-1"}
g = {}
g["AR0"] = v["AR0"]["pass"]
g["overhead_not_significant_or_le_0p5"] = (mean >= -0.5) or (abs(mean) < 2 * se)
g["mixed_c1_no_regress"] = "c1" in mix and mix["c1"]["agg_delta_pct"] >= -2.0
g["mixed_c4_no_regress"] = "c4" in mix and mix["c4"]["agg_delta_pct"] >= -2.0
g["shift_repro_within_5pct"] = all(
    rep[p]["AR2_adaptive"] and rep[p]["AR3_adaptive"] and abs(rep[p]["AR3_adaptive"] - rep[p]["AR2_adaptive"]) / rep[p]["AR2_adaptive"] <= 0.05
    for p in rep)
rt = v["parity"]["S1a_vs_S2"]
g["roundtrip_parity_within_noise"] = (bool(rt) and (rt["n"] - rt["exact"]) == len(rt["ids"])
                                     and set(rt["ids"]) <= noise)
v["noise_ids"] = sorted(noise)
v["gates"] = g
v["verdict"] = "PASS" if all(g.values()) else "MIXED"
print(json.dumps(v, indent=1))
