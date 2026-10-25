#!/usr/bin/env python3
"""kv3_verdict.py — Card KV3 full-battery verdict (KV3-SPEC-DRAFT.md gates; DECISIONS in DEPLOY-KV3.md). Pure stdlib.

Usage:  kv3_verdict.py <R> [--out verdict.json]   -> <R>/verdict.json + human table on stdout. rc 0 PASS, 1 FAIL, 2 no input.

Inputs (<R> = ~/dsv41/m3/kv3/results-cardKV3, written by cardKV3_runner.sh):
  <arm>/facts.json            kv_tokens ("4,400,013"), connector_lines, pin_warn_lines, kvdecl_in_args, tier_in_args
  <arm>/mixed-<arm>-c1-r<k>.json   ar_mixed_v2 HELDOUT=1 C1 (agg_tok_s_all primary, agg_tok_s_post60 reported, errors)
  <arm>/cold64k.json          probes[] (probe 0 dropped), kept_median_wall_s
  <arm>/warm.json             runs[r1,r2].decode_tok_s, mode, forced_ok
  <arm>/needle1m.json         recall, prompt_tokens, wall_s
  K1/evict.json               lengths[]: hit, load, identical, recall, speedup ; K0*/evictctl.json (info: cross-arm output1)
  <arm>/wd.json               timeouts (status words[2]-base), trips (TRIP lines), alive_at_collect
  errcount-<arm>.txt          FF-regex count on the server log over the cells window (-rtx: sidecar, info only)
  mem.tsv                     "<epoch> <arm:phase> <MemAvailable GiB>" every 5 s ; oom-start.txt / oom-end.txt
  gauntlet-K1/summary.json    success_count / task_count
Every gate is [PASS|FAIL, text]; a gate whose inputs are missing is FAIL with "UNMEASURED: <why>".
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from pathlib import Path

ARMS = ("K0", "K1", "K0R")
CTRL = ("K0", "K0R")
KV_EXPECT = "4,400,013"
C1_FLOOR = 0.02          # G2 C1 bar = max(2 %, 1.5 x pooled control spread)
C1_SPREAD_K = 1.5
TTFT_TOL = 0.05          # cold-64K: K1 <= control mean x 1.05
WARM_TOL = 0.05          # warm r2: K1 >= min(K0, K0R) x 0.95
MEM_FLOOR = 60.0         # GiB, boot + cells, every arm
GAUNTLET_N = 12
NEEDLE_MIN_PT = 800000   # "1M-class": the overflow retry may scale the 950k target down, never below this
C1_METRIC = "agg_tok_s_all"


def jl(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:  # noqa: BLE001
        return None


def rint(p):
    try:
        return int(Path(p).read_text().strip())
    except Exception:  # noqa: BLE001
        return None


def pct(x):
    return "n/a" if x is None else f"{x * 100:+.2f}%"


def load_arm(R, arm):
    A = R / arm
    c = {"present": A.is_dir()}
    c["facts"] = jl(A / "facts.json")
    reps = []
    for f in sorted(glob.glob(str(A / f"mixed-{arm}-c1-r*.json")), key=lambda s: int(s.rsplit("-r", 1)[1].split(".")[0])):
        d = jl(f)
        if d:
            reps.append({"rep": int(f.rsplit("-r", 1)[1].split(".")[0]), "agg_tok_s_all": d.get("agg_tok_s_all"),
                         "agg_tok_s_post60": d.get("agg_tok_s_post60"), "n": d.get("n"), "errors": d.get("errors", 0),
                         "agent_med": ((d.get("per_req") or {}).get("agent") or {}).get("median_all"),
                         "jpde_med": ((d.get("per_req") or {}).get("jpde") or {}).get("median_all")})
    c["c1"] = reps
    v = [r[C1_METRIC] for r in reps if r[C1_METRIC] is not None]
    c["c1_vals"] = v
    c["c1_mean"] = statistics.mean(v) if v else None
    c["c1_spread"] = ((max(v) - min(v)) / statistics.mean(v)) if len(v) >= 2 else None
    c["c1_post60_mean"] = (statistics.mean([r["agg_tok_s_post60"] for r in reps if r["agg_tok_s_post60"] is not None])
                           if any(r["agg_tok_s_post60"] is not None for r in reps) else None)
    c["c1_errors"] = sum(r["errors"] or 0 for r in reps)
    cold = jl(A / "cold64k.json")
    c["cold64k"] = cold
    if cold:
        kept = [p for p in cold.get("probes", []) if not p.get("dropped") and p.get("wall_s") is not None]
        c["cold_kept"] = [p["wall_s"] for p in kept]
        c["cold_med"] = statistics.median(c["cold_kept"]) if kept else None
        c["cold_pt"] = [p.get("prompt_tokens") for p in kept]
        c["cold_errors"] = cold.get("errors", 0)
    else:
        c["cold_kept"], c["cold_med"], c["cold_pt"], c["cold_errors"] = [], None, [], 0
    w = jl(A / "warm.json")
    c["warm"] = w
    c["warm_r1"] = (w or {}).get("r1_decode_tok_s")
    c["warm_r2"] = (w or {}).get("r2_decode_tok_s")
    c["warm_errors"] = 1 if (w and w.get("error")) else 0
    n = jl(A / "needle1m.json")
    c["needle"] = n
    c["needle_errors"] = 1 if (n and n.get("error") and n.get("wall_s") is None) else 0
    c["evict"] = jl(A / "evict.json")
    c["evictctl"] = jl(A / "evictctl.json")
    c["wd"] = jl(A / "wd.json")
    c["err_server"] = rint(R / f"errcount-{arm}.txt")
    c["err_sidecar"] = rint(R / f"errcount-{arm}-rtx.txt")
    return c


def load_mem(R):
    out = {}
    p = R / "mem.tsv"
    if not p.exists():
        return out
    for ln in p.read_text().splitlines():
        parts = ln.split()
        if len(parts) != 3:
            continue
        try:
            g = float(parts[2])
        except ValueError:
            continue
        out.setdefault(parts[1], []).append(g)
    return {k: {"min": min(v), "n": len(v)} for k, v in out.items()}


def verdict(R):
    cells = {a: load_arm(R, a) for a in ARMS}
    mem = load_mem(R)
    gates = {}

    # ---- G1 KV tokens unchanged on every arm
    bad, txt = [], []
    for a in ARMS:
        kv = ((cells[a]["facts"] or {}).get("kv_tokens")) or None
        txt.append(f"{a}={kv or 'MISSING'}")
        if kv != KV_EXPECT:
            bad.append(a)
    gates["G1_kv_tokens"] = ["PASS" if not bad else "FAIL",
                             f"GPU KV cache size must be {KV_EXPECT} on all arms: {' '.join(txt)}" + (f" (bad: {bad})" if bad else "")]

    # ---- G2a C1 speed-neutral
    k1, ctl = cells["K1"], [cells[a] for a in CTRL]
    if k1["c1_mean"] is None or any(c["c1_mean"] is None or c["c1_spread"] is None for c in ctl):
        gates["G2_c1"] = ["FAIL", f"UNMEASURED: C1 reps K0={len(cells['K0']['c1_vals'])} K1={len(k1['c1_vals'])} "
                                  f"K0R={len(cells['K0R']['c1_vals'])} (need >=2 per control, >=1 K1)"]
        g2c1 = None
    else:
        allc = cells["K0"]["c1_vals"] + cells["K0R"]["c1_vals"]
        cmean = statistics.mean(allc)
        pooled = statistics.mean([c["c1_spread"] for c in ctl])
        bar = max(C1_FLOOR, C1_SPREAD_K * pooled)
        rel = k1["c1_mean"] / cmean - 1
        drift = abs(cells["K0"]["c1_mean"] - cells["K0R"]["c1_mean"]) / cmean
        ok = rel >= -bar
        g2c1 = {"ctrl_mean": cmean, "pooled_spread": pooled, "bar": bar, "k1_rel": rel, "bracket_drift": drift}
        gates["G2_c1"] = ["PASS" if ok else "FAIL",
                          f"K1 C1 {C1_METRIC} mean {k1['c1_mean']:.2f} vs K0/K0R {cmean:.2f} ({pct(rel)}); bar -{bar * 100:.2f}% "
                          f"= max(2%, 1.5 x pooled spread {pooled * 100:.2f}%); bracket drift K0->K0R {drift * 100:.2f}%"
                          + (" (drift > bar: control unstable, read with care)" if drift > bar else "")]

    # ---- G2b cold-64K TTFT
    if k1["cold_med"] is None or any(c["cold_med"] is None for c in ctl):
        gates["G2_ttft64k"] = ["FAIL", "UNMEASURED: cold-64K kept probes missing on " +
                               ",".join(a for a in ARMS if cells[a]["cold_med"] is None)]
    else:
        cm = statistics.mean([c["cold_med"] for c in ctl])
        rel = k1["cold_med"] / cm - 1
        gates["G2_ttft64k"] = ["PASS" if rel <= TTFT_TOL else "FAIL",
                               f"K1 cold-64K TTFT-proxy median {k1['cold_med']:.3f}s vs K0/K0R mean {cm:.3f}s ({pct(rel)}; bar +5%) "
                               f"[K0 {cells['K0']['cold_med']:.3f} K0R {cells['K0R']['cold_med']:.3f}; first probe dropped]"]

    # ---- G2c forced-length warm replay r2
    if k1["warm_r2"] is None or any(c["warm_r2"] is None for c in ctl):
        gates["G2_warm_r2"] = ["FAIL", "UNMEASURED: warm r2 decode tok/s missing on " +
                               ",".join(a for a in ARMS if cells[a]["warm_r2"] is None)]
    else:
        lo = min(c["warm_r2"] for c in ctl)
        rel = k1["warm_r2"] / lo - 1
        modes = {a: (cells[a]["warm"] or {}).get("mode") for a in ARMS}
        forced = {a: (cells[a]["warm"] or {}).get("forced_ok") for a in ARMS}
        same = len(set(modes.values())) == 1
        ok = rel >= -WARM_TOL and same
        gates["G2_warm_r2"] = ["PASS" if ok else "FAIL",
                               f"K1 warm r2 {k1['warm_r2']:.2f} tok/s vs min(K0 {cells['K0']['warm_r2']:.2f}, K0R {cells['K0R']['warm_r2']:.2f}) "
                               f"= {lo:.2f} ({pct(rel)}; bar -5%); mode {modes}{'' if same else ' MODE MISMATCH'}; forced_len {forced}"]

    # ---- G3 tier works in-window (K1)
    ev = k1["evict"]
    f1 = k1["facts"] or {}
    if not ev or not ev.get("lengths"):
        gates["G3_tier"] = ["FAIL", "UNMEASURED: K1 evict.json missing/empty"]
    else:
        L = ev["lengths"]
        hits = all(r.get("hit") for r in L)
        ident = all(r.get("identical") for r in L)
        recall = all(r.get("recall") for r in L)
        noerr = all(not r.get("error") for r in L)
        conn = (f1.get("connector_lines") or 0) > 0 and (f1.get("tier_in_args") or 0) > 0
        pin = (f1.get("pin_warn_lines") or 0) == 0
        ctl_clean = all(((cells[a]["facts"] or {}).get("tier_in_args") or 0) == 0 for a in CTRL)
        ok = hits and ident and recall and noerr and conn and pin
        det = "; ".join(f"{r.get('prompt_tokens')}tok(mod256={r.get('mod256')}) hit={r.get('hit')} load={r.get('load')} "
                        f"ident={r.get('identical')} recall={r.get('recall')} x{r.get('speedup')}" for r in L)
        gates["G3_tier"] = ["PASS" if ok else "FAIL",
                            f"evict-replay hits={hits} identical={ident} recall={recall} errors={not noerr} | connector={conn} "
                            f"pin_warn={f1.get('pin_warn_lines')} controls_tier_free={ctl_clean} | {det} (speedup info only)"]

    # ---- G4a watchdog
    bad, txt = [], []
    for a in ARMS:
        w = cells[a]["wd"]
        if not w or w.get("timeouts") is None or w.get("trips") is None:
            bad.append(a); txt.append(f"{a}=UNMEASURED"); continue
        t = f"{a}: timeouts={w['timeouts']} trips={w['trips']} alive={w.get('alive_at_collect')}"
        txt.append(t)
        if w["timeouts"] != 0 or w["trips"] != 0 or not w.get("alive_at_collect"):
            bad.append(a)
    gates["G4_watchdog"] = ["PASS" if not bad else "FAIL", "; ".join(txt)]

    # ---- G4b errors (server FF regex + harness errors); sidecar FF count reported only
    bad, txt = [], []
    for a in ARMS:
        c = cells[a]
        h = c["c1_errors"] + c["cold_errors"] + c["warm_errors"] + c["needle_errors"] + \
            ((c["evict"] or c["evictctl"] or {}).get("errors") or 0)
        if c["err_server"] is None:
            bad.append(a); txt.append(f"{a}: server=UNMEASURED harness={h}"); continue
        txt.append(f"{a}: server={c['err_server']} harness={h} (sidecar={c['err_sidecar']}, info)")
        if c["err_server"] != 0 or h != 0:
            bad.append(a)
    gates["G4_errors"] = ["PASS" if not bad else "FAIL", "; ".join(txt)]

    # ---- G4c needles (1M every arm + K1 evict recall is in G3)
    bad, txt = [], []
    for a in ARMS:
        n = cells[a]["needle"]
        if not n:
            bad.append(a); txt.append(f"{a}=UNMEASURED"); continue
        txt.append(f"{a}: recall={n.get('recall')} pt={n.get('prompt_tokens')} wall={n.get('wall_s')}s")
        if not n.get("recall") or (n.get("prompt_tokens") or 0) < NEEDLE_MIN_PT:
            bad.append(a)
    gates["G4_needles"] = ["PASS" if not bad else "FAIL", "; ".join(txt) + f" (need recall and prompt_tokens >= {NEEDLE_MIN_PT})"]

    # ---- G4d MemAvailable floor (boot + cells, every arm) + no OOM kills
    mins = {}
    for a in ARMS:
        for ph in ("boot", "cells"):
            m = mem.get(f"{a}:{ph}")
            mins[f"{a}:{ph}"] = m["min"] if m else None
    o0, o1 = rint(R / "oom-start.txt"), rint(R / "oom-end.txt")
    oom = (o1 - o0) if (o0 is not None and o1 is not None) else None
    missing = [k for k, v in mins.items() if v is None]
    low = [k for k, v in mins.items() if v is not None and v < MEM_FLOOR]
    ok = not missing and not low and (oom in (None, 0))
    gates["G4_mem"] = ["PASS" if ok else "FAIL",
                       "MemAvailable min GiB " + " ".join(f"{k}={'n/a' if v is None else f'{v:.1f}'}" for k, v in mins.items())
                       + f" (floor {MEM_FLOOR:.0f}); oom_kill_lines_delta={'unreadable' if oom is None else oom}"
                       + (f"; UNMEASURED {missing}" if missing else "") + (f"; LOW {low}" if low else "")]

    # ---- G5 gauntlet on K1
    g = jl(R / "gauntlet-K1" / "summary.json")
    if not g:
        gates["G5_gauntlet"] = ["FAIL", "UNMEASURED: gauntlet-K1/summary.json missing"]
    else:
        s, t = g.get("success_count"), g.get("task_count")
        gates["G5_gauntlet"] = ["PASS" if (s == GAUNTLET_N and t == GAUNTLET_N) else "FAIL", f"K1 gauntlet {s}/{t} (need {GAUNTLET_N}/{GAUNTLET_N})"]

    failed = [k for k, v in gates.items() if v[0] != "PASS"]
    res = "PASS" if not failed else "FAIL"
    short = {"G1_kv_tokens": "G1", "G2_c1": "G2c1", "G2_ttft64k": "G2ttft", "G2_warm_r2": "G2warm", "G3_tier": "G3",
             "G4_watchdog": "G4wd", "G4_errors": "G4err", "G4_needles": "G4ndl", "G4_mem": "G4mem", "G5_gauntlet": "G5"}
    suffix = "KV3PASS" if res == "PASS" else "KV3FAIL-" + "-".join(short[k] for k in failed)

    # cross-arm output identity (info): K1 evict output1 vs K0/K0R evictctl output1 per length
    xarm = []
    if ev and ev.get("lengths"):
        for r in ev["lengths"]:
            row = {"target": r.get("target"), "K1": r.get("output1")}
            for a in CTRL:
                ec = cells[a]["evictctl"] or {}
                m = [x for x in ec.get("lengths", []) if x.get("target") == r.get("target")]
                row[a] = m[0].get("output1") if m else None
            row["identical_to_controls"] = all(row[a] == row["K1"] for a in CTRL if row[a] is not None)
            xarm.append(row)

    slim = {}
    for a in ARMS:
        c = cells[a]
        slim[a] = {"present": c["present"], "facts": c["facts"], "c1": c["c1"], "c1_mean": c["c1_mean"],
                   "c1_spread": c["c1_spread"], "c1_post60_mean": c["c1_post60_mean"],
                   "cold64k_kept_wall_s": c["cold_kept"], "cold64k_median_s": c["cold_med"], "cold64k_prompt_tokens": c["cold_pt"],
                   "warm": {"mode": (c["warm"] or {}).get("mode"), "forced_ok": (c["warm"] or {}).get("forced_ok"),
                            "r1_decode_tok_s": c["warm_r1"], "r2_decode_tok_s": c["warm_r2"],
                            "runs": (c["warm"] or {}).get("runs")},
                   "needle1m": {k: (c["needle"] or {}).get(k) for k in ("prompt_tokens", "wall_s", "recall", "answer", "error")},
                   "wd": {k: (c["wd"] or {}).get(k) for k in ("timeouts", "trips", "alive_at_collect")},
                   "errcount_server": c["err_server"], "errcount_sidecar": c["err_sidecar"],
                   "mem_min_boot": mins.get(f"{a}:boot"), "mem_min_cells": mins.get(f"{a}:cells")}
    slim["K1"]["evict"] = ev
    slim["G2_c1_calc"] = g2c1
    slim["cross_arm_outputs"] = xarm
    slim["mem_by_phase"] = mem
    out = {"card": "KV3 full battery", "result": res, "suffix": suffix, "gates": gates, "cells": slim, "R": str(R)}
    return out


def fmt(x, nd=2):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x))


def table(v):
    L = [f"Card KV3 full battery verdict: {v['result']}   suffix {v['suffix']}", "", "| gate | result | detail |", "|---|---|---|"]
    for k, (r, t) in v["gates"].items():
        L.append(f"| {k} | {r} | {t} |")
    c = v["cells"]
    L += ["", "| arm | KV tokens | C1 mean (all) | C1 spread | C1 post60 | cold64K med s | warm r1 | warm r2 | 1M needle | wd t/trips | err srv | mem min boot/cells |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in ARMS:
        x = c[a]
        sp = "-" if x["c1_spread"] is None else "%.2f%%" % (x["c1_spread"] * 100)
        L.append(f"| {a} | {(x['facts'] or {}).get('kv_tokens', '-')} | {fmt(x['c1_mean'])} | "
                 f"{sp} | {fmt(x['c1_post60_mean'])} | "
                 f"{fmt(x['cold64k_median_s'], 3)} | {fmt(x['warm']['r1_decode_tok_s'])} | {fmt(x['warm']['r2_decode_tok_s'])} | "
                 f"{x['needle1m'].get('recall')} ({fmt(x['needle1m'].get('wall_s'), 1)}s) | {x['wd'].get('timeouts')}/{x['wd'].get('trips')} | "
                 f"{fmt(x['errcount_server'], 0)} | {fmt(x['mem_min_boot'], 1)}/{fmt(x['mem_min_cells'], 1)} |")
    if c.get("cross_arm_outputs"):
        L.append("")
        L.append("cross-arm evict outputs identical to controls (info): " +
                 ", ".join(f"{r['target']}:{r['identical_to_controls']}" for r in c["cross_arm_outputs"]))
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("R")
    ap.add_argument("--out")
    a = ap.parse_args()
    R = Path(a.R)
    if not R.is_dir():
        print(f"no results dir {R}", file=sys.stderr)
        return 2
    v = verdict(R)
    out = Path(a.out) if a.out else R / "verdict.json"
    tmp = str(out) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(v, f, indent=1)
        f.write("\n")
    os.replace(tmp, out)
    print(table(v))
    return 0 if v["result"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
