#!/usr/bin/env python3
"""br2_verdict.py — Card BR2 verdict (DSpark draft length k above the trained block of 5, block verification ON; arms from
<R>/arms.json, default A K9 A2). Pure stdlib. Adapted from br1/br1_verdict.py (DIFF-br2-verdict.txt).

Usage:  br2_verdict.py <R> [--out verdict.json]   -> <R>/verdict.json + human tables on stdout.
        rc 0 PASS, 3 INVALID-CONTROL (a control missing/polluted or the controls disagree), 1 INVALID-ARM (controls fine, a
        K/B arm not VALID), 2 no input.

K arms are EXPLORATORY: their speed is never gated. The card is decided by:
  G0_binding   every arm bound; Args spec == production spec with num_speculative_tokens = table kspec,
               num_speculative_tokens_per_batch_size = table ksched (launcher default on A/A2/B), rejection_sample_method
               "block" iff the table's SPEC_EXTRA is set; every other Arg, Env and -rtx Args == production. kv_tokens must be
               4,400,013 on A/A2; on K/B arms kv_tokens is REPORTED and flagged KV-LOWER if lower (never fails).
  G0b_foreign  any non-127.0.0.1 POST /v1 (bind or cells window) => that arm INVALID
  G1_controls  per class at C1 in the temp-1.0 window: |A-A2|/mean <= 6 % (C8 advisory) else INVALID-CONTROL-A-A2
  G2_cells     every arm completed its cells: perclass-t1 table with all 5 classes x C1/C8, >= 2 mixed C1 reps, gauntlet
               summary, replay_turns passes. (peerbench is an annotation instrument; its presence is reported, not required.)
  G7_watchdog  card watchdog timeouts 0, trips 0, alive at collect, server FF errors 0
A failure on A/A2 makes that control INVALID; on a K/B arm it makes THAT ARM INVALID (not the card FAIL).
Result: any invalid control -> INVALID-CONTROL-<arms> (suffix BR2-INVALID-CONTROL-...); else any invalid K/B arm ->
INVALID-ARM-<arms> (suffix BR2-INVALID-ARM-...); else PASS (suffix BR2PASS). "gates" keeps the KV3/BR1 adapter shape
[PASS|FAIL, text] (INVALID emitted as FAIL with an INVALID-CONTROL(...)/INVALID-ARM(...) prefix); the tri-state is
"gate_status", "invalid_arms", "arm_valid".

ANNOTATIONS (the deliverable): per arm x class — C1/C8 tok/s at temp 1.0 vs mean(A, A2); tokens per verify step (1 + accepted/
drafts) from the decoder's per-class /metrics deltas; the per-position acceptance curve curve[p] = accepted-at-position-(p-1) /
drafts for p = 1..k (vllm:spec_decode_num_accepted_tokens_per_pos_total{...,position="0".."k-1"}, label shape read from a live
BR1 dump 2026-10-25 19:4x: engine="0",model_name="dsv41-flash-uva",position="N"), marginal[p] = curve[p]/curve[p-1]
(curve[0] = 1), and "k* candidate" = the first p whose marginal < 0.5 (">k" if none). Also: mixed C1 curve (runner snapshots),
autotune/boot shape (retune visible), cg_tokens, peerbench tables (peer's harness on our lane).

Inputs (<R> = ~/dsv41/m3/br1/results-cardBR2, written by cardBR2_runner.sh):
  arms.json ; args-before-<V25>.json, env-before-<V25>.json, args-before-<V25>-rtx.json
  <X>/facts.json (+ kspec, ksched, cg_*, autotune{...}), args.json, env.json, rtx-args.json, boot-autotune.log
  <X>/perclass-t1/perclass-table-BR2-<X>-t1.json + perclass-BR2-<X>-t1-r{1,2}-{c1,c8}.json (res[cls].acc_per_pos)
  <X>/mixed-BR2-<X>-c1-r<k>.json ; <X>/metrics/<cell>-{start,end}.txt ; <X>/peerbench-BR2-<X>.json
  gauntlet-<X>/summary.json ; <X>/turns-BR2-<X>.json ; <X>/wd.json ; errcount-<X>[-rtx].txt ; <X>/foreign-*.txt
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import statistics
import sys
from pathlib import Path

DEFAULT_ARMS = ("A", "K9", "A2")
CTRL = ("A", "A2")
CLASSES = ("prose", "structured", "code", "shell_ops", "tool_json")
MODES = ("c1", "c8")
KV_EXPECT = "4,400,013"
V25 = "dsv41-vllm-1017-v25m-kvtier"
SPEC_BLOCK = ',"rejection_sample_method":"block"'
TABLE = {"A": ("", 5, None), "A2": ("", 5, None), "B": (SPEC_BLOCK, 5, None),
         "K7": (SPEC_BLOCK, 7, "[[1,4,7],[5,8,3],[9,32,3]]"), "K9": (SPEC_BLOCK, 9, "[[1,4,9],[5,8,3],[9,32,3]]")}
CTRL_TOL = 0.06          # G1: |A - A2| / mean(A, A2) per class at C1 (temp-1.0 window); C8 advisory
MARGINAL_BAR = 0.5       # k* candidate: first position whose marginal acceptance drops below this
MIN_MIXED = 2
SHORT = {"G0_binding": "G0", "G0b_foreign": "G0b", "G1_controls": "G1", "G2_cells": "G2", "G7_watchdog": "G7"}
POS_RE = re.compile(r'position="(\d+)"')


def jl(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:  # noqa: BLE001
        return None


def rint(p):
    try:
        return int(Path(p).read_text().strip().split()[0])
    except Exception:  # noqa: BLE001
        return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def kvint(s):
    try:
        return int(str(s).replace(",", ""))
    except Exception:  # noqa: BLE001
        return None


def spec_of(args):
    if not isinstance(args, list) or "--speculative-config" not in args:
        return None
    i = args.index("--speculative-config")
    return args[i + 1] if i + 1 < len(args) else None


def metrics_file(p):
    """sum vllm:spec_decode_* counters by name (all label sets) + per-position accepted dict {pos: value}"""
    out, pos = {}, {}
    try:
        for ln in Path(p).read_text().splitlines():
            if not ln.startswith("vllm:spec_decode_"):
                continue
            name = ln.split("{", 1)[0].split(" ", 1)[0]
            try:
                v = float(ln.split()[-1])
            except ValueError:
                continue
            if name in ("vllm:spec_decode_num_accepted_tokens_per_pos_total", "vllm:spec_decode_num_accepted_tokens_per_pos"):
                m = POS_RE.search(ln)
                if m:
                    pos[int(m.group(1))] = pos.get(int(m.group(1)), 0.0) + v
                continue
            out[name] = out.get(name, 0.0) + v
    except Exception:  # noqa: BLE001
        return None
    out["_pos"] = pos
    return out


def mdelta(A, cell):
    s, e = metrics_file(A / "metrics" / f"{cell}-start.txt"), metrics_file(A / "metrics" / f"{cell}-end.txt")
    if s is None or e is None:
        return None
    g = lambda d, k: d.get(f"vllm:spec_decode_{k}_total", d.get(f"vllm:spec_decode_{k}", 0.0))
    dr, dt_, ac = g(e, "num_drafts") - g(s, "num_drafts"), g(e, "num_draft_tokens") - g(s, "num_draft_tokens"), \
        g(e, "num_accepted_tokens") - g(s, "num_accepted_tokens")
    n = max(list(s["_pos"]) + list(e["_pos"]) + [-1]) + 1
    pos = [e["_pos"].get(i, 0.0) - s["_pos"].get(i, 0.0) for i in range(n)] if n else None
    return {"drafts": dr, "draft_tokens": dt_, "accepted": ac, "acc_pos": pos,
            "tokens_per_verify_step": (1 + ac / dr) if dr else None, "accept_rate": (ac / dt_) if dt_ else None,
            "eff_k": (dt_ / dr) if dr else None}


TS = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})")


def autotune_log(p):
    d = {"loaded_n": None, "saved_n": None, "saved_new_n": None, "elapsed_s": None}
    try:
        L = Path(p).read_text().splitlines()
    except Exception:  # noqa: BLE001
        return d
    ts_s, ts_e = [], []
    for ln in L:
        m = re.search(r"Loaded (\d+) configs", ln)
        if m:
            d["loaded_n"] = int(m.group(1))
        m = re.search(r"Saved (\d+) configs.*?(?:\((\d+) new)?$", ln) or re.search(r"Saved (\d+) configs", ln)
        if m:
            d["saved_n"] = int(m.group(1))
            mm = re.search(r"\((\d+) new", ln)
            d["saved_new_n"] = int(mm.group(1)) if mm else None
        t = TS.search(ln)
        if t and "Autotuning process starts" in ln:
            ts_s.append(dt.datetime.strptime(t.group(1), "%Y-%m-%d %H:%M:%S"))
        if t and "Autotuning process ends" in ln:
            ts_e.append(dt.datetime.strptime(t.group(1), "%Y-%m-%d %H:%M:%S"))
    if ts_s and ts_e:
        d["elapsed_s"] = (max(ts_e) - min(ts_s)).total_seconds()
    d["retune"] = bool(d["loaded_n"] is None or (d["saved_new_n"] or 0) > 0)
    return d


def curve_of(acc_pos, drafts, k):
    """curve[p] (p = 1..k) = accepted at position p-1 / drafts; marginal[p] = curve[p] / curve[p-1] (curve[0] = 1)"""
    if not drafts or acc_pos is None:
        return None
    cur = [(acc_pos[i] if i < len(acc_pos) else 0.0) / drafts for i in range(k)]
    marg, prev = [], 1.0
    for c in cur:
        marg.append((c / prev) if prev > 0 else None); prev = c
    ks = next((p + 1 for p, m in enumerate(marg) if m is not None and m < MARGINAL_BAR), None)
    return {"curve": cur, "marginal": marg, "kstar": ks if ks is not None else f">{k}",
            "tokens_per_step_from_curve": 1 + sum(cur)}


def load_perclass(A, arm, k):
    tag = f"BR2-{arm}-t1"
    t = jl(A / "perclass-t1" / f"perclass-table-{tag}.json")
    out = {"table": t is not None, "cells": {}, "x3": {}, "hygiene": (t or {}).get("hygiene"), "acc": {}, "curve": {}, "eff_k": {}}
    S = (t or {}).get("summary") or {}
    for c in CLASSES:
        for m in MODES:
            out["cells"][(c, m)] = ((S.get(c) or {}).get(m) or {}).get("mean")
        out["x3"][c] = ((S.get(c) or {}).get("x3") or {}).get("accept")
        for m in MODES:   # pooled over r1 + r2 from the raw files (the decoder's per-class /metrics deltas)
            acc = steps = drafted = 0.0; pos = None; ok = False
            for r in ("r1", "r2"):
                raw = jl(A / "perclass-t1" / f"perclass-{tag}-{r}-{m}.json")
                x = ((raw or {}).get("res") or {}).get(c) or {}
                if x.get("draft_steps"):
                    acc += x.get("accepted") or 0; steps += x["draft_steps"]; drafted += x.get("drafted") or 0; ok = True
                    ap = x.get("acc_per_pos")
                    if ap is not None:
                        pos = [((pos[i] if pos and i < len(pos) else 0.0) + (ap[i] if i < len(ap) else 0.0))
                               for i in range(max(len(ap), len(pos or [])))]
            out["acc"][(c, m)] = (acc / steps) if (ok and steps) else None
            out["eff_k"][(c, m)] = (drafted / steps) if (ok and steps and drafted) else None
            out["curve"][(c, m)] = curve_of(pos, steps, k) if ok else None
    return out


def load_arm(R, arm, k):
    A = R / arm
    f = jl(A / "facts.json")
    c = {"present": A.is_dir(), "facts": f, "args": jl(A / "args.json"), "env": jl(A / "env.json"),
         "rtx_args": jl(A / "rtx-args.json")}
    c["autotune"] = (f or {}).get("autotune") or autotune_log(A / "boot-autotune.log")
    c["pc"] = load_perclass(A, arm, k)
    reps = []
    for fn in sorted(glob.glob(str(A / f"mixed-BR2-{arm}-c1-r*.json")), key=lambda s: int(s.rsplit("-r", 1)[1].split(".")[0])):
        d = jl(fn)
        if d and d.get("agg_tok_s_all") is not None:
            reps.append({"rep": int(fn.rsplit("-r", 1)[1].split(".")[0]), "agg_tok_s_all": d["agg_tok_s_all"],
                         "errors": d.get("errors", 0), "n": d.get("n"), "temperature": (d.get("sampling") or {}).get("temperature")})
    c["mixed"] = reps
    c["mixed_mean"] = mean([r["agg_tok_s_all"] for r in reps])
    c["gauntlet"] = jl(R / f"gauntlet-{arm}" / "summary.json")
    t = jl(A / f"turns-BR2-{arm}.json")
    if t and t.get("passes"):
        c["turns"] = [{"tool_emitted": sum(1 for r in p if r.get("has_tool") and r.get("emitted")),
                       "tool_turns": sum(1 for r in p if r.get("has_tool")),
                       "errors": sum(1 for r in p if r.get("error"))} for p in t["passes"]]
    else:
        c["turns"] = None
    c["wd"] = jl(A / "wd.json")
    c["err_server"] = rint(R / f"errcount-{arm}.txt")
    c["err_sidecar"] = rint(R / f"errcount-{arm}-rtx.txt")
    fr = {}
    for fn in glob.glob(str(A / "foreign-*.txt")):
        fr[Path(fn).stem.split("-", 1)[1]] = rint(fn)
    c["foreign"] = fr
    c["m_pct1"] = mdelta(A, "pct1")
    mx = [m for m in (mdelta(A, f"c1-r{i}") for i in (1, 2, 3)) if m]
    if mx:
        ac, dr = sum(m["accepted"] for m in mx), sum(m["drafts"] for m in mx)
        n = max(len(m["acc_pos"] or []) for m in mx)
        pos = [sum(((m["acc_pos"] or [])[i] if i < len(m["acc_pos"] or []) else 0.0) for m in mx) for i in range(n)] if n else None
        c["m_mixed"] = {"accepted": ac, "drafts": dr, "tokens_per_verify_step": (1 + ac / dr) if dr else None,
                        "curve": curve_of(pos, dr, k)}
    else:
        c["m_mixed"] = None
    c["m_peer"] = mdelta(A, "peerbench")
    c["peerbench"] = jl(A / f"peerbench-BR2-{arm}.json")
    return c


class G:
    def __init__(self):
        self.bad, self.txt = set(), []

    def status(self, ctrl_bad):
        return "INVALID" if (self.bad & set(CTRL)) else ("FAIL" if self.bad else "PASS")


def verdict(R):
    aj = jl(R / "arms.json") or {}
    arms = tuple(aj.get("arms") or DEFAULT_ARMS)
    table = {}
    for a in arms:
        t = (aj.get("table") or {}).get(a)
        table[a] = (t["spec_extra"], int(t["kspec"]), t.get("ksched")) if t else TABLE[a]
    cells = {a: load_arm(R, a, table[a][1]) for a in arms}
    prod_args = jl(R / f"args-before-{V25}.json")
    prod_env = jl(R / f"env-before-{V25}.json")
    prod_rtx = jl(R / f"args-before-{V25}-rtx.json")
    try:
        prod_spec = json.loads(spec_of(prod_args))
    except Exception:  # noqa: BLE001
        prod_spec = None
    gs = {}

    # ---- G0 binding + wiring
    g = G(); kvflag = {}
    for a in arms:
        f = cells[a]["facts"]
        sx, k, ksch = table[a]
        if not f or not f.get("bound_at"):
            g.bad.add(a); g.txt.append(f"{a}: NOT BOUND"); continue
        bits = [f"kv={f.get('kv_tokens') or 'MISSING'}"]
        if a in CTRL:
            if f.get("kv_tokens") != KV_EXPECT:
                g.bad.add(a); bits.append(f"KV! (want {KV_EXPECT})")
        else:
            kc, kv = kvint(KV_EXPECT), kvint(f.get("kv_tokens"))
            kvflag[a] = None if kv is None else ("KV-LOWER" if kv < kc else ("KV-HIGHER" if kv > kc else "KV-SAME"))
            bits.append(f"({kvflag[a] or 'KV MISSING'}; reported, not gated)")
        args = cells[a]["args"]
        if prod_args is None or args is None or prod_spec is None:
            g.bad.add(a); bits.append("args UNMEASURED"); g.txt.append(f"{a}: " + " ".join(bits)); continue
        bs = spec_of(args)
        try:
            got = json.loads(bs)
        except Exception:  # noqa: BLE001
            got = None
        want = dict(prod_spec, num_speculative_tokens=k)
        if ksch:
            want["num_speculative_tokens_per_batch_size"] = json.loads(ksch)
        if sx:
            want["rejection_sample_method"] = "block"
        spec_ok = got == want
        rest_ok = len(args) == len(prod_args) and all(x == y for x, y in zip(args, prod_args) if x != bs or y != spec_of(prod_args))
        if a in CTRL:
            rest_ok = rest_ok and args == prod_args
        bits.append(f"kspec_in_args={(got or {}).get('num_speculative_tokens')}(want {k}) "
                    f"ksched={json.dumps((got or {}).get('num_speculative_tokens_per_batch_size'), separators=(',', ':'))} "
                    f"block={(got or {}).get('rejection_sample_method') or '-'} spec==table {spec_ok} args-rest==prod {rest_ok}")
        if f.get("kspec") is not None and f.get("kspec") != k:
            spec_ok = False; bits.append(f"facts.kspec {f.get('kspec')}!")
        env, rtx = cells[a]["env"], cells[a]["rtx_args"]
        eok = env is not None and prod_env is not None and env == prod_env
        rok = rtx is not None and prod_rtx is not None and rtx == prod_rtx
        bits += [f"env==prod {eok}", f"rtx_args==prod {rok}"]
        if not (spec_ok and rest_ok and eok and rok):
            g.bad.add(a)
        g.txt.append(f"{a}: " + " ".join(bits))
    gs["G0_binding"] = (g, "; ".join(g.txt))

    # ---- G0b foreign
    g = G()
    for a in arms:
        fr = cells[a]["foreign"]
        n = sum(v or 0 for v in fr.values()) if fr else None
        g.txt.append(f"{a}={'UNMEASURED' if n is None else n} {fr or ''}")
        if (n is None and cells[a]["facts"]) or (n or 0) > 0:
            g.bad.add(a)
    gs["G0b_foreign"] = (g, "non-127.0.0.1 POST /v1 per arm: " + "; ".join(g.txt))

    # ---- G1 controls agree at temp 1.0 (C1 per class; C8 advisory)
    g = G(); c8 = []
    for c in CLASSES:
        va, va2 = (cells[x]["pc"]["cells"][(c, "c1")] if x in cells else None for x in CTRL)
        if va is None or va2 is None:
            g.bad.update([x for x, v in zip(CTRL, (va, va2)) if v is None]); g.txt.append(f"t1/{c}/c1 UNMEASURED"); continue
        d = abs(va - va2) / ((va + va2) / 2)
        if d > CTRL_TOL:
            g.bad.update(CTRL); g.txt.append(f"t1/{c}/c1 A={va:.1f} A2={va2:.1f} d={d * 100:.1f}% > 6%")
        else:
            g.txt.append(f"t1/{c}/c1 d={d * 100:.1f}%")
        ca, ca2 = cells["A"]["pc"]["cells"][(c, "c8")], cells["A2"]["pc"]["cells"][(c, "c8")]
        if ca and ca2:
            c8.append(f"{c}/c8 d={abs(ca - ca2) / ((ca + ca2) / 2) * 100:.1f}%")
    gs["G1_controls"] = (g, "C1 per class (temp 1.0): " + "; ".join(g.txt) + " | C8 (advisory): " + "; ".join(c8))

    # ---- G2 cells complete
    g = G()
    for a in arms:
        x = cells[a]
        if not x["facts"]:
            g.bad.add(a); g.txt.append(f"{a}: no boot -> no cells"); continue
        miss = [f"{c}/{m}" for c in CLASSES for m in MODES if x["pc"]["cells"][(c, m)] is None]
        probs = []
        if miss:
            probs.append(f"perclass-t1 missing {miss if len(miss) < 10 else str(len(miss)) + ' cells'}")
        if len(x["mixed"]) < MIN_MIXED:
            probs.append(f"mixed reps {len(x['mixed'])} < {MIN_MIXED}")
        if not x["gauntlet"]:
            probs.append("gauntlet missing")
        if not x["turns"]:
            probs.append("turns missing")
        pb = x["peerbench"]
        pbs = "absent" if not pb else f"{sum(1 for cc in pb.get('cells', []) if cc.get('status') == 'OK')}/{len(pb.get('cells', []))} OK"
        if probs:
            g.bad.add(a)
        g.txt.append(f"{a}: " + ("; ".join(probs) if probs else "complete") + f" (peerbench {pbs}, info)")
    gs["G2_cells"] = (g, "; ".join(g.txt))

    # ---- G7 watchdog + FF errors
    g = G()
    for a in arms:
        w, e = cells[a]["wd"], cells[a]["err_server"]
        if not cells[a]["facts"]:
            g.bad.add(a); g.txt.append(f"{a}: not bound"); continue
        if not w or w.get("timeouts") is None or w.get("trips") is None or e is None:
            g.bad.add(a); g.txt.append(f"{a}: UNMEASURED (wd={bool(w)} err={e})"); continue
        bad = w["timeouts"] != 0 or w["trips"] != 0 or not w.get("alive_at_collect") or e != 0
        if bad:
            g.bad.add(a)
        g.txt.append(f"{a}: timeouts={w['timeouts']} trips={w['trips']} alive={w.get('alive_at_collect')} ff={e} (sidecar ff={cells[a]['err_sidecar']}){' BAD' if bad else ''}")
    gs["G7_watchdog"] = (g, "; ".join(g.txt))

    # ---- result
    bad_all = set().union(*(v[0].bad for v in gs.values()))
    inv_ctrl = [a for a in arms if a in CTRL and a in bad_all]
    inv_arm = [a for a in arms if a not in CTRL and a in bad_all]
    status = {k: v[0].status(inv_ctrl) for k, v in gs.items()}
    if inv_ctrl:
        res = "INVALID-CONTROL-" + "-".join(inv_ctrl); suffix = "BR2-" + res
    elif inv_arm:
        res = "INVALID-ARM-" + "-".join(inv_arm); suffix = "BR2-" + res
    else:
        res = "PASS"; suffix = "BR2PASS"
    gates = {}
    for k, (g, txt) in gs.items():
        s = status[k]
        if s == "PASS":
            gates[k] = ["PASS", txt]
        else:
            ic = [a for a in arms if a in g.bad and a in CTRL]; ia = [a for a in arms if a in g.bad and a not in CTRL]
            pre = (f"INVALID-CONTROL({','.join(ic)}): " if ic else "") + (f"INVALID-ARM({','.join(ia)}): " if ia else "")
            gates[k] = ["FAIL", pre + txt]

    # ---- annotations
    ann = {}
    spd = {}
    for a in arms:
        rowd = {}
        for c in CLASSES:
            for m in MODES:
                v = cells[a]["pc"]["cells"][(c, m)]
                cm = mean([cells[x]["pc"]["cells"][(c, m)] for x in CTRL if x in cells])
                rowd[f"{c}/{m}"] = {"tok_s": v, "ratio": (v / cm) if (v and cm) else None}
        mm = cells[a]["mixed_mean"]; cmm = mean([cells[x]["mixed_mean"] for x in CTRL if x in cells])
        rowd["mixed/c1"] = {"tok_s": mm, "ratio": (mm / cmm) if (mm and cmm) else None}
        spd[a] = rowd
    ann["speed_t1_vs_ctrl"] = spd
    ann["tokens_per_verify_step_t1"] = {a: {f"{c}/{m}": (None if cells[a]["pc"]["acc"][(c, m)] is None else 1 + cells[a]["pc"]["acc"][(c, m)])
                                            for c in CLASSES for m in MODES} for a in arms}
    ann["tokens_per_verify_step_windows"] = {a: {"pct1": (cells[a]["m_pct1"] or {}).get("tokens_per_verify_step"),
                                                 "mixed_c1": (cells[a]["m_mixed"] or {}).get("tokens_per_verify_step"),
                                                 "peerbench": (cells[a]["m_peer"] or {}).get("tokens_per_verify_step")} for a in arms}
    ann["eff_k"] = {a: {f"{c}/{m}": cells[a]["pc"]["eff_k"][(c, m)] for c in CLASSES for m in MODES} for a in arms}
    ann["per_position"] = {a: {f"{c}/{m}": cells[a]["pc"]["curve"][(c, m)] for c in CLASSES for m in MODES} for a in arms}
    ann["per_position_mixed_c1"] = {a: (cells[a]["m_mixed"] or {}).get("curve") for a in arms}
    ann["kstar_candidate_c1"] = {a: {c: (cells[a]["pc"]["curve"][(c, "c1")] or {}).get("kstar") for c in CLASSES} for a in arms}
    ann["boot"] = {a: {k2: (cells[a]["facts"] or {}).get(k2) for k2 in ("kv_tokens", "boot_s", "kspec", "ksched", "cg_tokens", "cg_over_128",
                                                                        "cg_tokens_sched", "cg_sizes_max")} | {"kv_flag": kvflag.get(a),
                                                                                                              "autotune": cells[a]["autotune"]} for a in arms}
    ann["gauntlet"] = {a: [(cells[a]["gauntlet"] or {}).get("success_count"), (cells[a]["gauntlet"] or {}).get("task_count")] for a in arms}
    ann["turns_tool_emitted"] = {a: [p["tool_emitted"] for p in (cells[a]["turns"] or [])] for a in arms}
    ann["peerbench"] = {a: ({"table_txt": cells[a]["peerbench"].get("table_txt"), "table_md": cells[a]["peerbench"].get("table_md"),
                             "notes": (cells[a]["peerbench"].get("meta") or {}).get("notes"),
                             "complete": (cells[a]["peerbench"].get("meta") or {}).get("complete"),
                             "sampling": (cells[a]["peerbench"].get("meta") or {}).get("sampling"),
                             "status": {f"{cc['size']}/C{cc['conc']}": cc.get("status") for cc in cells[a]["peerbench"].get("cells", [])}}
                            if cells[a]["peerbench"] else None) for a in arms}
    ann["restore_watchdog"] = jl(R / "restore-watchdog.json")
    slim = {a: {"facts": cells[a]["facts"], "mixed": cells[a]["mixed"],
                "perclass_t1": {f"{c}/{m}": cells[a]["pc"]["cells"][(c, m)] for c in CLASSES for m in MODES},
                "wd": {k2: (cells[a]["wd"] or {}).get(k2) for k2 in ("timeouts", "trips", "alive_at_collect")},
                "errcount_server": cells[a]["err_server"], "errcount_sidecar": cells[a]["err_sidecar"], "foreign": cells[a]["foreign"],
                "metrics_pct1": cells[a]["m_pct1"], "metrics_mixed": cells[a]["m_mixed"], "metrics_peerbench": cells[a]["m_peer"]} for a in arms}
    return {"card": "BR2 DSpark k ladder (block verification)", "arms": list(arms), "table": {a: list(table[a]) for a in arms},
            "result": res, "suffix": suffix, "gates": gates, "gate_status": status, "invalid_arms": inv_ctrl + inv_arm,
            "arm_valid": {a: a not in bad_all for a in arms}, "valid": not inv_ctrl, "annotations": ann, "cells": slim, "R": str(R)}


def f2(x, nd=1):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x))


def table(v):
    arms = v["arms"]; an = v["annotations"]
    L = [f"Card BR2 verdict: {v['result']}   suffix {v['suffix']}   arms {' '.join(arms)}   arm_valid {v['arm_valid']}", "",
         "| gate | status | detail |", "|---|---|---|"]
    for k, (r, t) in v["gates"].items():
        L.append(f"| {k} | {v['gate_status'][k]} | {t} |")
    L += ["", "ANNOTATION (not gated) — boot / autotune shape:",
          "| arm | kspec | ksched | KV tokens | kv flag | boot s | autotune loaded/saved/new | autotune span s | retune | cg_tokens (>128) | cg_sched |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in arms:
        b = an["boot"][a]; at = b.get("autotune") or {}
        L.append(f"| {a} | {b.get('kspec')} | {b.get('ksched')} | {b.get('kv_tokens')} | {b.get('kv_flag') or '-'} | {b.get('boot_s')} | "
                 f"{at.get('loaded_n')}/{at.get('saved_n')}/{at.get('saved_new_n')} | {f2(at.get('elapsed_s'), 0)} | {at.get('retune')} | "
                 f"{b.get('cg_tokens')} ({b.get('cg_over_128')}) | {b.get('cg_tokens_sched')} |")
    L += ["", "ANNOTATION — temp 1.0 decode tok/s per class (ratio vs mean(A, A2)); mixed = ar_mixed_t C1 agg tok/s:",
          "| arm | " + " | ".join(f"{c} C1" for c in CLASSES) + " | " + " | ".join(f"{c} C8" for c in CLASSES) + " | mixed C1 |",
          "|---|" + "---|" * (2 * len(CLASSES) + 1)]
    for a in arms:
        s = an["speed_t1_vs_ctrl"][a]
        cellf = lambda key: f"{f2(s[key]['tok_s'])} ({f2(s[key]['ratio'], 3)})"
        L.append(f"| {a} | " + " | ".join(cellf(f"{c}/c1") for c in CLASSES) + " | " + " | ".join(cellf(f"{c}/c8") for c in CLASSES)
                 + f" | {cellf('mixed/c1')} |")
    L += ["", "ANNOTATION — tokens per verify step at temp 1.0 (1 + accepted/drafts, per class from the decoder's /metrics deltas; "
              "eff k = draft_tokens/drafts):",
          "| arm | " + " | ".join(f"{c} C1" for c in CLASSES) + " | " + " | ".join(f"{c} C8" for c in CLASSES) + " | pct1 window | mixed C1 | peerbench |",
          "|---|" + "---|" * (2 * len(CLASSES) + 3)]
    for a in arms:
        t = an["tokens_per_verify_step_t1"][a]; ek = an["eff_k"][a]; w = an["tokens_per_verify_step_windows"][a]
        L.append(f"| {a} | " + " | ".join(f"{f2(t[f'{c}/{m}'], 2)} (k {f2(ek[f'{c}/{m}'], 1)})" for m in MODES for c in CLASSES)
                 + f" | {f2(w['pct1'], 2)} | {f2(w['mixed_c1'], 2)} | {f2(w['peerbench'], 2)} |")
    L += ["", f"ANNOTATION — per-position acceptance at temp 1.0, C1 (curve[p] = accepted-at-p / drafts; marginal = curve[p]/curve[p-1]; "
              f"k* candidate = first p with marginal < {MARGINAL_BAR}):"]
    for a in arms:
        L.append(f"  arm {a} (kspec {v['table'][a][1]}):")
        for c in CLASSES:
            cv = an["per_position"][a][f"{c}/c1"]
            if not cv:
                L.append(f"    {c:10s} UNMEASURED"); continue
            L.append(f"    {c:10s} curve  " + " ".join(f"p{p + 1}={x:.3f}" for p, x in enumerate(cv["curve"])))
            L.append(f"    {'':10s} marg   " + " ".join(f"p{p + 1}={f2(x, 3)}" for p, x in enumerate(cv["marginal"]))
                     + f"   k* candidate {cv['kstar']}   (1+sum curve {cv['tokens_per_step_from_curve']:.2f})")
        mc = an["per_position_mixed_c1"][a]
        if mc:
            L.append(f"    {'mixed C1':10s} curve  " + " ".join(f"p{p + 1}={x:.3f}" for p, x in enumerate(mc["curve"])) + f"   k* {mc['kstar']}")
    L += ["", "ANNOTATION — per-position at C8 (KSCHED caps k at 3 for batch 5-8, so positions > 3 read 0 by construction):"]
    for a in arms:
        L.append(f"  {a}: " + "; ".join(f"{c} " + "/".join(f"{x:.2f}" for x in (an['per_position'][a][f'{c}/c8'] or {}).get('curve', [])[:4])
                                        for c in CLASSES))
    L.append("ANNOTATION — k* candidate (C1) per class: " + "; ".join(f"{a}: " + ", ".join(f"{c}={an['kstar_candidate_c1'][a][c]}" for c in CLASSES) for a in arms))
    L.append("ANNOTATION — gauntlet (temp 0): " + ", ".join(f"{a} {x[0]}/{x[1]}" for a, x in an["gauntlet"].items())
             + " | replay_turns tool_emitted per pass: " + ", ".join(f"{a} {x}" for a, x in an["turns_tool_emitted"].items()))
    L += ["", "ANNOTATION — peerbench (port of the BRRR peer's bench.py, their sampling: temperature 1.0, top_p 1.0, thinking off; "
              "same column layout as their README table):"]
    for a in arms:
        pb = an["peerbench"][a]
        if not pb:
            L.append(f"  arm {a}: peerbench ABSENT"); continue
        L.append(f"  arm {a}: complete={pb.get('complete')} status={pb.get('status')}" + (f" notes={pb.get('notes')}" if pb.get("notes") else ""))
        L += ["    " + ln for ln in (pb.get("table_txt") or "").splitlines()]
    L.append(f"ANNOTATION — production watchdog after restore: {an['restore_watchdog']}")
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
        json.dump(v, f, indent=1, default=str)
        f.write("\n")
    os.replace(tmp, out)
    print(table(v))
    return 0 if v["result"] == "PASS" else (3 if v["result"].startswith("INVALID-CONTROL") else 1)


if __name__ == "__main__":
    sys.exit(main())
