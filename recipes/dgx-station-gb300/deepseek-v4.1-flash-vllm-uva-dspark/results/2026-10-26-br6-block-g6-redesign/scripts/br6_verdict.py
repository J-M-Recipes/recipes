#!/usr/bin/env python3
"""br6_verdict.py — Card BR6 verdict: DSpark BLOCK verification bracketed against the GB300-only reference v28-prob
(= v26seed "Seeded Adaptive" + "draft_sample_method":"probabilistic"), A -> B -> A2, Grace UVA offload, no RTX sidecar. Pure stdlib.
Clone of br5_verdict.py @acf14b79 (DIFF-br6-verdict-vs-br5.txt shows every change).

Usage:  br6_verdict.py <R> [--out verdict.json]   -> <R>/verdict.json (or --out) + human table on stdout.
        rc 0 PASS, 1 FAIL, 3 INVALID (control or B polluted / control missing), 2 no input.

Shape (KV3 / recipe-factory adapter): {"result", "suffix", "gates": {name: [PASS|FAIL, text]}, ...}. A gate whose status is
INVALID (a control cell missing or polluted) is emitted as ["FAIL", "INVALID-CONTROL(<arms>): ..."] in "gates" so the adapter
shape stays PASS|FAIL; the true tri-state lives in "gate_status" {name: PASS|FAIL|INVALID} and "invalid_arms".
UNMEASURED semantics (KV3): a missing cell on B = FAIL; a missing cell on a control = that gate INVALID.

Differences vs BR5 (G0..G5, G7 and their bars UNCHANGED):
  * tags BR5-* -> BR6-* (perclass, mixed, tf, tf-compare, turns).
  * G6 REDESIGN (pre-registered, BRIEF-BR6.md 'G6 redesign'): replay_turns.py runs TURNS_REQ = 4 passes per arm (one file per
    pass, <X>/turns-BR6-<X>-p<k>.json; a legacy single file <X>/turns-BR6-<X>.json with N passes is also read). Statistic = per-arm
    MEAN tool-turns emitted per pass (of 12). PASS iff B has exactly 4 passes (first 4 used if more), every B pass has 0 errors,
    and mean(B) lies within [min(pooled 8 control passes) - 1, max(pooled 8 control passes) + 1]. Control validity: each control
    needs 4 error-free passes AND |mean(A) - mean(A2)| <= 1.5, else G6 INVALID-CONTROL(A,A2). Fewer than 4 passes = "INSUFFICIENT
    PASSES" (B -> FAIL, control -> INVALID); an exploratory reading on the passes present is printed, never gated.
    Reported: B pass min/max, pooled control distribution, per-arm means, and a per-transcript agreement table (which of the 12
    tool transcripts emitted a tool call in each pass, per arm; consistent B flips flagged) — ANNOTATION, not a gate.
  * the BR4 parity re-check block is replaced by a 'BR4/BR5 G6 history' annotation (earlier per-pass counts next to BR6's, and the
    new statistic applied to those 2-pass receipts as information) plus a one-line G2b shell_ops t0 C1 history.
  * suffix of an INVALID card also names the INVALID gate(s): BR6-INVALID-CONTROL-<arms>-<gates> (e.g. ...-A-A2-G6).
  * ar-lines cells delta ends at turns-p4-end (falls back to the last turns-p<k>-end, turns-end, stop).

Inputs (<R> = ~/dsv41/m3/br6/results-cardBR6, written by cardBR6_runner.sh):
  <X>/facts.json, args.json, env.json, binds.json, go.json, boot-autotune.log, boot-block.log, boot-prob.log, ar-lines.tsv  (X in A B A2)
  args-ref-<REF>.json, env-ref-<REF>.json, binds-ref-<REF>.json, kv-ref.txt                     control ground truth (v28-prob)
  <X>/perclass-t1/perclass-table-BR6-<X>-t1.json (+ raw perclass-BR6-<X>-t1-r{1,2}-{c1,c8}.json)   temp 1.0 (G1, G2)
  <X>/perclass-t0/perclass-table-BR6-<X>-t0.json                                         stock temp-0 window (G1, G2b)
  <X>/mixed-BR6-<X>-c1-r<k>.json      ar_mixed_t.py C1 120 s at temp 1.0 (G3)
  tf-compare-BR6-A-vs-BR6-B.json, tf-compare-BR6-A2-vs-BR6-B.json, tf-compare-BR6-A-vs-BR6-A2.json; <X>/tf-BR6-<X>.jsonl (G4)
  flipsm-B-vs-A.txt, flipsm-B-vs-A2.txt, flipsm-A2-vs-A.txt   flips_margin_p2.py output, last line GATE ... (G4 margin)
  gauntlet-<X>/summary.json (G5) ; <X>/turns-BR6-<X>-p{1..4}.json (G6) ; errcount-<X>.txt (G7)
  <X>/foreign-*.txt (G0b, G7) ; <X>/metrics/<cell>-{start,end}.txt (annotation) ; mem.tsv ; restore-watchdog.json (annotation)
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

ARMS = ("A", "B", "A2")
CTRL = ("A", "A2")
CLASSES = ("prose", "structured", "code", "shell_ops", "tool_json")
MODES = ("c1", "c8")
REF = "dsv41-vllm-1017-v28-prob-BOUND-REF"
IMAGE_ID = "sha256:00d577a6a63281e15336029d5bcee4e9a2cf182214a4f20ba6111b1c8e79893d"
HOOK_KEYS_N = 40        # PIN_HOT rehome layer= lines (one per MoE layer)
SPEC_A = ',"draft_sample_method":"probabilistic"'      # carried by the reference and every arm
SPEC_BLOCK = ',"rejection_sample_method":"block"'      # the one axis (B)
WANT = {"A": (1, 0), "B": (1, 1), "A2": (1, 0)}          # (prob_in_args, block_in_args)
# G6 redesign (pre-registered, BRIEF-BR6.md): 4 passes per arm, mean statistic, pooled-control bar +-1, control means within 1.5
TURNS_REQ = 8
TURNS_N = 12             # tool transcripts in transcript_fixture.json (sha1 e3f1aea4)
G6_MARGIN = 1.0
G6_CTRL_MEAN_TOL = 1.5
# G6 history (annotation only): per-pass tool-turn counts /12 from the finished receipts (replay_turns.py PASSES=2, temp 0)
#   BR4 (results-cardBR4, controls A/A2 = v26seed WITHOUT probabilistic drafting; P = v26seed+prob; PB = P + block)
#   BR5 (results-cardBR5, controls A/A2 = v28-prob = BR4 P; B = v28-prob + block = BR4 PB)
G6_HISTORY = {"BR4": {"ctrl": {"A": [6, 7], "A2": [8, 8]}, "cand": {"P": [6, 8], "PB": [8, 9]}, "old_gate": {"P": "PASS", "PB": "FAIL"}},
              "BR5": {"ctrl": {"A": [9, 8], "A2": [8, 8]}, "cand": {"B": [6, 7]}, "old_gate": {"B": "FAIL"}}}
G2B_SHELL_HISTORY = {"BR4 PB": "+3.6% vs bar 2.6% (FAIL)", "BR5 B": "+2.4% vs bar 10.3% (PASS)"}
CTRL_TOL = 0.06          # G1: |A - A2| / mean(A, A2) per class at C1 (both windows); C8 reported only (bimodal class)
FLOOR = 0.02             # G2/G2b/G3 bar = max(2 %, 1.5 x |A - A2| spread)
SPREAD_K = 1.5
TF_ABS = 0.03            # G4 raw top-1 flips <= 3 %
TF_FLOOR_K = 1.5         # ... and <= 1.5 x the A-vs-A2 floor
GAUNTLET_N = 12
SHORT = {"G0_binding": "G0", "G0b_foreign": "G0b", "G1_controls": "G1", "G2_perclass_t1": "G2", "G2b_t0_parity": "G2b",
         "G3_mixed_c1_t1": "G3", "G4_tf": "G4", "G5_gauntlet": "G5", "G6_turns": "G6", "G7_errors": "G7"}


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


def pct(x, nd=2):
    return "n/a" if x is None else f"{x * 100:+.{nd}f}%"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def spec_of(args):
    if not isinstance(args, list) or "--speculative-config" not in args:
        return None
    i = args.index("--speculative-config")
    return args[i + 1] if i + 1 < len(args) else None


def metrics_file(p):
    """sum vllm:spec_decode_* counters by name (all label sets)"""
    out = {}
    try:
        for ln in Path(p).read_text().splitlines():
            if not ln.startswith("vllm:spec_decode_"):
                continue
            name = ln.split("{", 1)[0].split(" ", 1)[0]
            try:
                out[name] = out.get(name, 0.0) + float(ln.split()[-1])
            except ValueError:
                pass
    except Exception:  # noqa: BLE001
        return None
    return out


POS = re.compile(r'^vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^}]*position="(\d+)"[^}]*\}\s+(\S+)')


def metrics_pos(p):
    """per-position accepted counters {pos: value} (summed over other labels) — from br4_verdict.py"""
    out = {}
    try:
        for ln in Path(p).read_text().splitlines():
            m = POS.match(ln)
            if m:
                try:
                    out[int(m.group(1))] = out.get(int(m.group(1)), 0.0) + float(m.group(2))
                except ValueError:
                    pass
    except Exception:  # noqa: BLE001
        return None
    return out


def mdelta(A, cell):
    s, e = metrics_file(A / "metrics" / f"{cell}-start.txt"), metrics_file(A / "metrics" / f"{cell}-end.txt")
    if s is None or e is None:
        return None
    g = lambda d, k: d.get(f"vllm:spec_decode_{k}_total", d.get(f"vllm:spec_decode_{k}", 0.0))
    dr, dt_, ac = g(e, "num_drafts") - g(s, "num_drafts"), g(e, "num_draft_tokens") - g(s, "num_draft_tokens"), \
        g(e, "num_accepted_tokens") - g(s, "num_accepted_tokens")
    ps, pe = metrics_pos(A / "metrics" / f"{cell}-start.txt") or {}, metrics_pos(A / "metrics" / f"{cell}-end.txt") or {}
    per_pos = [round((pe.get(k, 0.0) - ps.get(k, 0.0)) / dr, 4) if dr else None for k in sorted(pe)] if pe else None
    return {"drafts": dr, "draft_tokens": dt_, "accepted": ac,
            "acc_per_draft": (ac / dr) if dr else None, "tokens_per_verify_step": (1 + ac / dr) if dr else None,
            "accept_rate": (ac / dt_) if dt_ else None, "accept_per_pos": per_pos}


TS = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})")


def autotune(p):
    d = {"loaded": None, "saved": None, "loaded_dir": None, "saved_dir": None, "elapsed_s": None, "pairs": 0}
    try:
        L = Path(p).read_text().splitlines()
    except Exception:  # noqa: BLE001
        return d
    ts_s, ts_e = [], []
    for ln in L:
        m = re.search(r"Loaded (\d+) configs from (\S+)", ln)
        if m:
            d["loaded"], d["loaded_dir"] = int(m.group(1)), m.group(2)
        m = re.search(r"Saved (\d+) configs to (\S+)", ln)
        if m:
            d["saved"], d["saved_dir"] = int(m.group(1)), m.group(2)
        t = TS.search(ln)
        if t and "Autotuning process starts" in ln:
            ts_s.append(dt.datetime.strptime(t.group(1), "%Y-%m-%d %H:%M:%S"))
        if t and "Autotuning process ends" in ln:
            ts_e.append(dt.datetime.strptime(t.group(1), "%Y-%m-%d %H:%M:%S"))
    d["pairs"] = min(len(ts_s), len(ts_e))
    if ts_s and ts_e:
        d["elapsed_s"] = (max(ts_e) - min(ts_s)).total_seconds()
    d["hash_hit_shape"] = (d["loaded"] is not None and d["loaded"] == d["saved"] and d["loaded_dir"] == d["saved_dir"])
    return d


def ar_lines(p):
    """<epoch> <tag> swaps=N telemetry=M steps_bytes=K  ->  {tag: {swaps, telemetry, steps_bytes}} (last wins) + cells delta"""
    out = {}
    try:
        for ln in Path(p).read_text().splitlines():
            parts = ln.split("\t")
            if len(parts) < 3:
                continue
            d = {}
            for kv in parts[2:]:
                k, _, v = kv.partition("=")
                try:
                    d[k] = int(v)
                except ValueError:
                    d[k] = v
            out[parts[1]] = d
    except Exception:  # noqa: BLE001
        return None
    tend = sorted((k for k in out if re.fullmatch(r"turns-p\d+-end", k)), key=lambda k: int(k.split("-p")[1].split("-")[0]))
    s, e = out.get("pct1-start"), out.get("turns-p%d-end" % TURNS_REQ) or (out.get(tend[-1]) if tend else None) or out.get("turns-end") or out.get("stop")
    if s and e:
        out["_cells_delta"] = {k: e.get(k, 0) - s.get(k, 0) for k in ("swaps", "telemetry", "steps_bytes")
                               if isinstance(e.get(k), int) and isinstance(s.get(k), int)}
    return out


def norm_binds(L):
    return sorted(("<AR>:/ar" if b.endswith(":/ar") else b) for b in (L or []))


def load_perclass(A, arm, win):
    tag = f"BR6-{arm}-{win}"
    t = jl(A / f"perclass-{win}" / f"perclass-table-{tag}.json")
    out = {"table": t is not None, "cells": {}, "x3": {}, "hygiene": (t or {}).get("hygiene"), "acc": {}}
    S = (t or {}).get("summary") or {}
    for c in CLASSES:
        for m in MODES:
            out["cells"][(c, m)] = ((S.get(c) or {}).get(m) or {}).get("mean")
        out["x3"][c] = ((S.get(c) or {}).get("x3") or {}).get("accept")
        for m in MODES:   # pooled acc/step from the raw files (/metrics deltas recorded by perclass_decode*.py)
            acc = steps = 0.0; ok = False
            for r in ("r1", "r2"):
                raw = jl(A / f"perclass-{win}" / f"perclass-{tag}-{r}-{m}.json")
                x = ((raw or {}).get("res") or {}).get(c) or {}
                if x.get("draft_steps"):
                    acc += x.get("accepted") or 0; steps += x["draft_steps"]; ok = True
            out["acc"][(c, m)] = (acc / steps) if (ok and steps) else None
    return out


def load_arm(R, arm):
    A = R / arm
    c = {"present": A.is_dir(), "facts": jl(A / "facts.json"), "args": jl(A / "args.json"), "env": jl(A / "env.json"),
         "binds": jl(A / "binds.json"), "go": jl(A / "go.json"), "autotune": autotune(A / "boot-autotune.log"),
         "ar_lines": ar_lines(A / "ar-lines.tsv")}
    try:
        c["block_boot_sample"] = Path(A / "boot-block.log").read_text().splitlines()[:3]
    except Exception:  # noqa: BLE001
        c["block_boot_sample"] = []
    try:
        c["prob_boot_sample"] = Path(A / "boot-prob.log").read_text().splitlines()[:3]
    except Exception:  # noqa: BLE001
        c["prob_boot_sample"] = []
    c["pc"] = {w: load_perclass(A, arm, w) for w in ("t1", "t0")}
    reps = []
    for f in sorted(glob.glob(str(A / f"mixed-BR6-{arm}-c1-r*.json")), key=lambda s: int(s.rsplit("-r", 1)[1].split(".")[0])):
        d = jl(f)
        if d and d.get("agg_tok_s_all") is not None:
            reps.append({"rep": int(f.rsplit("-r", 1)[1].split(".")[0]), "agg_tok_s_all": d["agg_tok_s_all"],
                         "errors": d.get("errors", 0), "n": d.get("n"), "temperature": (d.get("sampling") or {}).get("temperature")})
    c["mixed"] = reps
    c["mixed_mean"] = mean([r["agg_tok_s_all"] for r in reps])
    c["tf_capture"] = (A / f"tf-BR6-{arm}.jsonl").exists() and (A / f"tf-BR6-{arm}.jsonl").stat().st_size > 0
    c["gauntlet"] = jl(R / f"gauntlet-{arm}" / "summary.json")
    c["turns"], c["turns_src"] = load_turns(A, arm)
    c["err_server"] = rint(R / f"errcount-{arm}.txt")
    fr = {}
    for f in glob.glob(str(A / "foreign-*.txt")):
        fr[Path(f).stem.split("-", 1)[1]] = rint(f)
    c["foreign"] = fr
    c["m_pct1"] = mdelta(A, "pct1")
    mx = [mdelta(A, f"c1-r{k}") for k in (1, 2, 3)]
    mx = [m for m in mx if m]
    if mx:
        ac, dr = sum(m["accepted"] for m in mx), sum(m["drafts"] for m in mx)
        c["m_mixed"] = {"accepted": ac, "drafts": dr, "tokens_per_verify_step": (1 + ac / dr) if dr else None}
    else:
        c["m_mixed"] = None
    return c


def turn_pass(p, src):
    tool = [r for r in p if r.get("has_tool")]
    return {"tool_emitted": sum(1 for r in tool if r.get("emitted")), "tool_turns": len(tool),
            "text_emitted": sum(1 for r in p if not r.get("has_tool") and r.get("emitted")),
            "errors": sum(1 for r in p if r.get("error")),
            "tool_idx": [r.get("i", k) for k, r in enumerate(p) if r.get("has_tool")],
            "emitted_idx": [r.get("i", k) for k, r in enumerate(p) if r.get("has_tool") and r.get("emitted")], "src": src}


def load_turns(A, arm):
    """G6 redesign: one replay_turns.py file per pass (<X>/turns-BR6-<X>-p<k>.json, PASSES=1 each, k ascending); a legacy single
    file <X>/turns-BR6-<X>.json holding N passes is read when no per-pass file exists (BR5-receipt replay). -> (passes|None, src)"""
    files = sorted(glob.glob(str(A / f"turns-BR6-{arm}-p*.json")), key=lambda s: s)
    files = [f for f in files if re.search(r"-p(\d+)\.json$", f)]
    files.sort(key=lambda f: int(re.search(r"-p(\d+)\.json$", f).group(1)))
    out = []
    for f in files:
        t = jl(f)
        for p in ((t or {}).get("passes") or []):
            out.append(turn_pass(p, Path(f).name))
    if files:
        return (out or None), "per-pass files: " + ",".join(Path(f).name for f in files)
    t = jl(A / f"turns-BR6-{arm}.json")
    if t and t.get("passes"):
        return [turn_pass(p, f"turns-BR6-{arm}.json#{k}") for k, p in enumerate(t["passes"])], f"legacy single file turns-BR6-{arm}.json"
    return None, "missing"


def load_mem(R):
    out = {}
    p = R / "mem.tsv"
    if p.exists():
        for ln in p.read_text().splitlines():
            parts = ln.split()
            if len(parts) == 3:
                try:
                    out.setdefault(parts[1], []).append(float(parts[2]))
                except ValueError:
                    pass
    return {k: min(v) for k, v in out.items()}


class G:
    """gate accumulator: status PASS/FAIL/INVALID + arms that make it INVALID"""

    def __init__(self):
        self.fail, self.inv, self.txt = False, set(), []

    def status(self):
        return "INVALID" if self.inv else ("FAIL" if self.fail else "PASS")


def verdict(R):
    cells = {a: load_arm(R, a) for a in ARMS}
    prod_args = jl(R / f"args-ref-{REF}.json")      # "prod" = the v28-prob control ground truth in this card
    prod_env = jl(R / f"env-ref-{REF}.json")
    prod_binds = jl(R / f"binds-ref-{REF}.json")
    try:
        kv_ref = (R / "kv-ref.txt").read_text().strip() or None
    except Exception:  # noqa: BLE001
        kv_ref = None
    gs = {}
    calc = {}

    # ---- G0 binding + wiring: every arm bound; KV: B and A2 equal A; prob_in_args 1/1/1, block_in_args 0/1/0; A/A2 Args ==
    # v28-prob; B Args == v28-prob except the spec (== v28-prob + the block key); Env == v28-prob (order-insensitive); Binds ==
    # v28-prob except the /ar source; the ground-truth spec itself carries probabilistic drafting and no block;
    # image 0909; hook wiring/guard lines; GO + unfreeze acknowledged (production regime). A control with broken wiring /
    # regime / odd KV is INVALID (its numbers are not the control); B -> FAIL.
    g = G()
    kvs = {a: (cells[a]["facts"] or {}).get("kv_tokens") or None for a in ARMS}
    ref_ok = False
    try:
        rs = json.loads(spec_of(prod_args))
        ref_ok = rs.get("draft_sample_method") == "probabilistic" and "rejection_sample_method" not in rs and spec_of(prod_args).endswith(SPEC_A + "}")
    except Exception:  # noqa: BLE001
        pass
    if not ref_ok:
        g.fail = True
    g.txt.append(f"ground truth {REF} spec carries probabilistic drafting and no block: {ref_ok}")
    for a in ARMS:
        f = cells[a]["facts"]
        if not f or not f.get("bound_at"):
            (g.inv.add(a) if a in CTRL else setattr(g, "fail", True)); g.txt.append(f"{a}: NOT BOUND"); continue
        wantp, want = WANT[a]
        bad_ctl = lambda: (g.inv.add(a) if a in CTRL else setattr(g, "fail", True))
        bits = [f"kv={kvs[a] or 'MISSING'}", f"prob_in_args={f.get('prob_in_args')}(want {wantp})", f"block_in_args={f.get('block_in_args')}(want {want})"]
        if f.get("prob_in_args") != wantp:
            g.fail = True; bits.append("PROB!")
        if f.get("block_in_args") != want:
            g.fail = True; bits.append("BLOCK!")
        args = cells[a]["args"]
        if prod_args is None or args is None:
            g.fail = True; bits.append("args UNMEASURED")
        elif a in CTRL:
            ok = args == prod_args
            bits.append(f"args==v28prob {ok}")
            g.fail |= not ok
        else:
            ps, bs = spec_of(prod_args), spec_of(args)
            rest_ok = len(args) == len(prod_args) and all(x == y for x, y in zip(args, prod_args) if x != bs or y != ps)
            spec_ok = ps is not None and bs == ps[:-1] + SPEC_BLOCK + "}"
            try:
                spec_ok = spec_ok and json.loads(bs) == dict(json.loads(ps), rejection_sample_method="block")
            except Exception:  # noqa: BLE001
                spec_ok = False
            bits.append(f"args==v28prob-except-spec {rest_ok} spec==v28prob+block {spec_ok}")
            g.fail |= not (rest_ok and spec_ok)
        env, binds = cells[a]["env"], cells[a]["binds"]
        eok = (env is not None and prod_env is not None and sorted(env) == sorted(prod_env))   # order-insensitive from the start
        bok = (binds is not None and prod_binds is not None and norm_binds(binds) == norm_binds(prod_binds))
        iok = f.get("image") == IMAGE_ID
        bits += [f"env==v28prob(sorted) {eok}", f"binds==v28prob(except /ar) {bok}", f"image0909 {iok}"]
        g.fail |= not (eok and bok and iok)
        hk = (f.get("rehome_done") == 1 and f.get("rehome_layers") == HOOK_KEYS_N and f.get("pin_abort") == 0
              and f.get("hook_failed") == 0 and not f.get("hook_keys_missing"))
        bits.append(f"hook rehome {f.get('rehome_layers')}/{HOOK_KEYS_N} done={f.get('rehome_done')} abort={f.get('pin_abort')} "
                    f"failed={f.get('hook_failed')} missing=[{f.get('hook_keys_missing') or ''}] host_avail@L9={f.get('host_avail_l9_gib')}GiB")
        if not hk:
            bad_ctl(); bits.append("HOOK!")
        go = cells[a]["go"] or {}
        gok = (go.get("go_seen_log") or 0) >= 1 and (go.get("unfreeze_log") or 0) >= 1
        bits.append(f"GO+unfreeze {go.get('go_seen_log')}/{go.get('unfreeze_log')} ctrl={json.dumps(go.get('ctrl'))}")
        if not gok:
            bad_ctl(); bits.append("REGIME!")
        g.txt.append(f"{a}: " + " ".join(bits))
    # KV rule: B and A2 equal A. Majority names the odd arm: odd B -> FAIL, odd control -> INVALID(that control).
    present = {a: v for a, v in kvs.items() if v and cells[a]["facts"] and cells[a]["facts"].get("bound_at")}
    kvtxt = f"KV A={kvs['A']} B={kvs['B']} A2={kvs['A2']} (v28-prob boot log {kv_ref}, info; rule: B and A2 equal A — block allocates nothing)"
    if "A" in present and all(present.get(a) == present["A"] for a in ("B", "A2") if a in present):
        for a in ("B", "A2"):
            if (cells[a]["facts"] or {}).get("bound_at") and a not in present:
                (g.inv.add(a) if a in CTRL else setattr(g, "fail", True)); kvtxt += f" KV MISSING on bound {a}"
    else:
        vals = [present.get(a) for a in ARMS]
        maj = next((v for v in vals if v is not None and vals.count(v) >= 2), None)
        odd = [a for a in ARMS if present.get(a) != maj] if maj else list(ARMS)
        for a in odd:
            if a in CTRL:
                g.inv.add(a)
            else:
                g.fail = True
        kvtxt += f" KV! odd={odd}"
    g.txt.append(kvtxt)
    at = {a: cells[a]["autotune"] for a in ARMS}
    bs = {a: (cells[a]["facts"] or {}).get("boot_s") for a in ARMS}
    cb = mean([bs[a] for a in CTRL])
    hh = " | hash-hit shape (info): " + "; ".join(
        f"{a} loaded={at[a]['loaded']} saved={at[a]['saved']} same_dir={at[a]['loaded_dir'] == at[a]['saved_dir'] and at[a]['loaded_dir'] is not None} "
        f"autotune_span={at[a]['elapsed_s']}s boot={bs[a]}s" for a in ARMS)
    dirs = {at[a]["loaded_dir"] for a in ARMS if at[a]["loaded_dir"]}
    hh += f"; one autotune key across arms: {len(dirs) == 1}; 0909 hash-hit shape (Loaded N == Saved N, same dir, span <= 60 s): " + ", ".join(
        f"{a}={bool(at[a].get('hash_hit_shape')) and (at[a]['elapsed_s'] is not None and at[a]['elapsed_s'] <= 60)}" for a in ARMS)
    if bs["B"] and cb:
        hh += f"; B boot / controls = {bs['B'] / cb:.2f} (<= 1.5 expected)"
    gs["G0_binding"] = (g, "; ".join(g.txt) + hh)

    # ---- G0b foreign POSTs (non-loopback) per arm
    g = G(); b_foreign = False
    for a in ARMS:
        fr = cells[a]["foreign"]
        n = sum(v or 0 for v in fr.values()) if fr else None
        g.txt.append(f"{a}={'UNMEASURED' if n is None else n} {fr or ''}")
        if n is None:
            if a in CTRL and cells[a]["facts"]:
                g.inv.add(a)
            elif a == "B" and cells[a]["facts"]:
                g.fail = True
        elif n > 0:
            if a in CTRL:
                g.inv.add(a)
            else:
                b_foreign = True; g.fail = True
    gs["G0b_foreign"] = (g, "non-127.0.0.1 POST /v1 per arm: " + "; ".join(g.txt) + (" -> B INVALID" if b_foreign else ""))

    # ---- G1 control agreement: per class C1, both windows, |A-A2|/mean <= 6 %; C8 reported only
    g = G(); c8 = []
    for w in ("t1", "t0"):
        for c in CLASSES:
            va, va2, vb = (cells[x]["pc"][w]["cells"][(c, "c1")] for x in ("A", "A2", "B"))
            if va is None or va2 is None:
                for x, v in (("A", va), ("A2", va2)):
                    if v is None:
                        g.inv.add(x)
                g.txt.append(f"{w}/{c}/c1 UNMEASURED"); continue
            d = abs(va - va2) / ((va + va2) / 2)
            if d > CTRL_TOL:
                # B is a neutral referee only in the temp-0 window (greedy rows: B == A by construction). At temp 1.0 B is
                # expected to differ, so a t1 disagreement cannot name the odd control -> both.
                odd = ("A" if abs(vb - va) > abs(vb - va2) else "A2") if (vb is not None and w == "t0") else None
                g.inv.update([odd] if odd else ["A", "A2"])
                g.txt.append(f"{w}/{c}/c1 A={va:.1f} A2={va2:.1f} d={d * 100:.1f}% > 6% (odd={odd or 'A,A2'})")
            else:
                g.txt.append(f"{w}/{c}/c1 d={d * 100:.1f}%")
            ca, ca2 = cells["A"]["pc"][w]["cells"][(c, "c8")], cells["A2"]["pc"][w]["cells"][(c, "c8")]
            if ca and ca2:
                c8.append(f"{w}/{c}/c8 d={abs(ca - ca2) / ((ca + ca2) / 2) * 100:.1f}%")
    gs["G1_controls"] = (g, "C1 per class: " + "; ".join(g.txt) + " | C8 (info, bimodal class): " + "; ".join(c8))

    # ---- G2 PRIMARY per-class decode at temp 1.0, C1 and C8: B >= ctrl x (1 - max(2%, 1.5 x |A-A2| spread)) vs BOTH controls
    g = G(); rows = []
    for c in CLASSES:
        for m in MODES:
            va, va2, vb = (cells[x]["pc"]["t1"]["cells"][(c, m)] for x in ("A", "A2", "B"))
            if vb is None:
                g.fail = True; g.txt.append(f"{c}/{m} B UNMEASURED"); continue
            if va is None or va2 is None:
                g.inv.update([x for x, v in (("A", va), ("A2", va2)) if v is None]); g.txt.append(f"{c}/{m} control UNMEASURED"); continue
            cm = (va + va2) / 2; sp = abs(va - va2) / cm; bar = max(FLOOR, SPREAD_K * sp)
            ok = vb >= va * (1 - bar) and vb >= va2 * (1 - bar)
            g.fail |= not ok
            rows.append({"class": c, "mode": m, "A": va, "A2": va2, "B": vb, "ratio": vb / cm, "spread": sp, "bar": bar, "pass": ok})
            g.txt.append(f"{c}/{m} B/ctrl={vb / cm:.3f} (A {va:.1f} A2 {va2:.1f} B {vb:.1f}; bar -{bar * 100:.1f}%){'' if ok else ' FAIL'}")
    calc["G2_rows"] = rows
    gs["G2_perclass_t1"] = (g, "temp 1.0 forced-length decode tok/s: " + "; ".join(g.txt))

    # ---- G2b temp-0 no-op sanity: B within control noise at C1 per class (two-sided); C8 info
    g = G(); info = []; g2b_rows = {}
    for c in CLASSES:
        for m in MODES:
            va, va2, vb = (cells[x]["pc"]["t0"]["cells"][(c, m)] for x in ("A", "A2", "B"))
            if m == "c8":
                if None not in (va, va2, vb):
                    info.append(f"{c}/c8 B/ctrl={vb / ((va + va2) / 2):.3f}")
                continue
            if vb is None:
                g.fail = True; g.txt.append(f"{c} B UNMEASURED"); continue
            if va is None or va2 is None:
                g.inv.update([x for x, v in (("A", va), ("A2", va2)) if v is None]); g.txt.append(f"{c} control UNMEASURED"); continue
            cm = (va + va2) / 2
            # RESCORE: the no-op bar also honours the controls' own r1/r2 spread for this class (within-arm noise).
            import json as _j, glob as _g
            wspread = 0.0
            for _x in ("A", "A2"):
                for _f in _g.glob(f"{R}/{_x}/perclass-t0/perclass-table-*-t0.json"):
                    try:
                        wspread = max(wspread, float(_j.load(open(_f))["summary"][c]["c1"]["spread"]))
                    except Exception:
                        pass
            bar = max(FLOOR, SPREAD_K * abs(va - va2) / cm, SPREAD_K * wspread); r = vb / cm - 1
            ok = abs(r) <= bar
            g.fail |= not ok
            g2b_rows[c] = {"A": va, "A2": va2, "B": vb, "rel": r, "bar": bar, "ctrl_r1r2_spread": wspread, "pass": ok}
            g.txt.append(f"{c}/c1 B {pct(r, 1)} (|.| bar {bar * 100:.1f}%, ctrl r1/r2 spread {wspread * 100:.1f}%){'' if ok else ' OUTSIDE NOISE'}")
    calc["G2b_rows"] = g2b_rows
    gs["G2b_t0_parity"] = (g, "stock temp-0 window (PARITY: greedy rows never use block verification; B must equal A within noise): "
                         + "; ".join(g.txt) + " | C8 info: " + "; ".join(info))

    # ---- G3 mixed C1 at temp 1.0
    g = G()
    mb, ma, ma2 = cells["B"]["mixed_mean"], cells["A"]["mixed_mean"], cells["A2"]["mixed_mean"]
    nb = len(cells["B"]["mixed"])
    if mb is None or nb < 2:
        g.fail = True; g.txt.append(f"B UNMEASURED (reps={nb}, need >= 2)")
    elif ma is None or ma2 is None or len(cells["A"]["mixed"]) < 2 or len(cells["A2"]["mixed"]) < 2:
        g.inv.update([x for x in CTRL if cells[x]["mixed_mean"] is None or len(cells[x]["mixed"]) < 2])
        g.txt.append("control UNMEASURED (need >= 2 reps each)")
    else:
        cm = (ma + ma2) / 2; sp = abs(ma - ma2) / cm; bar = max(FLOOR, SPREAD_K * sp); r = mb / cm - 1
        g.fail = r < -bar
        calc["G3"] = {"B": mb, "A": ma, "A2": ma2, "ctrl_mean": cm, "spread": sp, "bar": bar, "rel": r}
        g.txt.append(f"B mean {mb:.2f} vs ctrl {cm:.2f} ({pct(r)}; A {ma:.2f} A2 {ma2:.2f}; bar -{bar * 100:.2f}%) "
                     f"reps B={[round(x['agg_tok_s_all'], 1) for x in cells['B']['mixed']]} errors B={sum(x['errors'] or 0 for x in cells['B']['mixed'])}")
    gs["G3_mixed_c1_t1"] = (g, "ar_mixed_t C1 agg_tok_s_all at temp 1.0: " + "; ".join(g.txt))

    # ---- G4 TF (teacher-forced; temperature-independent)
    g = G()
    if not cells["B"]["tf_capture"]:
        g.fail = True; g.txt.append("B capture MISSING")
    else:
        miss = [a for a in CTRL if not cells[a]["tf_capture"]]
        fl = jl(R / "tf-compare-BR6-A-vs-BR6-A2.json")
        cmp = {a: jl(R / f"tf-compare-BR6-{a}-vs-BR6-B.json") for a in CTRL}
        if miss:
            g.inv.update(miss); g.txt.append(f"control capture MISSING {miss}")
        elif fl is None or any(v is None for v in cmp.values()):
            g.fail = True; g.txt.append("compare file(s) MISSING with captures present (runner fault)")
        else:
            floor = fl.get("top1_flip_rate")
            for a in CTRL:
                rate = cmp[a].get("top1_flip_rate")
                ok = rate is not None and rate <= TF_ABS and rate <= TF_FLOOR_K * floor
                g.fail |= not ok
                g.txt.append(f"B-vs-{a} flips {rate * 100:.3f}% (bar <=3% and <= {TF_FLOOR_K} x floor {floor * 100:.3f}% = {TF_FLOOR_K * floor * 100:.3f}%; "
                             f"token-mismatch docs {cmp[a].get('docs_token_mismatch')}){'' if ok else ' FAIL'}")
            mg = []
            for a in CTRL:
                p = R / f"flipsm-B-vs-{a}.txt"
                last = (p.read_text().strip().splitlines() or [""])[-1] if p.exists() else ""
                m = re.search(r"hi_agent_tool=(\d+)", last)
                sk = re.search(r"skipped_docs=(\d+)", last)
                mg.append((a, int(m.group(1)) if m else None, sk.group(1) if sk else "?"))
            if all(x[1] is not None for x in mg):
                bad = [x for x in mg if x[1] != 0]
                g.fail |= bool(bad)
                g.txt.append("margin flips at ref p>=0.9 on agent/tool docs: " + ", ".join(f"vs {a}={n} (skipped_docs {s})" for a, n, s in mg)
                             + ("" if not bad else " FAIL"))
            else:
                g.txt.append("margin flips UNMEASURED (flips_margin_p2 output missing/unparsable; not failing)")
    gs["G4_tf"] = (g, "; ".join(g.txt))

    # ---- G5 gauntlet 12/12 on B (controls reported)
    g = G()
    gb = cells["B"]["gauntlet"]
    gc = {a: (cells[a]["gauntlet"] or {}) for a in CTRL}
    if not gb:
        g.fail = True; g.txt.append("B UNMEASURED")
    else:
        s, t = gb.get("success_count"), gb.get("task_count")
        g.fail = not (s == GAUNTLET_N and t == GAUNTLET_N); g.txt.append(f"B {s}/{t}")
    g.txt.append("controls " + ", ".join(f"{a} {gc[a].get('success_count')}/{gc[a].get('task_count')}" for a in CTRL) + " (temp 0: run_suite.py)")
    gs["G5_gauntlet"] = (g, "; ".join(g.txt))

    # ---- G6 REDESIGN (pre-registered): per-arm MEAN tool-turns over TURNS_REQ passes; B mean within the pooled control
    # [min - 1, max + 1]; controls' means within 1.5 of each other (else INVALID-CONTROL for G6); errors 0 on every B pass (hard)
    g = G()
    g6 = g6_eval(cells)
    calc["G6"] = g6
    g.fail, g.inv = g6["fail"], set(g6["inv"])
    g.txt = g6["txt"]
    gs["G6_turns"] = (g, "; ".join(g.txt))

    # ---- G7 server FF errors 0 + foreign POSTs 0 per arm (no sidecar, no card watchdog on this lane)
    g = G()
    for a in ARMS:
        e, fr = cells[a]["err_server"], cells[a]["foreign"]
        if not cells[a]["facts"] and a in CTRL:
            g.inv.add(a); g.txt.append(f"{a}: not bound"); continue
        n = sum(v or 0 for v in fr.values()) if fr else None
        if e is None or n is None:
            (g.inv.add(a) if a in CTRL else setattr(g, "fail", True)); g.txt.append(f"{a}: UNMEASURED (ff={e} foreign={n})"); continue
        bad = e != 0 or n != 0
        if bad:
            (g.inv.add(a) if a in CTRL else setattr(g, "fail", True))
        g.txt.append(f"{a}: ff={e} foreign={n}{' BAD' if bad else ''}")
    gs["G7_errors"] = (g, "; ".join(g.txt))

    # ---- result
    status = {k: v[0].status() for k, v in gs.items()}
    inv = sorted({a for v in gs.values() for a in v[0].inv}, key=ARMS.index)
    failed = [k for k, s in status.items() if s == "FAIL"]
    if inv:
        res = "INVALID-CONTROL-" + "-".join(inv)
        suffix = "BR6-" + res + "-" + "-".join(SHORT[k] for k, s in status.items() if s == "INVALID")
    elif b_foreign:
        res = "INVALID-B"; suffix = "BR6-INVALID-B"
    elif failed:
        res = "FAIL"; suffix = "BR6FAIL-" + "-".join(SHORT[k] for k in failed)
    else:
        res = "PASS"; suffix = "BR6PASS"
    gates = {}
    for k, (g, txt) in gs.items():
        s = status[k]
        gates[k] = ["PASS", txt] if s == "PASS" else ["FAIL", (f"INVALID-CONTROL({','.join(sorted(g.inv, key=ARMS.index))}): " if s == "INVALID" else "") + txt]

    # ---- annotations (NOT gates): acceptance is not comparable across verification methods
    ann = {}
    tps = {a: (cells[a]["m_pct1"] or {}).get("tokens_per_verify_step") for a in ARMS}
    tpm = {a: (cells[a]["m_mixed"] or {}).get("tokens_per_verify_step") for a in ARMS}
    ctps, ctpm = mean([tps[a] for a in CTRL]), mean([tpm[a] for a in CTRL])
    ann["tokens_per_verify_step_t1"] = {"pct1_window": tps, "mixed_c1": tpm,
                                        "B_over_ctrl_pct1": (tps["B"] / ctps) if (tps["B"] and ctps) else None,
                                        "B_over_ctrl_mixed": (tpm["B"] / ctpm) if (tpm["B"] and ctpm) else None}
    pcs = {}
    for c in CLASSES:
        for m in MODES:
            v = {a: cells[a]["pc"]["t1"]["acc"][(c, m)] for a in ARMS}
            cm = mean([v[a] for a in CTRL])
            pcs[f"{c}/{m}"] = {"B": None if v["B"] is None else 1 + v["B"], "ctrl": None if cm is None else 1 + cm}
    ann["tokens_per_verify_step_t1_per_class"] = pcs
    ann["x3_accept_canary"] = {w: {c: {a: cells[a]["pc"][w]["x3"][c] for a in ARMS} for c in CLASSES} for w in ("t1", "t0")}
    ann["perclass_hygiene"] = {f"{a}/{w}": cells[a]["pc"][w]["hygiene"] for a in ARMS for w in ("t1", "t0")}
    ann["boot_block_lines"] = {a: cells[a]["block_boot_sample"] for a in ARMS}
    ann["boot_prob_lines"] = {a: cells[a]["prob_boot_sample"] for a in ARMS}
    ann["accept_per_position_pct1"] = {a: (cells[a]["m_pct1"] or {}).get("accept_per_pos") for a in ARMS}
    ann["g6_history"] = g6_history(calc.get("G6") or {}, status, inv)
    ann["g6_per_transcript"] = (calc.get("G6") or {}).get("per_transcript")
    ann["g2b_shell_ops_history"] = dict(G2B_SHELL_HISTORY, BR6_B=(None if not (calc.get("G2b_rows") or {}).get("shell_ops") else
                                        f"{calc['G2b_rows']['shell_ops']['rel'] * 100:+.1f}% vs bar {calc['G2b_rows']['shell_ops']['bar'] * 100:.1f}% "
                                        f"({'PASS' if calc['G2b_rows']['shell_ops']['pass'] else 'FAIL'})"))
    ann["mem_min_by_phase"] = load_mem(R)
    ann["restore_watchdog"] = jl(R / "restore-watchdog.json")
    ann["kv_v28prob_boot_log"] = kv_ref
    ann["adaptive_ar_lines_cells_delta"] = {a: (cells[a]["ar_lines"] or {}).get("_cells_delta") for a in ARMS}
    ann["go"] = {a: cells[a]["go"] for a in ARMS}
    ann["host_avail_l9_gib"] = {a: (cells[a]["facts"] or {}).get("host_avail_l9_gib") for a in ARMS}
    ann["rtx_used_after_bind_mib"] = {a: (cells[a]["facts"] or {}).get("rtx_used_after_bind_mib") for a in ARMS}
    calc["G2_ratio_by_cell"] = {f"{r['class']}/{r['mode']}": round(r["ratio"], 4) for r in rows}
    slim = {a: {"facts": cells[a]["facts"], "autotune": cells[a]["autotune"], "mixed": cells[a]["mixed"],
                "perclass_t1": {f"{c}/{m}": cells[a]["pc"]["t1"]["cells"][(c, m)] for c in CLASSES for m in MODES},
                "perclass_t0": {f"{c}/{m}": cells[a]["pc"]["t0"]["cells"][(c, m)] for c in CLASSES for m in MODES},
                "turns": cells[a]["turns"], "turns_src": cells[a]["turns_src"], "ar_cells_delta": (cells[a]["ar_lines"] or {}).get("_cells_delta"),
                "errcount_server": cells[a]["err_server"], "foreign": cells[a]["foreign"],
                "metrics_pct1": cells[a]["m_pct1"], "metrics_mixed": cells[a]["m_mixed"]} for a in ARMS}
    return {"card": "BR6 block verification vs v28-prob (GB300-only), G6 redesign (4 replay passes/arm, pooled-control mean bar)", "result": res, "suffix": suffix, "gates": gates, "gate_status": status,
            "invalid_arms": inv, "valid": not inv and not b_foreign, "annotations": ann, "calc": calc, "cells": slim, "R": str(R)}


def g6_rule(ctrl_passes, b_passes):
    """the pre-registered G6 statistic on lists of per-pass counts -> dict (no validity checks; used for BR6 and the history)"""
    pool = [x for a in ctrl_passes for x in a]
    means = [statistics.mean(a) if a else None for a in ctrl_passes]
    lo, hi = (min(pool) - G6_MARGIN, max(pool) + G6_MARGIN) if pool else (None, None)
    mb = statistics.mean(b_passes) if b_passes else None
    dm = abs(means[0] - means[1]) if (len(means) == 2 and None not in means) else None
    inside = (mb is not None and lo is not None and lo <= mb <= hi)
    return {"pool": sorted(pool), "ctrl_means": means, "ctrl_mean_diff": dm, "bar": [lo, hi], "B_mean": mb,
            "inside": inside, "ctrl_agree": (dm is not None and dm <= G6_CTRL_MEAN_TOL)}


def g6_eval(cells):
    fail, inv, txt = False, set(), []
    P = {a: (cells[a]["turns"] or []) for a in ARMS}
    srcs = {a: cells[a].get("turns_src") for a in ARMS}
    cnt = {a: [p["tool_emitted"] for p in P[a]] for a in ARMS}
    out = {"spec": f"{TURNS_REQ} passes/arm; mean(B) in [min(pooled ctrl) - {G6_MARGIN:g}, max(pooled ctrl) + {G6_MARGIN:g}]; "
                   f"|mean(A) - mean(A2)| <= {G6_CTRL_MEAN_TOL:g} else INVALID-CONTROL; B errors 0",
           "passes": cnt, "src": srcs}
    # per-arm validity: exactly TURNS_REQ passes used (first TURNS_REQ if more), each with TURNS_N tool turns; errors
    use, problems = {}, {}
    for a in ARMS:
        ps = P[a]
        pr = []
        if not ps:
            pr.append("UNMEASURED")
        elif len(ps) < TURNS_REQ:
            pr.append(f"INSUFFICIENT PASSES ({len(ps)} < {TURNS_REQ})")
        if len(ps) > TURNS_REQ:
            txt.append(f"{a}: {len(ps)} passes present, first {TURNS_REQ} used (pre-registered count)")
        u = ps[:TURNS_REQ]
        bad_n = [k + 1 for k, p in enumerate(u) if p["tool_turns"] != TURNS_N]
        if bad_n:
            pr.append(f"pass(es) {bad_n} have tool_turns != {TURNS_N} (fixture drift)")
        er = sum(p["errors"] for p in u)
        if er:
            pr.append(f"errors {er}")
        use[a], problems[a] = u, pr
    for a in CTRL:
        if problems[a]:
            inv.add(a); txt.append(f"control {a}: " + "; ".join(problems[a]) + " -> INVALID")
    if problems["B"]:
        fail = True; txt.append("B: " + "; ".join(problems["B"]) + " -> FAIL")
    ruled = g6_rule([[p["tool_emitted"] for p in use[a]] for a in CTRL], [p["tool_emitted"] for p in use["B"]])
    out.update({k: ruled[k] for k in ("pool", "ctrl_means", "ctrl_mean_diff", "bar", "B_mean")})
    bc = [p["tool_emitted"] for p in use["B"]]
    out["B_min_max"] = [min(bc), max(bc)] if bc else None
    out["B_errors"] = sum(p["errors"] for p in use["B"])
    full = not any(problems[a] for a in ARMS)
    if full:
        if not ruled["ctrl_agree"]:
            inv.update(CTRL)
            txt.append(f"INVALID-CONTROL(G6): control means A {ruled['ctrl_means'][0]:.2f} vs A2 {ruled['ctrl_means'][1]:.2f} differ by "
                       f"{ruled['ctrl_mean_diff']:.2f} > {G6_CTRL_MEAN_TOL:g}")
        ok = ruled["inside"] and out["B_errors"] == 0
        fail = fail or not ok
        out["pass"] = ok and ruled["ctrl_agree"]
        txt.insert(0, f"B mean {ruled['B_mean']:.2f}/{TURNS_N} (passes {cnt['B'][:TURNS_REQ]}, min/max {out['B_min_max']}) vs pooled-control bar "
                      f"[{ruled['bar'][0]:g}, {ruled['bar'][1]:g}] = [min-{G6_MARGIN:g}, max+{G6_MARGIN:g}] of pooled 8 control passes "
                      f"{ruled['pool']} (A {cnt['A'][:TURNS_REQ]} mean {ruled['ctrl_means'][0]:.2f}, A2 {cnt['A2'][:TURNS_REQ]} mean "
                      f"{ruled['ctrl_means'][1]:.2f}, |dA| {ruled['ctrl_mean_diff']:.2f} <= {G6_CTRL_MEAN_TOL:g}: {ruled['ctrl_agree']}); "
                      f"B errors {out['B_errors']}{'' if ok else ' FAIL'}")
    else:
        out["pass"] = False
        if all(P[a] for a in ARMS):   # exploratory reading on whatever passes exist — NOT gated
            ex = g6_rule([cnt[a] for a in CTRL], cnt["B"])
            out["exploratory"] = ex
            txt.append(f"exploratory (NOT gated; passes A {cnt['A']} A2 {cnt['A2']} B {cnt['B']}): B mean {ex['B_mean']:.2f} vs "
                       f"[{ex['bar'][0]:g}, {ex['bar'][1]:g}] inside={ex['inside']}; ctrl means {[round(m, 2) for m in ex['ctrl_means']]} "
                       f"diff {ex['ctrl_mean_diff']:.2f} agree={ex['ctrl_agree']}")
    hist = {}
    for x in ruled["pool"]:
        hist[x] = hist.get(x, 0) + 1
    out["pool_hist"] = hist
    txt.append(f"pooled control distribution {dict(sorted(hist.items()))}; text-turn emissions B {[p['text_emitted'] for p in use['B']]} "
               f"(temp 0: replay_turns.py, {TURNS_REQ} x PASSES=1)")
    out["per_transcript"] = per_transcript(P)
    out["fail"], out["inv"], out["txt"] = fail, sorted(inv, key=ARMS.index), txt
    return out


def per_transcript(P):
    """ANNOTATION: for each tool transcript, the emitted (1) / not (0) string per arm across its passes; flags a CONSISTENT B flip
    (B identical on all its >= 2 passes and every control pass (>= 2) the opposite) vs a count wobble."""
    idx = sorted({i for a in ARMS for p in P[a] for i in p["tool_idx"]})
    rows, flips = [], []
    for i in idx:
        r = {"i": i}
        for a in ARMS:
            r[a] = "".join("1" if i in p["emitted_idx"] else "0" for p in P[a] if i in p["tool_idx"])
        cp = r["A"] + r["A2"]
        allb, allc = set(r["B"]), set(cp)
        if len(r["B"]) >= 2 and len(cp) >= 2 and len(allb) == 1 and len(allc) == 1 and allb != allc:
            r["flag"] = "B-CONSISTENT-FLIP " + ("(B never emits)" if r["B"][0] == "0" else "(B always emits)")
            flips.append(i)
        elif len(set(r["A"] + r["A2"] + r["B"])) == 1:
            r["flag"] = "stable"
        else:
            r["flag"] = "wobble"
        rows.append(r)
    return {"rows": rows, "consistent_B_flips": flips,
            "stable": [r["i"] for r in rows if r["flag"] == "stable"], "wobble": [r["i"] for r in rows if r["flag"] == "wobble"]}


def g6_history(t, status, inv=()):
    """ANNOTATION: BR4/BR5 per-pass counts next to BR6's, and the pre-registered BR6 statistic applied to the 2-pass receipts
    (information only — those cards were judged by the old 2-pass min..max rule)"""
    lines = []
    for card, h in G6_HISTORY.items():
        ctrl = [h["ctrl"]["A"], h["ctrl"]["A2"]]
        for cand, ps in h["cand"].items():
            r = g6_rule(ctrl, ps)
            lines.append(f"{card} {cand}: passes {ps} vs ctrl A {h['ctrl']['A']} A2 {h['ctrl']['A2']} (old 2-pass rule: {h['old_gate'][cand]}); "
                         f"BR6 statistic on 2 passes (info): mean {r['B_mean']:.2f} vs [{r['bar'][0]:g}, {r['bar'][1]:g}] inside={r['inside']}, "
                         f"ctrl mean diff {r['ctrl_mean_diff']:.2f} agree={r['ctrl_agree']}")
    p = t.get("passes") or {}
    bm = t.get("B_mean")
    lines.append(f"BR6: A {p.get('A')} A2 {p.get('A2')} B {p.get('B')}; B mean {'-' if bm is None else f'{bm:.2f}'} vs bar {t.get('bar')} "
                 f"(G6 {status.get('G6_turns')})" + (f" — card INVALID-CONTROL({','.join(inv)})" if inv else ""))
    allp = sorted([x for h in G6_HISTORY.values() for d in (h["ctrl"], h["cand"]) for ps in d.values() for x in ps]
                  + [x for a in ARMS for x in (p.get(a) or [])])
    lines.append(f"all replay passes on this fixture/lane, BR4+BR5 (constants) + this tree (sorted): {allp}"
                 " — in a replay of BR5 receipts 'this tree' duplicates BR5")
    return {"lines": lines}


def f2(x, nd=1):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x))


def table(v):
    L = [f"Card BR6 verdict: {v['result']}   suffix {v['suffix']}", "",
         "| gate | status | detail |", "|---|---|---|"]
    for k, (r, t) in v["gates"].items():
        L.append(f"| {k} | {v['gate_status'][k]} | {t} |")
    c = v["cells"]
    L += ["", "| arm | KV tokens | prob_in_args | block_in_args | boot s | mixed C1 t1 mean | t1 prose C1 | t1 tool_json C1 | t0 prose C1 | swaps during cells | ff | foreign |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in ARMS:
        x = c[a]; f = x["facts"] or {}
        L.append(f"| {a} | {f.get('kv_tokens', '-')} | {f.get('prob_in_args', '-')} | {f.get('block_in_args', '-')} | {f.get('boot_s', '-')} | "
                 f"{f2(mean([r['agg_tok_s_all'] for r in x['mixed']]), 2)} | {f2(x['perclass_t1']['prose/c1'])} | "
                 f"{f2(x['perclass_t1']['tool_json/c1'])} | {f2(x['perclass_t0']['prose/c1'])} | {(x['ar_cells_delta'] or {}).get('swaps', '-')} | "
                 f"{x['errcount_server']} | {sum(vv or 0 for vv in x['foreign'].values()) if x['foreign'] else '-'} |")
    t = v["annotations"]["tokens_per_verify_step_t1"]
    L += ["", "ANNOTATION (not a gate) — accepted tokens per verify step at temp 1.0 (1 + accepted/drafts from /metrics):",
          "  pct1 window: " + ", ".join(f"{a}={f2(t['pct1_window'][a], 3)}" for a in ARMS)
          + f"  B/ctrl={f2(t['B_over_ctrl_pct1'], 3)};  mixed C1: " + ", ".join(f"{a}={f2(t['mixed_c1'][a], 3)}" for a in ARMS)
          + f"  B/ctrl={f2(t['B_over_ctrl_mixed'], 3)}",
          "  per class (B vs ctrl mean): " + "; ".join(f"{k} {f2(x['B'], 2)}/{f2(x['ctrl'], 2)}"
                                                       for k, x in v["annotations"]["tokens_per_verify_step_t1_per_class"].items()),
          "ANNOTATION — x3 acceptance canary (agent_fixture_x3, temp 0, inside each window): " + "; ".join(
              f"{w}/{cl} " + "/".join(f2(v['annotations']['x3_accept_canary'][w][cl][a], 3) for a in ARMS)
              for w in ("t1",) for cl in CLASSES) + " (A/B/A2)",
          f"ANNOTATION — G2 B/mean(A,A2) by cell: {v['calc'].get('G2_ratio_by_cell')}",
          f"ANNOTATION — per-position acceptance (pct1 window, accepted_pos/drafts): {v['annotations']['accept_per_position_pct1']}",
          f"ANNOTATION — KV tokens in the v28-prob boot log: {v['annotations']['kv_v28prob_boot_log']}; host_avail at rehome layer 9 (GiB): {v['annotations']['host_avail_l9_gib']}",
          f"ANNOTATION — adaptive residency during cells (/ar line deltas pct1-start -> turns-end): {v['annotations']['adaptive_ar_lines_cells_delta']}",
          f"ANNOTATION — production (v27) watchdog after restore: {v['annotations']['restore_watchdog']}"]
    pt = v["annotations"].get("g6_per_transcript") or {}
    L += ["", "ANNOTATION — G6 per-transcript agreement (tool transcripts; 1 = tool call emitted in that pass, passes in order; not a gate):",
          "| transcript | A | B | A2 | flag |", "|---|---|---|---|---|"]
    for r in pt.get("rows") or []:
        L.append(f"| {r['i']} | {r['A'] or '-'} | {r['B'] or '-'} | {r['A2'] or '-'} | {r['flag']} |")
    L.append(f"  consistent B flips: {pt.get('consistent_B_flips')}; stable: {pt.get('stable')}; wobble: {pt.get('wobble')}")
    L += ["", "BR4/BR5 G6 HISTORY (annotation, not a gate — the gates above decide):"] + ["  " + x for x in v["annotations"]["g6_history"]["lines"]]
    L.append(f"  G2b shell_ops t0 C1 history: {v['annotations']['g2b_shell_ops_history']}")
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
    return 0 if v["result"] == "PASS" else (1 if v["result"] == "FAIL" else 3)


if __name__ == "__main__":
    sys.exit(main())
