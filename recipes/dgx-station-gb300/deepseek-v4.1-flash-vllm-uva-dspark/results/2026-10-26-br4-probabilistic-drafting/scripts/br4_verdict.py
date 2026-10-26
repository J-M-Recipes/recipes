#!/usr/bin/env python3
"""br4_verdict.py — Card BR4 verdict (DSpark PROBABILISTIC drafting, alone [P] and with BLOCK verification on top [PB], vs the
production greedy-draft / sequential-verify config, A -> P -> PB -> A2) on the GB300-ONLY v26seed "Seeded Adaptive" recipe (Grace
UVA offload, no RTX sidecar). Pure stdlib. Clone of br3_verdict.py @da59e401 (DIFF-br4-verdict-vs-br3.txt shows every change).

Usage:  br4_verdict.py <R> [--out verdict.json]   -> <R>/verdict.json + human table on stdout.
        rc 0 PASS (both candidates), 1 FAIL or PASS-P / PASS-PB (one candidate), 3 INVALID (control polluted/missing, or both
        candidates INVALID), 2 no input.

Two candidates, each judged against mean(A, A2) with the SAME bars as BR3 (G0..G7). Controls are shared: a control that is
missing / polluted makes the whole card INVALID-CONTROL-<arms> (no candidate can be scored without both controls).
Shape (KV3 / recipe-factory adapter): {"result", "suffix", "gates": {name: [PASS|FAIL, text]}, ...}. "gates" PASS only if the gate
passes for BOTH candidates; the per-candidate tri-state lives in "gate_status_by_candidate" {name: {P: .., PB: ..}} and the
overall tri-state in "gate_status" {name: PASS|FAIL|INVALID}; "per_candidate" {P: {result, failed}, PB: {...}}.
UNMEASURED semantics (KV3): a missing cell on a candidate = FAIL for that candidate; on a control = that gate INVALID.

Differences vs BR3:
  * arms A P PB A2; candidates P (spec + "draft_sample_method":"probabilistic") and PB (+ "rejection_sample_method":"block").
  * G0: prob_in_args want 0/1/1/0, block_in_args want 0/0/1/0; P/PB Args == v26seed except the spec element, which must equal the
    exact expected strings (string AND json level).
  * G0 KV rule: A2 == A exactly (odd control named by the v26seed boot-log value); each candidate within [A x (1 - 1.5 %), A]
    (probabilistic drafting allocates a bf16 draft-logits cache max_num_reqs x k x vocab = 24 x 5 x 129280 x 2 B ~ 31 MB in the
    speculator __init__, i.e. BEFORE KV profiling; at ~2,254 B/token that is ~ -13.8k tokens ~ -0.55 %). P == PB is reported (info).
    KV_CAND_MAX_DROP = 0.0 restores the literal "all arms equal A" rule.
  * G1 temp-0 referee for an odd control = mean of the candidates' temp-0 values (both are parity at temp 0, see G2b).
  * G2b stays a two-sided PARITY gate: in the 0909 image, at temperature 0 gumbel_noised_argmax is a plain argmax (no noise, no
    temperature) so probabilistic drafts == greedy drafts, the greedy rejection path never reads the draft logits, and block
    verification is skipped on greedy rows (USE_BLOCK_VERIFICATION and not is_greedy). Only the draft-logits cache write is extra.
  * annotations: headline PB vs P vs A on tokens/verify-step (pct1 window, mixed C1) and per class C1/C8 (tok/s ratio and tokens/
    step), per-position acceptance (vllm:spec_decode_num_accepted_tokens_per_pos), TF P-vs-PB.

Inputs (<R> = ~/dsv41/m3/br4/results-cardBR4, written by cardBR4_runner.sh):
  <X>/facts.json, args.json, env.json, binds.json, go.json, boot-autotune.log, boot-block.log, boot-prob.log, ar-lines.tsv  (X in ARMS)
  args-ref-<REF>.json, env-ref-<REF>.json, binds-ref-<REF>.json, kv-ref.txt                     control ground truth
  <X>/perclass-t1/perclass-table-BR4-<X>-t1.json (+ raw perclass-BR4-<X>-t1-r{1,2}-{c1,c8}.json)   temp 1.0 (G1, G2)
  <X>/perclass-t0/perclass-table-BR4-<X>-t0.json                                         stock temp-0 window (G1, G2b)
  <X>/mixed-BR4-<X>-c1-r<k>.json      ar_mixed_t.py C1 120 s at temp 1.0 (G3)
  tf-compare-BR4-{A,A2}-vs-BR4-{P,PB}.json, tf-compare-BR4-A-vs-BR4-A2.json (floor), tf-compare-BR4-P-vs-BR4-PB.json (info) (G4)
  flipsm-{P,PB}-vs-{A,A2}.txt, flipsm-A2-vs-A.txt   flips_margin_p2.py output, last line GATE ... (G4 margin)
  gauntlet-<X>/summary.json (G5) ; <X>/turns-BR4-<X>.json (G6) ; errcount-<X>.txt (G7)
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

ARMS = ("A", "P", "PB", "A2")
CTRL = ("A", "A2")
CANDS = ("P", "PB")
CLASSES = ("prose", "structured", "code", "shell_ops", "tool_json")
MODES = ("c1", "c8")
REF = "dsv41-vllm-1017-v26seed-BOUND-REF"
IMAGE_ID = "sha256:00d577a6a63281e15336029d5bcee4e9a2cf182214a4f20ba6111b1c8e79893d"
HOOK_KEYS_N = 40        # PIN_HOT rehome layer= lines (one per MoE layer)
SPEC_STR = {"P": ',"draft_sample_method":"probabilistic"',
            "PB": ',"draft_sample_method":"probabilistic","rejection_sample_method":"block"'}
SPEC_KEYS = {"P": {"draft_sample_method": "probabilistic"},
             "PB": {"draft_sample_method": "probabilistic", "rejection_sample_method": "block"}}
WANT = {"A": (0, 0), "P": (1, 0), "PB": (1, 1), "A2": (0, 0)}   # (prob_in_args, block_in_args)
KV_CAND_MAX_DROP = 0.015   # candidate KV may sit up to 1.5 % below A (draft-logits cache precedes KV profiling); 0.0 = strict
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


def ratio(a, b):
    return (a / b) if (a is not None and b) else None


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
    """per-position accepted counters {pos: value} (summed over other labels)"""
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
    s, e = out.get("pct1-start"), out.get("turns-end") or out.get("stop")
    if s and e:
        out["_cells_delta"] = {k: e.get(k, 0) - s.get(k, 0) for k in ("swaps", "telemetry", "steps_bytes")
                               if isinstance(e.get(k), int) and isinstance(s.get(k), int)}
    return out


def norm_binds(L):
    return sorted(("<AR>:/ar" if b.endswith(":/ar") else b) for b in (L or []))


def load_perclass(A, arm, win):
    tag = f"BR4-{arm}-{win}"
    t = jl(A / f"perclass-{win}" / f"perclass-table-{tag}.json")
    out = {"table": t is not None, "cells": {}, "x3": {}, "hygiene": (t or {}).get("hygiene"), "acc": {}, "spread": {}}
    S = (t or {}).get("summary") or {}
    for c in CLASSES:
        for m in MODES:
            out["cells"][(c, m)] = ((S.get(c) or {}).get(m) or {}).get("mean")
            out["spread"][(c, m)] = ((S.get(c) or {}).get(m) or {}).get("spread")
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
    for k in ("block", "prob"):
        try:
            c[f"{k}_boot_sample"] = Path(A / f"boot-{k}.log").read_text().splitlines()[:3]
        except Exception:  # noqa: BLE001
            c[f"{k}_boot_sample"] = []
    c["pc"] = {w: load_perclass(A, arm, w) for w in ("t1", "t0")}
    reps = []
    for f in sorted(glob.glob(str(A / f"mixed-BR4-{arm}-c1-r*.json")), key=lambda s: int(s.rsplit("-r", 1)[1].split(".")[0])):
        d = jl(f)
        if d and d.get("agg_tok_s_all") is not None:
            reps.append({"rep": int(f.rsplit("-r", 1)[1].split(".")[0]), "agg_tok_s_all": d["agg_tok_s_all"],
                         "errors": d.get("errors", 0), "n": d.get("n"), "temperature": (d.get("sampling") or {}).get("temperature")})
    c["mixed"] = reps
    c["mixed_mean"] = mean([r["agg_tok_s_all"] for r in reps])
    c["tf_capture"] = (A / f"tf-BR4-{arm}.jsonl").exists() and (A / f"tf-BR4-{arm}.jsonl").stat().st_size > 0
    c["gauntlet"] = jl(R / f"gauntlet-{arm}" / "summary.json")
    t = jl(A / f"turns-BR4-{arm}.json")
    if t and t.get("passes"):
        c["turns"] = [{"tool_emitted": sum(1 for r in p if r.get("has_tool") and r.get("emitted")),
                       "tool_turns": sum(1 for r in p if r.get("has_tool")),
                       "text_emitted": sum(1 for r in p if not r.get("has_tool") and r.get("emitted")),
                       "errors": sum(1 for r in p if r.get("error"))} for p in t["passes"]]
    else:
        c["turns"] = None
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
    """gate accumulator: failed candidates + controls that make it INVALID"""

    def __init__(self):
        self.fail, self.inv, self.txt = set(), set(), []

    def bad(self, arm):
        """a defect on <arm>: control -> INVALID(arm); candidate -> FAIL(that candidate)"""
        (self.inv.add(arm) if arm in CTRL else self.fail.add(arm))

    def status(self, cand=None):
        if self.inv:
            return "INVALID"
        if cand is None:
            return "FAIL" if self.fail else "PASS"
        return "FAIL" if cand in self.fail else "PASS"


def verdict(R):
    cells = {a: load_arm(R, a) for a in ARMS}
    prod_args = jl(R / f"args-ref-{REF}.json")      # "prod" = the v26seed control ground truth in this card
    prod_env = jl(R / f"env-ref-{REF}.json")
    prod_binds = jl(R / f"binds-ref-{REF}.json")
    try:
        kv_ref = (R / "kv-ref.txt").read_text().strip() or None
    except Exception:  # noqa: BLE001
        kv_ref = None
    gs = {}
    calc = {}

    # ---- G0 binding + wiring: every arm bound; KV rule (below); prob_in_args 0/1/1/0, block_in_args 0/0/1/0; A/A2 Args ==
    # v26seed; P/PB Args == v26seed except the spec (== v26seed + the exact keys); Env == v26seed (order-insensitive); Binds ==
    # v26seed except the /ar source; image 0909; hook wiring/guard lines; GO + unfreeze acknowledged (production regime).
    # A control with broken wiring / regime / odd KV is INVALID; a candidate -> FAIL. As in BR3, a control's Args/Env/Binds/
    # image/key-count mismatch is a recipe FAIL (both candidates), not INVALID.
    g = G()
    kvs = {a: (cells[a]["facts"] or {}).get("kv_tokens") or None for a in ARMS}
    for a in ARMS:
        f = cells[a]["facts"]
        if not f or not f.get("bound_at"):
            g.bad(a); g.txt.append(f"{a}: NOT BOUND"); continue
        recipe_bad = (lambda: g.fail.update(CANDS)) if a in CTRL else (lambda a=a: g.fail.add(a))
        wp, wb = WANT[a]
        bits = [f"kv={kvs[a] or 'MISSING'}", f"prob_in_args={f.get('prob_in_args')}(want {wp})",
                f"block_in_args={f.get('block_in_args')}(want {wb})"]
        if f.get("prob_in_args") != wp:
            recipe_bad(); bits.append("PROB!")
        if f.get("block_in_args") != wb:
            recipe_bad(); bits.append("BLOCK!")
        args = cells[a]["args"]
        if prod_args is None or args is None:
            recipe_bad(); bits.append("args UNMEASURED")
        elif a in CTRL:
            ok = args == prod_args
            bits.append(f"args==v26seed {ok}")
            if not ok:
                recipe_bad()
        else:
            ps, bs = spec_of(prod_args), spec_of(args)
            rest_ok = len(args) == len(prod_args) and all(x == y for x, y in zip(args, prod_args) if x != bs or y != ps)
            spec_ok = ps is not None and bs == ps[:-1] + SPEC_STR[a] + "}"
            try:
                spec_ok = spec_ok and json.loads(bs) == dict(json.loads(ps), **SPEC_KEYS[a])
            except Exception:  # noqa: BLE001
                spec_ok = False
            bits.append(f"args==v26seed-except-spec {rest_ok} spec==v26seed+{'+'.join(SPEC_KEYS[a].values())} {spec_ok}")
            if not (rest_ok and spec_ok):
                recipe_bad()
        env, binds = cells[a]["env"], cells[a]["binds"]
        eok = (env is not None and prod_env is not None and sorted(env) == sorted(prod_env))   # order-insensitive
        bok = (binds is not None and prod_binds is not None and norm_binds(binds) == norm_binds(prod_binds))
        iok = f.get("image") == IMAGE_ID
        bits += [f"env==v26seed(sorted) {eok}", f"binds==v26seed(except /ar) {bok}", f"image0909 {iok}"]
        if not (eok and bok and iok):
            recipe_bad()
        hk = (f.get("rehome_done") == 1 and f.get("rehome_layers") == HOOK_KEYS_N and f.get("pin_abort") == 0
              and f.get("hook_failed") == 0 and not f.get("hook_keys_missing"))
        bits.append(f"hook rehome {f.get('rehome_layers')}/{HOOK_KEYS_N} done={f.get('rehome_done')} abort={f.get('pin_abort')} "
                    f"failed={f.get('hook_failed')} missing=[{f.get('hook_keys_missing') or ''}] host_avail@L9={f.get('host_avail_l9_gib')}GiB")
        if not hk:
            g.bad(a); bits.append("HOOK!")
        go = cells[a]["go"] or {}
        gok = (go.get("go_seen_log") or 0) >= 1 and (go.get("unfreeze_log") or 0) >= 1
        bits.append(f"GO+unfreeze {go.get('go_seen_log')}/{go.get('unfreeze_log')} ctrl={json.dumps(go.get('ctrl'))}")
        if not gok:
            g.bad(a); bits.append("REGIME!")
        g.txt.append(f"{a}: " + " ".join(bits))
    # KV rule. Controls: A2 == A exactly; if not, the v26seed boot-log value names the odd control (neither matches -> both).
    # Candidates: A x (1 - KV_CAND_MAX_DROP) <= X <= A (draft-logits cache precedes KV profiling). P == PB is info.
    bound = {a: bool(cells[a]["facts"] and cells[a]["facts"].get("bound_at")) for a in ARMS}
    kvi = {a: kvint(kvs[a]) for a in ARMS}
    kvr = kvint(kv_ref)
    kvtxt = f"KV A={kvs['A']} P={kvs['P']} PB={kvs['PB']} A2={kvs['A2']} (v26seed boot log {kv_ref}; candidate drop bar {KV_CAND_MAX_DROP * 100:.1f}%)"
    for a in CTRL:
        if bound[a] and kvi[a] is None:
            g.inv.add(a); kvtxt += f" KV MISSING on bound {a}"
    if kvi["A"] is not None and kvi["A2"] is not None and kvi["A"] != kvi["A2"]:
        odd = [x for x in CTRL if kvr is None or kvi[x] != kvr] or list(CTRL)
        g.inv.update(odd); kvtxt += f" KV! A2 != A (odd control(s) {odd} vs v26seed log)"
    for x in CANDS:
        if not bound[x]:
            continue
        if kvi[x] is None:
            g.fail.add(x); kvtxt += f" KV MISSING on bound {x}"; continue
        if kvi["A"] is None:
            continue
        drop = (kvi["A"] - kvi[x]) / kvi["A"]
        ok = 0.0 <= drop <= KV_CAND_MAX_DROP
        kvtxt += f"; {x} vs A {-drop * 100:+.2f}%{'' if ok else ' KV!'}"
        if not ok:
            g.fail.add(x)
    if kvi["P"] is not None and kvi["PB"] is not None:
        kvtxt += f"; P==PB {kvi['P'] == kvi['PB']} (info)"
    calc["kv"] = {"tokens": kvi, "v26seed_log": kvr,
                  "cand_drop": {x: ((kvi["A"] - kvi[x]) / kvi["A"]) if (kvi["A"] and kvi[x] is not None) else None for x in CANDS}}
    g.txt.append(kvtxt)
    at = {a: cells[a]["autotune"] for a in ARMS}
    bs_ = {a: (cells[a]["facts"] or {}).get("boot_s") for a in ARMS}
    cb = mean([bs_[a] for a in CTRL])
    hh = " | hash-hit shape (info): " + "; ".join(
        f"{a} loaded={at[a]['loaded']} saved={at[a]['saved']} same_dir={at[a]['loaded_dir'] == at[a]['saved_dir'] and at[a]['loaded_dir'] is not None} "
        f"autotune_span={at[a]['elapsed_s']}s boot={bs_[a]}s" for a in ARMS)
    dirs = {at[a]["loaded_dir"] for a in ARMS if at[a]["loaded_dir"]}
    hh += f"; one autotune key across arms: {len(dirs) == 1}; 0909 hash-hit shape (Loaded N == Saved N, same dir, span <= 60 s): " + ", ".join(
        f"{a}={bool(at[a].get('hash_hit_shape')) and (at[a]['elapsed_s'] is not None and at[a]['elapsed_s'] <= 60)}" for a in ARMS)
    for x in CANDS:
        if bs_[x] and cb:
            hh += f"; {x} boot / controls = {bs_[x] / cb:.2f} (<= 1.5 expected)"
    gs["G0_binding"] = (g, "; ".join(g.txt) + hh)

    # ---- G0b foreign POSTs (non-loopback) per arm; a candidate that served a foreign client is INVALID-<cand>
    g = G(); cand_foreign = set()
    for a in ARMS:
        fr = cells[a]["foreign"]
        n = sum(v or 0 for v in fr.values()) if fr else None
        g.txt.append(f"{a}={'UNMEASURED' if n is None else n} {fr or ''}")
        if n is None:
            if cells[a]["facts"]:
                g.bad(a)
        elif n > 0:
            g.bad(a)
            if a in CANDS:
                cand_foreign.add(a)
    gs["G0b_foreign"] = (g, "non-127.0.0.1 POST /v1 per arm: " + "; ".join(g.txt)
                         + (f" -> {','.join(sorted(cand_foreign, key=ARMS.index))} INVALID" if cand_foreign else ""))

    # ---- G1 control agreement: per class C1, both windows, |A-A2|/mean <= 6 %; C8 reported only
    g = G(); c8 = []
    for w in ("t1", "t0"):
        for c in CLASSES:
            va, va2 = (cells[x]["pc"][w]["cells"][(c, "c1")] for x in ("A", "A2"))
            vref = mean([cells[x]["pc"][w]["cells"][(c, "c1")] for x in CANDS])
            if va is None or va2 is None:
                for x, v in (("A", va), ("A2", va2)):
                    if v is None:
                        g.inv.add(x)
                g.txt.append(f"{w}/{c}/c1 UNMEASURED"); continue
            d = abs(va - va2) / ((va + va2) / 2)
            if d > CTRL_TOL:
                # The candidates are neutral referees only in the temp-0 window (both keys are no-ops on greedy rows). At temp
                # 1.0 they are expected to differ, so a t1 disagreement cannot name the odd control -> both.
                odd = ("A" if abs(vref - va) > abs(vref - va2) else "A2") if (vref is not None and w == "t0") else None
                g.inv.update([odd] if odd else ["A", "A2"])
                g.txt.append(f"{w}/{c}/c1 A={va:.1f} A2={va2:.1f} d={d * 100:.1f}% > 6% (odd={odd or 'A,A2'})")
            else:
                g.txt.append(f"{w}/{c}/c1 d={d * 100:.1f}%")
            ca, ca2 = cells["A"]["pc"][w]["cells"][(c, "c8")], cells["A2"]["pc"][w]["cells"][(c, "c8")]
            if ca and ca2:
                c8.append(f"{w}/{c}/c8 d={abs(ca - ca2) / ((ca + ca2) / 2) * 100:.1f}%")
    gs["G1_controls"] = (g, "C1 per class: " + "; ".join(g.txt) + " | C8 (info, bimodal class): " + "; ".join(c8))

    # ---- G2 PRIMARY per-class decode at temp 1.0, C1 and C8: X >= ctrl x (1 - max(2%, 1.5 x |A-A2| spread)) vs BOTH controls
    g = G(); rows = []
    for x in CANDS:
        sub = []
        for c in CLASSES:
            for m in MODES:
                va, va2, vb = (cells[y]["pc"]["t1"]["cells"][(c, m)] for y in ("A", "A2", x))
                if vb is None:
                    g.fail.add(x); sub.append(f"{c}/{m} {x} UNMEASURED"); continue
                if va is None or va2 is None:
                    g.inv.update([y for y, v in (("A", va), ("A2", va2)) if v is None]); sub.append(f"{c}/{m} control UNMEASURED"); continue
                cm = (va + va2) / 2; sp = abs(va - va2) / cm; bar = max(FLOOR, SPREAD_K * sp)
                ok = vb >= va * (1 - bar) and vb >= va2 * (1 - bar)
                if not ok:
                    g.fail.add(x)
                rows.append({"cand": x, "class": c, "mode": m, "A": va, "A2": va2, "X": vb, "ratio": vb / cm, "spread": sp, "bar": bar, "pass": ok})
                sub.append(f"{c}/{m} {x}/ctrl={vb / cm:.3f} (A {va:.1f} A2 {va2:.1f} {x} {vb:.1f}; bar -{bar * 100:.1f}%){'' if ok else ' FAIL'}")
        g.txt.append(f"[{x}] " + "; ".join(sub))
    calc["G2_rows"] = rows
    gs["G2_perclass_t1"] = (g, "temp 1.0 forced-length decode tok/s: " + " || ".join(g.txt))

    # ---- G2b temp-0 PARITY: X within control noise at C1 per class (two-sided); C8 info. In the 0909 image both keys are
    # no-ops on greedy rows (plain argmax drafting at temp 0; greedy rejection ignores draft logits; block skipped when greedy).
    g = G(); info = []
    for x in CANDS:
        sub = []
        for c in CLASSES:
            for m in MODES:
                va, va2, vb = (cells[y]["pc"]["t0"]["cells"][(c, m)] for y in ("A", "A2", x))
                if m == "c8":
                    if None not in (va, va2, vb):
                        info.append(f"{x} {c}/c8 {x}/ctrl={vb / ((va + va2) / 2):.3f}")
                    continue
                if vb is None:
                    g.fail.add(x); sub.append(f"{c} {x} UNMEASURED"); continue
                if va is None or va2 is None:
                    g.inv.update([y for y, v in (("A", va), ("A2", va2)) if v is None]); sub.append(f"{c} control UNMEASURED"); continue
                cm = (va + va2) / 2
                # BR1-rescore rule: the parity bar also honours the controls' own r1/r2 spread for this class (within-arm noise).
                wsp = []
                for y in CTRL:
                    try:
                        wsp.append(float(cells[y]["pc"]["t0"]["spread"].get((c, "c1"))))
                    except (TypeError, ValueError):
                        pass
                wspread = max(wsp or [0.0])
                bar = max(FLOOR, SPREAD_K * abs(va - va2) / cm, SPREAD_K * wspread); r = vb / cm - 1
                ok = abs(r) <= bar
                if not ok:
                    g.fail.add(x)
                sub.append(f"{c}/c1 {x} {pct(r, 1)} (|.| bar {bar * 100:.1f}%, ctrl r1/r2 spread {wspread * 100:.1f}%){'' if ok else ' OUTSIDE NOISE'}")
        g.txt.append(f"[{x}] " + "; ".join(sub))
    gs["G2b_t0_parity"] = (g, "stock temp-0 window (PARITY: draft_sample_method and block are no-ops on greedy rows; X must equal A within noise): "
                           + " || ".join(g.txt) + " | C8 info: " + "; ".join(info))

    # ---- G3 mixed C1 at temp 1.0
    g = G()
    ma, ma2 = cells["A"]["mixed_mean"], cells["A2"]["mixed_mean"]
    ctrl_ok = not (ma is None or ma2 is None or len(cells["A"]["mixed"]) < 2 or len(cells["A2"]["mixed"]) < 2)
    if not ctrl_ok:
        g.inv.update([y for y in CTRL if cells[y]["mixed_mean"] is None or len(cells[y]["mixed"]) < 2])
        g.txt.append("control UNMEASURED (need >= 2 reps each)")
    calc["G3"] = {}
    for x in CANDS:
        mb, nb = cells[x]["mixed_mean"], len(cells[x]["mixed"])
        if mb is None or nb < 2:
            g.fail.add(x); g.txt.append(f"[{x}] UNMEASURED (reps={nb}, need >= 2)"); continue
        if not ctrl_ok:
            continue
        cm = (ma + ma2) / 2; sp = abs(ma - ma2) / cm; bar = max(FLOOR, SPREAD_K * sp); r = mb / cm - 1
        if r < -bar:
            g.fail.add(x)
        calc["G3"][x] = {"X": mb, "A": ma, "A2": ma2, "ctrl_mean": cm, "spread": sp, "bar": bar, "rel": r}
        g.txt.append(f"[{x}] mean {mb:.2f} vs ctrl {cm:.2f} ({pct(r)}; A {ma:.2f} A2 {ma2:.2f}; bar -{bar * 100:.2f}%) "
                     f"reps={[round(y['agg_tok_s_all'], 1) for y in cells[x]['mixed']]} errors={sum(y['errors'] or 0 for y in cells[x]['mixed'])}"
                     f"{'' if r >= -bar else ' FAIL'}")
    gs["G3_mixed_c1_t1"] = (g, "ar_mixed_t C1 agg_tok_s_all at temp 1.0: " + "; ".join(g.txt))

    # ---- G4 TF (teacher-forced; temperature-independent; draft keys do not touch the target forward)
    g = G()
    miss = [a for a in CTRL if not cells[a]["tf_capture"]]
    fl = jl(R / "tf-compare-BR4-A-vs-BR4-A2.json")
    if miss:
        g.inv.update(miss); g.txt.append(f"control capture MISSING {miss}")
    for x in CANDS:
        if not cells[x]["tf_capture"]:
            g.fail.add(x); g.txt.append(f"[{x}] capture MISSING"); continue
        if miss:
            continue
        cmp = {a: jl(R / f"tf-compare-BR4-{a}-vs-BR4-{x}.json") for a in CTRL}
        if fl is None or any(v is None for v in cmp.values()):
            g.fail.add(x); g.txt.append(f"[{x}] compare file(s) MISSING with captures present (runner fault)"); continue
        floor = fl.get("top1_flip_rate")
        sub = []
        for a in CTRL:
            rate = cmp[a].get("top1_flip_rate")
            ok = rate is not None and floor is not None and rate <= TF_ABS and rate <= TF_FLOOR_K * floor
            if not ok:
                g.fail.add(x)
            sub.append(f"{x}-vs-{a} flips {(rate or 0) * 100:.3f}% (bar <=3% and <= {TF_FLOOR_K} x floor {(floor or 0) * 100:.3f}% = "
                       f"{TF_FLOOR_K * (floor or 0) * 100:.3f}%; token-mismatch docs {cmp[a].get('docs_token_mismatch')}){'' if ok else ' FAIL'}")
        mg = []
        for a in CTRL:
            p = R / f"flipsm-{x}-vs-{a}.txt"
            last = (p.read_text().strip().splitlines() or [""])[-1] if p.exists() else ""
            m = re.search(r"hi_agent_tool=(\d+)", last)
            sk = re.search(r"skipped_docs=(\d+)", last)
            mg.append((a, int(m.group(1)) if m else None, sk.group(1) if sk else "?"))
        if all(y[1] is not None for y in mg):
            bad = [y for y in mg if y[1] != 0]
            if bad:
                g.fail.add(x)
            sub.append("margin flips at ref p>=0.9 on agent/tool docs: " + ", ".join(f"vs {a}={n} (skipped_docs {s})" for a, n, s in mg)
                       + ("" if not bad else " FAIL"))
        else:
            sub.append("margin flips UNMEASURED (flips_margin_p2 output missing/unparsable; not failing)")
        g.txt.append(f"[{x}] " + "; ".join(sub))
    gs["G4_tf"] = (g, "; ".join(g.txt))

    # ---- G5 gauntlet 12/12 on each candidate (controls reported)
    g = G()
    gc = {a: (cells[a]["gauntlet"] or {}) for a in CTRL}
    for x in CANDS:
        gb = cells[x]["gauntlet"]
        if not gb:
            g.fail.add(x); g.txt.append(f"{x} UNMEASURED"); continue
        s, t = gb.get("success_count"), gb.get("task_count")
        if not (s == GAUNTLET_N and t == GAUNTLET_N):
            g.fail.add(x)
        g.txt.append(f"{x} {s}/{t}")
    g.txt.append("controls " + ", ".join(f"{a} {gc[a].get('success_count')}/{gc[a].get('task_count')}" for a in CTRL) + " (temp 0: run_suite.py)")
    gs["G5_gauntlet"] = (g, "; ".join(g.txt))

    # ---- G6 replay_turns tool-call emission per pass within [min, max] of A and A2 passes
    g = G()
    tc = [p["tool_emitted"] for a in CTRL for p in (cells[a]["turns"] or [])]
    cmiss = [a for a in CTRL if cells[a]["turns"] is None]
    if cmiss:
        g.inv.update(cmiss); g.txt.append("control UNMEASURED")
    for x in CANDS:
        tb = cells[x]["turns"]
        if not tb:
            g.fail.add(x); g.txt.append(f"[{x}] UNMEASURED"); continue
        if cmiss:
            continue
        lo, hi = min(tc), max(tc)
        be = [p["tool_emitted"] for p in tb]; berr = sum(p["errors"] for p in tb)
        ok = all(lo <= y <= hi for y in be) and berr == 0
        if not ok:
            g.fail.add(x)
        g.txt.append(f"[{x}] per pass {be}/{tb[0]['tool_turns']} vs control range [{lo},{hi}] (A {[p['tool_emitted'] for p in cells['A']['turns']]}, "
                     f"A2 {[p['tool_emitted'] for p in cells['A2']['turns']]}); errors {berr}; text-turn emissions "
                     f"{[p['text_emitted'] for p in tb]}{'' if ok else ' FAIL'}")
    g.txt.append("(temp 0: replay_turns.py)")
    gs["G6_turns"] = (g, "; ".join(g.txt))

    # ---- G7 server FF errors 0 + foreign POSTs 0 per arm (no sidecar, no card watchdog on this lane)
    g = G()
    for a in ARMS:
        e, fr = cells[a]["err_server"], cells[a]["foreign"]
        if not cells[a]["facts"] and a in CTRL:
            g.inv.add(a); g.txt.append(f"{a}: not bound"); continue
        n = sum(v or 0 for v in fr.values()) if fr else None
        if e is None or n is None:
            g.bad(a); g.txt.append(f"{a}: UNMEASURED (ff={e} foreign={n})"); continue
        bad = e != 0 or n != 0
        if bad:
            g.bad(a)
        g.txt.append(f"{a}: ff={e} foreign={n}{' BAD' if bad else ''}")
    gs["G7_errors"] = (g, "; ".join(g.txt))

    # ---- result (per candidate, shared controls)
    status = {k: v[0].status() for k, v in gs.items()}
    by_cand = {k: {x: v[0].status(x) for x in CANDS} for k, v in gs.items()}
    inv = sorted({a for v in gs.values() for a in v[0].inv}, key=ARMS.index)
    per = {}
    for x in CANDS:
        failed = [k for k in gs if by_cand[k][x] == "FAIL"]
        if inv:
            r = "INVALID-CONTROL"
        elif x in cand_foreign:
            r = "INVALID"
        elif failed:
            r = "FAIL"
        else:
            r = "PASS"
        per[x] = {"result": r, "failed": failed}
    if inv:
        res = "INVALID-CONTROL-" + "-".join(inv); suffix = "BR4-" + res
    else:
        tok = lambda x: x + (per[x]["result"] if per[x]["result"] != "FAIL" else "FAIL-" + "-".join(SHORT[k] for k in per[x]["failed"]))
        suffix = "BR4-" + "_".join(tok(x) for x in CANDS)
        passed = [x for x in CANDS if per[x]["result"] == "PASS"]
        if len(passed) == len(CANDS):
            res = "PASS"
        elif passed:
            res = "PASS-" + passed[0]
        elif all(per[x]["result"] == "INVALID" for x in CANDS):
            res = "INVALID-" + "-".join(CANDS)
        else:
            res = "FAIL"
    gates = {}
    for k, (g, txt) in gs.items():
        s = status[k]
        head = " ".join(f"{x}:{by_cand[k][x]}" for x in CANDS)
        gates[k] = ["PASS", txt] if s == "PASS" else ["FAIL", (f"INVALID-CONTROL({','.join(sorted(g.inv, key=ARMS.index))}): " if s == "INVALID" else f"[{head}] ") + txt]

    # ---- annotations (NOT gates). Headline: is PB > P > A at temp 1.0 on tokens/verify-step and per class C1/C8?
    ann = {}
    tps = {a: (cells[a]["m_pct1"] or {}).get("tokens_per_verify_step") for a in ARMS}
    tpm = {a: (cells[a]["m_mixed"] or {}).get("tokens_per_verify_step") for a in ARMS}
    ctps, ctpm = mean([tps[a] for a in CTRL]), mean([tpm[a] for a in CTRL])
    ann["tokens_per_verify_step_t1"] = {
        "pct1_window": tps, "mixed_c1": tpm,
        "P_over_ctrl_pct1": ratio(tps["P"], ctps), "PB_over_ctrl_pct1": ratio(tps["PB"], ctps), "PB_over_P_pct1": ratio(tps["PB"], tps["P"]),
        "P_over_ctrl_mixed": ratio(tpm["P"], ctpm), "PB_over_ctrl_mixed": ratio(tpm["PB"], ctpm), "PB_over_P_mixed": ratio(tpm["PB"], tpm["P"])}
    pcs, pct_ratio = {}, {}
    for c in CLASSES:
        for m in MODES:
            v = {a: cells[a]["pc"]["t1"]["acc"][(c, m)] for a in ARMS}
            cm = mean([v[a] for a in CTRL])
            pcs[f"{c}/{m}"] = {"P": None if v["P"] is None else 1 + v["P"], "PB": None if v["PB"] is None else 1 + v["PB"],
                               "ctrl": None if cm is None else 1 + cm}
            t = {a: cells[a]["pc"]["t1"]["cells"][(c, m)] for a in ARMS}
            ct = mean([t[a] for a in CTRL])
            pct_ratio[f"{c}/{m}"] = {"P/ctrl": ratio(t["P"], ct), "PB/ctrl": ratio(t["PB"], ct), "PB/P": ratio(t["PB"], t["P"])}
    ann["tokens_per_verify_step_t1_per_class"] = pcs
    ann["tok_s_ratio_t1_per_class"] = pct_ratio

    def order3(p, pb, c_):
        return None if None in (p, pb, c_) else (pb > p > c_)

    hl = {"pct1_tokens_per_step_PB>P>ctrl": order3(tps["P"], tps["PB"], ctps),
          "mixed_tokens_per_step_PB>P>ctrl": order3(tpm["P"], tpm["PB"], ctpm)}
    for m in MODES:
        o_tps = [order3(pcs[f"{c}/{m}"]["P"], pcs[f"{c}/{m}"]["PB"], pcs[f"{c}/{m}"]["ctrl"]) for c in CLASSES]
        o_tok = [None if None in (pct_ratio[f"{c}/{m}"]["P/ctrl"], pct_ratio[f"{c}/{m}"]["PB/P"]) else
                 (pct_ratio[f"{c}/{m}"]["PB/P"] > 1 and pct_ratio[f"{c}/{m}"]["P/ctrl"] > 1) for c in CLASSES]
        hl[f"{m}_classes_tokens_per_step_PB>P>ctrl"] = f"{sum(1 for o in o_tps if o)}/{sum(1 for o in o_tps if o is not None)}"
        hl[f"{m}_classes_tok_s_PB>P>ctrl"] = f"{sum(1 for o in o_tok if o)}/{sum(1 for o in o_tok if o is not None)}"
    ann["headline_PB_P_A"] = hl
    ann["accept_per_position_pct1"] = {a: (cells[a]["m_pct1"] or {}).get("accept_per_pos") for a in ARMS}
    tpp = jl(R / "tf-compare-BR4-P-vs-BR4-PB.json")
    ann["tf_P_vs_PB"] = None if tpp is None else {"top1_flip_rate": tpp.get("top1_flip_rate"), "docs_token_mismatch": tpp.get("docs_token_mismatch")}
    ann["x3_accept_canary"] = {w: {c: {a: cells[a]["pc"][w]["x3"][c] for a in ARMS} for c in CLASSES} for w in ("t1", "t0")}
    ann["perclass_hygiene"] = {f"{a}/{w}": cells[a]["pc"][w]["hygiene"] for a in ARMS for w in ("t1", "t0")}
    ann["boot_block_lines"] = {a: cells[a]["block_boot_sample"] for a in ARMS}
    ann["boot_prob_lines"] = {a: cells[a]["prob_boot_sample"] for a in ARMS}
    ann["mem_min_by_phase"] = load_mem(R)
    ann["restore_watchdog"] = jl(R / "restore-watchdog.json")
    ann["kv_v26seed_boot_log"] = kv_ref
    ann["adaptive_ar_lines_cells_delta"] = {a: (cells[a]["ar_lines"] or {}).get("_cells_delta") for a in ARMS}
    ann["go"] = {a: cells[a]["go"] for a in ARMS}
    ann["host_avail_l9_gib"] = {a: (cells[a]["facts"] or {}).get("host_avail_l9_gib") for a in ARMS}
    ann["rtx_used_after_bind_mib"] = {a: (cells[a]["facts"] or {}).get("rtx_used_after_bind_mib") for a in ARMS}
    calc["G2_ratio_by_cell"] = {f"{r['cand']} {r['class']}/{r['mode']}": round(r["ratio"], 4) for r in rows}
    slim = {a: {"facts": cells[a]["facts"], "autotune": cells[a]["autotune"], "mixed": cells[a]["mixed"],
                "perclass_t1": {f"{c}/{m}": cells[a]["pc"]["t1"]["cells"][(c, m)] for c in CLASSES for m in MODES},
                "perclass_t0": {f"{c}/{m}": cells[a]["pc"]["t0"]["cells"][(c, m)] for c in CLASSES for m in MODES},
                "turns": cells[a]["turns"], "ar_cells_delta": (cells[a]["ar_lines"] or {}).get("_cells_delta"),
                "errcount_server": cells[a]["err_server"], "foreign": cells[a]["foreign"],
                "metrics_pct1": cells[a]["m_pct1"], "metrics_mixed": cells[a]["m_mixed"]} for a in ARMS}
    return {"card": "BR4 probabilistic drafting (P) and probabilistic + block (PB) (GB300-only v26seed)", "result": res, "suffix": suffix,
            "gates": gates, "gate_status": status, "gate_status_by_candidate": by_cand, "per_candidate": per,
            "invalid_arms": inv, "invalid_candidates": sorted(cand_foreign, key=ARMS.index), "valid": not inv,
            "annotations": ann, "calc": calc, "cells": slim, "R": str(R)}


def f2(x, nd=1):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x))


def table(v):
    pc = v["per_candidate"]
    L = [f"Card BR4 verdict: {v['result']}   suffix {v['suffix']}   (P: {pc['P']['result']}; PB: {pc['PB']['result']})", "",
         "| gate | status | P | PB | detail |", "|---|---|---|---|---|"]
    for k, (r, t) in v["gates"].items():
        L.append(f"| {k} | {v['gate_status'][k]} | {v['gate_status_by_candidate'][k]['P']} | {v['gate_status_by_candidate'][k]['PB']} | {t} |")
    c = v["cells"]
    L += ["", "| arm | KV tokens | prob_in_args | block_in_args | boot s | mixed C1 t1 mean | t1 prose C1 | t1 tool_json C1 | t0 prose C1 | swaps during cells | ff | foreign |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in ARMS:
        x = c[a]; f = x["facts"] or {}
        L.append(f"| {a} | {f.get('kv_tokens', '-')} | {f.get('prob_in_args', '-')} | {f.get('block_in_args', '-')} | {f.get('boot_s', '-')} | "
                 f"{f2(mean([r['agg_tok_s_all'] for r in x['mixed']]), 2)} | {f2(x['perclass_t1']['prose/c1'])} | "
                 f"{f2(x['perclass_t1']['tool_json/c1'])} | {f2(x['perclass_t0']['prose/c1'])} | {(x['ar_cells_delta'] or {}).get('swaps', '-')} | "
                 f"{x['errcount_server']} | {sum(vv or 0 for vv in x['foreign'].values()) if x['foreign'] else '-'} |")
    an = v["annotations"]; t = an["tokens_per_verify_step_t1"]
    L += ["", f"HEADLINE (annotation, not a gate) — PB > P > A at temp 1.0? {an['headline_PB_P_A']}",
          "ANNOTATION — accepted tokens per verify step at temp 1.0 (1 + accepted/drafts from /metrics):",
          "  pct1 window: " + ", ".join(f"{a}={f2(t['pct1_window'][a], 3)}" for a in ARMS)
          + f"  P/ctrl={f2(t['P_over_ctrl_pct1'], 3)} PB/ctrl={f2(t['PB_over_ctrl_pct1'], 3)} PB/P={f2(t['PB_over_P_pct1'], 3)}",
          "  mixed C1: " + ", ".join(f"{a}={f2(t['mixed_c1'][a], 3)}" for a in ARMS)
          + f"  P/ctrl={f2(t['P_over_ctrl_mixed'], 3)} PB/ctrl={f2(t['PB_over_ctrl_mixed'], 3)} PB/P={f2(t['PB_over_P_mixed'], 3)}",
          "  per class tokens/step (P / PB / ctrl): " + "; ".join(f"{k} {f2(x['P'], 2)}/{f2(x['PB'], 2)}/{f2(x['ctrl'], 2)}"
                                                       for k, x in an["tokens_per_verify_step_t1_per_class"].items()),
          "  per class tok/s ratios (P/ctrl, PB/ctrl, PB/P): " + "; ".join(
              f"{k} {f2(x['P/ctrl'], 3)},{f2(x['PB/ctrl'], 3)},{f2(x['PB/P'], 3)}" for k, x in an["tok_s_ratio_t1_per_class"].items()),
          f"ANNOTATION — per-position acceptance (pct1 window, accepted_pos/drafts): {an['accept_per_position_pct1']}",
          f"ANNOTATION — TF P vs PB: {an['tf_P_vs_PB']}",
          "ANNOTATION — x3 acceptance canary (agent_fixture_x3, temp 0, inside each window): " + "; ".join(
              f"{w}/{cl} " + "/".join(f2(an['x3_accept_canary'][w][cl][a], 3) for a in ARMS)
              for w in ("t1",) for cl in CLASSES) + " (A/P/PB/A2)",
          f"ANNOTATION — G2 X/mean(A,A2) by cell: {v['calc'].get('G2_ratio_by_cell')}",
          f"ANNOTATION — KV: {v['calc'].get('kv')}; host_avail at rehome layer 9 (GiB): {an['host_avail_l9_gib']}",
          f"ANNOTATION — adaptive residency during cells (/ar line deltas pct1-start -> turns-end): {an['adaptive_ar_lines_cells_delta']}",
          f"ANNOTATION — production (v27) watchdog after restore: {an['restore_watchdog']}"]
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
    return 0 if v["result"] == "PASS" else (3 if v["result"].startswith("INVALID") else 1)


if __name__ == "__main__":
    sys.exit(main())
