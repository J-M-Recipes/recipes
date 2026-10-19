#!/usr/bin/env python3
"""flips_margin_p2.py — margin-weighted teacher-forced flip gate for Card P2 (generalises results-cardR flips_split.py).

Usage: flips_margin_p2.py REF.jsonl CAND.jsonl [--label X] [--calib R2B.jsonl]
  REF/CAND = tf_logprob.py capture files (fields: id, tokens, logprobs, top1).
Prints per-class bins (ref-confident p>=0.9 / mid 0.5-0.9 / low) and a margin-weighted score:
  W = 1000 * sum_{flips where ref top1 == actual token} p_ref(token) / positions
  (a flip at a position the reference was sure about costs ~1; a near-tie flip costs ~0.5 or less;
   flips where the reference itself was wrong cost 0 — they are reordering noise, not regressions).
GATE line (machine-readable, last line):  GATE <PASS|FAIL> hi_agent_tool=<n> hi_prose_pct=<x> W=<w> W_calib=<w0>
  PASS iff  ref-confident flips on agent+tool docs == 0
        AND ref-confident prose flip rate <= 0.05 %          (R1/R2b on 0.31.0: 0.01 %)
        AND W <= 1.25 * W_calib   (W_calib = same score for --calib file vs REF; default = Card R2b value below)
Raw flip % is printed for the record only and is NOT part of the gate (v19 lesson, SKILL.md 'Flip-rate fidelity gate').
NOTE: prompt_logprobs requests keep all rows in the #58132 replay path (deepseek_v41/nvidia/model_state.py:281-283),
so this gate measures the non-replay forward. Replay numerics are covered by BFCL/gauntlet/acceptance, not by TF.
"""
import json, math, sys, collections, argparse

W_CALIB_DEFAULT = 3.077  # Card R2b (0.31.0 + hook, off58) vs tf-R-ref, measured 2026-10-18 with this script; R1 = 3.219
HI, MID = math.log(0.9), math.log(0.5)

def load(p):
    return {r["id"]: r for r in map(json.loads, open(p))}

def cls(i):
    i = i.lower()
    if "tool" in i: return "tool"
    if "agent" in i: return "agent"
    if "code" in i or "shell" in i: return "code/shell"
    return "prose"

def score(ref, c):
    agg = collections.defaultdict(collections.Counter)
    wsum = 0.0; npos = 0; skipped = 0
    for i, r in ref.items():
        if i not in c or c[i]["tokens"] != r["tokens"]:
            skipped += 1; continue
        k = cls(i)
        for tok, lp, t1r, t1c in zip(r["tokens"], r["logprobs"], r["top1"], c[i]["top1"]):
            npos += 1
            conf = "hi" if (t1r == tok and lp >= HI) else ("mid" if (t1r == tok and lp >= MID) else "lo")
            agg[k][("n", conf)] += 1
            if t1r != t1c:
                agg[k][("flip", conf)] += 1
                if t1r == tok:
                    wsum += math.exp(lp)
    return agg, (1000.0 * wsum / npos if npos else float("nan")), npos, skipped

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref"); ap.add_argument("cand")
    ap.add_argument("--label", default="CAND"); ap.add_argument("--calib")
    a = ap.parse_args()
    ref = load(a.ref)
    agg, W, npos, skipped = score(ref, load(a.cand))
    Wc = score(ref, load(a.calib))[1] if a.calib else W_CALIB_DEFAULT
    print(f"== {a.label}: {a.cand} vs {a.ref} == positions={npos} skipped_docs={skipped}")
    tot_n = tot_f = 0
    for k in sorted(agg):
        g = agg[k]
        n = sum(v for (kind, _), v in g.items() if kind == "n"); f = sum(v for (kind, _), v in g.items() if kind == "flip")
        tot_n += n; tot_f += f
        print(f"{k:11s} n={n:6d} raw_flips={f:5d} ({100*f/max(n,1):.2f}%) | ref-confident n={g[('n','hi')]:6d} flips={g[('flip','hi')]:4d} "
              f"({100*g[('flip','hi')]/max(g[('n','hi')],1):.3f}%) | mid flips={g[('flip','mid')]}/{g[('n','mid')]} | low flips={g[('flip','lo')]}/{g[('n','lo')]}")
    print(f"all         n={tot_n} raw_flips={tot_f} ({100*tot_f/max(tot_n,1):.2f}%)  [record only — not gated]")
    hi_at = sum(agg[k][("flip", "hi")] for k in ("agent", "tool"))
    hp_n = agg["prose"][("n", "hi")]; hp_pct = 100 * agg["prose"][("flip", "hi")] / max(hp_n, 1)
    if Wc is None:
        Wc = W  # no calibration given: report only
        ok_w = True
    else:
        ok_w = W <= 1.25 * Wc
    ok = hi_at == 0 and hp_pct <= 0.05 and ok_w and skipped == 0
    print(f"margin-weighted W={W:.3f} per 1k positions (calib {Wc:.3f}; bar <= {1.25*Wc:.3f})")
    print(f"GATE {'PASS' if ok else 'FAIL'} hi_agent_tool={hi_at} hi_prose_pct={hp_pct:.3f} W={W:.3f} W_calib={Wc:.3f} skipped_docs={skipped}")

if __name__ == "__main__":
    main()
