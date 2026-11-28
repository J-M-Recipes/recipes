#!/usr/bin/env python3
"""f3_verdict.py <results dir> — Card F3 verdict: who wins one-user (C1) decode, Pg67c or Flare's current gb300 core.

Pre-registered rule (CARD-F3-SPEC.md, written before any cell ran):
  primary   = llm-inference-bench 0.6.2 C1 ctx0 aggregate tok/s, median of the valid runs per arm (>= 2 needed)
  secondary = per-class C1 decode tok/s (T=1, thinking off, 256 forced tokens), mean of seeds 42/43/44 (>= 2 needed)
  an instrument is a WIN for a side only when it leads by >= 3 %; otherwise that instrument is a TIE
  winner    = the side with at least one WIN and no loss; anything else is SPLIT (incumbent Pg67c is restored)
Prints one line `WINNER=<flare|pg67c|split|invalid>` last; writes F3-VERDICT.md + F3-verdict.json in the results dir.
"""
import json, os, statistics as st, sys

R = sys.argv[1]
CLS = ["prose", "structured", "code", "shell_ops", "tool_json"]
MARGIN = 0.03

def J(p):
    try:
        return json.load(open(os.path.join(R, p)))
    except Exception:
        return None

def lib_cell(j, ctx, conc):
    try:
        v = j["summary_table"][str(ctx)][str(conc)]
        return float(v) if v is not None and float(v) > 0 else None
    except Exception:
        return None

def arm(a):
    r = {"lib_c1_runs": [], "pc_seeds": []}
    for f in ("lib.json", "libc1-b.json", "libc1-c.json"):
        v = lib_cell(J(f"{a}/{f}") or {}, 0, 1)
        if v is not None:
            r["lib_c1_runs"].append(round(v, 1))
    full = J(f"{a}/lib.json") or {}
    r["lib_table"] = {f"C{c} {('0' if x == 0 else '16K')}": lib_cell(full, x, c) for x in (0, 16384) for c in (1, 8, 32)}
    for s in (42, 43, 44):
        j = J(f"{a}/pc/perclass-F3-{a}-s{s}-c1.json")
        if j:
            try:
                r["pc_seeds"].append({c: j["res"][c]["decode_tok_s_pooled"] for c in CLS})
            except Exception:
                pass
    j = J(f"{a}/pc/perclass-F3-{a}-nat-c1-nat.json")
    if j:
        try:
            r["pc_nat"] = round(st.mean(j["res"][c]["decode_tok_s_pooled"] for c in CLS), 1)
        except Exception:
            pass
    r["lib_c1"] = round(st.median(r["lib_c1_runs"]), 1) if len(r["lib_c1_runs"]) >= 2 else None
    if len(r["pc_seeds"]) >= 2:
        r["pc_cls"] = {c: round(st.mean(p[c] for p in r["pc_seeds"]), 1) for c in CLS}
        r["pc"] = round(st.mean(r["pc_cls"].values()), 1)
    else:
        r["pc"] = None
    r["pc_n"] = len(r.pop("pc_seeds"))
    return r

def judge(f, p):
    if f is None or p is None:
        return "missing"
    if f >= p * (1 + MARGIN):
        return "flare"
    if p >= f * (1 + MARGIN):
        return "pg67c"
    return "tie"

P, Fl = arm("P1"), arm("F26")
out = {"P1": P, "F26": Fl, "margin": MARGIN}
if os.path.isdir(os.path.join(R, "P3")):
    out["P3"] = arm("P3")
j1, j2 = judge(Fl["lib_c1"], P["lib_c1"]), judge(Fl["pc"], P["pc"])
out["primary"], out["secondary"] = j1, j2
if "missing" in (j1, j2):
    w = "invalid"
else:
    wins = {j1, j2} - {"tie"}
    w = wins.pop() if len(wins) == 1 else "split"
out["winner"] = w
json.dump(out, open(os.path.join(R, "F3-verdict.json"), "w"), indent=1)

def f(v, n=1):
    return "—" if v is None else f"{v:,.{n}f}"

L = ["# Card F3 verdict: one-user decode, Pg67c vs Flare gb300 core d683cc1a (runtime 7db06e0c)", "",
     f"**Winner: {w}** (primary {j1}, secondary {j2}; a lead under {int(MARGIN*100)} % counts as a tie)", "",
     "| instrument | Pg67c (GB300 + RTX sidecar) | Flare (one GB300) | Flare / Pg67c |", "|---|--:|--:|--:|",
     f"| llm-inference-bench C1 ctx0, median of {len(P['lib_c1_runs'])}/{len(Fl['lib_c1_runs'])} runs | {f(P['lib_c1'])} ({P['lib_c1_runs']}) | {f(Fl['lib_c1'])} ({Fl['lib_c1_runs']}) | {f(Fl['lib_c1']/P['lib_c1'], 2) if Fl['lib_c1'] and P['lib_c1'] else '—'} |",
     f"| per-class C1, mean of {P['pc_n']}/{Fl['pc_n']} seeds | {f(P['pc'])} | {f(Fl['pc'])} | {f(Fl['pc']/P['pc'], 2) if Fl['pc'] and P['pc'] else '—'} |",
     f"| per-class C1, natural length | {f(P.get('pc_nat'))} | {f(Fl.get('pc_nat'))} | |", "",
     "## llm-inference-bench full table (one run per arm; aggregate tok/s)", "", "| arm | " + " | ".join(P["lib_table"].keys()) + " |", "|---|" + "--:|" * 6]
for lab, r in (("Pg67c", P), ("Flare", Fl)):
    L.append(f"| {lab} | " + " | ".join(f(v, 0) for v in r["lib_table"].values()) + " |")
if P.get("pc_cls") and Fl.get("pc_cls"):
    L += ["", "## Per-class C1 by class", "", "| arm | " + " | ".join(CLS) + " |", "|---|" + "--:|" * len(CLS)]
    for lab, r in (("Pg67c", P), ("Flare", Fl)):
        L.append(f"| {lab} | " + " | ".join(f(r["pc_cls"][c]) for c in CLS) + " |")
open(os.path.join(R, "F3-VERDICT.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
print(f"WINNER={w}")
