#!/usr/bin/env python3
"""cardP2f_verdict.py — Card P2f promotion-gate verdict (pre-registered in cardP2f_runner.sh header).
Usage: python3 cardP2f_verdict.py <R> <PC_BASE> <PC_C> <PC_D>   -> markdown on stdout, verdict.json in R.
Stdlib only. A missing measurement is reported as MISSING and fails its gate (never defaulted to a number).
Armed 2026-10-18 by Milo (hermes milo) subagent claude-opus-5-5.
"""
import json, os, re, sys

R, PCB, PCC, PCD = sys.argv[1:5]
ORDER = ["prose", "structured", "code", "shell_ops", "tool_json"]
K = 5  # num_speculative_tokens; k-schedule [[1,8,5],[9,24,1]] keeps k=5 at C1 and C8


def jl(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def f(x, nd=1):
    return "MISSING" if x is None else f"{x:.{nd}f}"


def pc(x):
    return "—" if x is None else f"{100 * x:+.1f}%"


L = []
gates = {}  # name -> (PASS/FAIL/MISSING, text)

# ---------------- validity ----------------
facts = open(f"{R}/facts-C.txt").read() if os.path.exists(f"{R}/facts-C.txt") else ""
runner = open(f"{R}/runner.log").read() if os.path.exists(f"{R}/runner.log") else ""
valid = []
valid.append(("rehome done layers=40", "rehome done layers=40" in facts))
valid.append(("no 'Decoder SWA bounded replay:' line", bool(facts) and "Decoder SWA bounded replay:" not in facts))
valid.append(("MegaAttn backend line", bool(re.search(r"MEGA_ATTN|MegaAttn", facts, re.I))))
for d, nm in ((PCB, "v23"), (PCC, "candC"), (PCD, "v23post")):
    try:
        lg = open(f"{d}/runner.log").read(); valid.append((f"prompts sha1 618898aa ({nm})", "sha1=618898aa" in lg))
    except Exception:
        valid.append((f"prompts sha1 618898aa ({nm})", False))
is_valid = all(v for _, v in valid)

# ---------------- per-class ----------------
def summ(d, arm):
    j = jl(f"{d}/perclass-table-{arm}.json")
    return j["summary"] if j else None

SB, SC, SD = summ(PCB, "v23"), summ(PCC, "candC"), summ(PCD, "v23post")
pcrows = []; pc_fail = []; pc_missing = False; trig = []
for c in ORDER:
    for m in ("c1", "c8"):
        cand = SC[c][m]["mean"] if SC and SC.get(c) else None
        row = {"class": c, "cell": m, "cand": cand}
        for nm, S in (("v23", SB), ("v23post", SD)):
            cm = S[c][m]["mean"] if S and S.get(c) else None
            sp = S[c][m]["spread"] if S and S.get(c) else None
            if cm is None or sp is None or cand is None:
                row[nm] = None; pc_missing = True; continue
            bar = cm * (1 - max(0.02, 1.5 * sp)); ok = cand >= bar; d = cand / cm - 1
            row[nm] = {"mean": cm, "spread": sp, "bar": bar, "delta": d, "ok": ok}
            if not ok: pc_fail.append((c, m, nm, d, bar, cm))
            ca = mean(SC[c][m]["accept"]); ka = mean(S[c][m]["accept"])
            if ca is not None and ka is not None:
                row[nm]["acc_c"] = ca; row[nm]["acc_k"] = ka
                if abs(ca - ka) > 0.02: trig.append(f"{c}/{m} vs {nm}: {100*ca:.1f} vs {100*ka:.1f} ({100*(ca-ka):+.1f} pts)")
        pcrows.append(row)

L.append("### Per-class decode wall-clock (GATE: every class x {C1,C8} >= bar vs BOTH controls; bar = ctl mean x (1 − max(2%, 1.5 x ctl spread)))")
L.append("")
L.append("| class | cell | candC mean | v23 (18:29) mean / spread | bar | Δ | verdict | v23post mean / spread | bar | Δ | verdict |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|")
for r in pcrows:
    cells = [r["class"], r["cell"], f(r["cand"])]
    for nm in ("v23", "v23post"):
        x = r.get(nm)
        if not x: cells += ["MISSING", "—", "—", "MISSING"]; continue
        cells += [f"{x['mean']:.1f} / {100*x['spread']:.1f}%", f"{x['bar']:.1f}", pc(x["delta"]), "PASS" if x["ok"] else "**FAIL**"]
    L.append("| " + " | ".join(cells) + " |")
n_ok = sum(1 for r in pcrows for nm in ("v23", "v23post") if r.get(nm) and r[nm]["ok"])
gates["perclass"] = ("MISSING" if pc_missing else ("PASS" if not pc_fail else "FAIL"), f"{n_ok}/20 comparisons pass")
L.append("")
L.append(f"Per-class gate: **{gates['perclass'][0]}** ({gates['perclass'][1]}).")

# acceptance table + arithmetic
L.append("")
L.append("### Per-class acceptance (/metrics, mean of r1/r2) and the acceptance-tax vs kernel-gain split")
L.append("tokens/step ≈ 1 + 5a (k=5 at C1 and C8). step-speed ratio = decode ratio ÷ tokens/step ratio (>1 = the engine's step is faster at equal acceptance).")
L.append("")
L.append("| class | cell | accept candC | v23 | v23post | Δ pts vs v23 / v23post | decode ratio vs mean(v23, v23post) | tokens/step ratio | step-speed ratio (kernel gain) |")
L.append("|---|---|---|---|---|---|---|---|---|")
arith = {}
for r in pcrows:
    c, m = r["class"], r["cell"]
    a, b = r.get("v23"), r.get("v23post")
    if not (a and b and "acc_c" in a and "acc_c" in b and r["cand"]):
        L.append(f"| {c} | {m} | MISSING | | | | | | |"); continue
    ac = a["acc_c"]; ka = mean([a["acc_k"], b["acc_k"]]); km = mean([a["mean"], b["mean"]])
    dr = r["cand"] / km; tr = (1 + K * ac) / (1 + K * ka); sr = dr / tr
    arith[f"{c}/{m}"] = {"decode_ratio": dr, "tps_ratio": tr, "step_ratio": sr}
    L.append(f"| {c} | {m} | {100*ac:.1f} | {100*a['acc_k']:.1f} | {100*b['acc_k']:.1f} | {100*(ac-a['acc_k']):+.1f} / {100*(ac-b['acc_k']):+.1f} | {dr:.3f} | {tr:.3f} | {sr:.3f} |")
L.append("")
L.append("TRIGGER (report, not gate) — acceptance moves > 2 pts: " + ("; ".join(trig) if trig else "none"))

# drift
drift = []
if SB and SD:
    for c in ORDER:
        for m in ("c1", "c8"):
            a, b = SB[c][m], SD[c][m]
            if a["mean"] and b["mean"] and a["spread"] is not None and b["spread"] is not None:
                d = b["mean"] / a["mean"] - 1
                if abs(d) > max(a["spread"], b["spread"]): drift.append(f"{c}/{m} {100*d:+.1f}% (spreads {100*a['spread']:.1f}/{100*b['spread']:.1f}%)")
L.append("")
L.append("Window drift (v23post vs v23 18:29, |Δ| > max of the two same-config spreads): " + ("; ".join(drift) if drift else "none"))

# ---------------- fidelity ----------------
def gsm(tag):
    j = jl(f"{R}/gsm8k200-{tag}.json")
    if not j: return None
    a = j["accuracy"]; c = j.get("comparison") or {}
    return {"acc": a["accuracy"] * 100, "correct": a["correct"], "n": a["scored"], "errors": a["errors"], "p": c.get("mcnemar_exact_p"),
            "dpp": c.get("delta_pp"), "ctok": (c.get("completion_tokens") or {}).get("candidate_mean")}

def bf(tag, suite):
    j = jl(f"{R}/bfcl-{tag}-{suite}-summary.json")
    return None if not j else {"acc": j["all"]["acc"] * 100, "ok": j["all"]["ok"], "n": j["all"]["n"], "err": j.get("errors")}

def gt(tag):
    j = jl(f"{R}/gauntlet-{tag}/summary.json")
    return None if not j else (j.get("success_count"), j.get("task_count"))

G = {t: gsm(t) for t in "ACD"}; BD = {t: bf(t, "dev") for t in "ACD"}; BH = {t: bf(t, "heldout") for t in "ACD"}; GT = {t: gt(t) for t in "ACD"}
if G["A"] and G["C"]:
    d = G["C"]["acc"] - G["A"]["acc"]; gates["gsm8k"] = ("PASS" if d >= -1.5 else "FAIL", f"C {G['C']['acc']:.1f} vs A {G['A']['acc']:.1f} ({d:+.1f} pt; McNemar p={G['C']['p']}) bar ≥ −1.5")
else: gates["gsm8k"] = ("MISSING", "GSM8K-200 missing on A or C")
for nm, B in (("bfcl_dev", BD), ("bfcl_heldout", BH)):
    if B["A"] and B["C"]:
        d = B["C"]["acc"] - B["A"]["acc"]; gates[nm] = ("PASS" if d >= -1.0 else "FAIL", f"C {B['C']['acc']:.2f} vs A {B['A']['acc']:.2f} ({d:+.2f} pt) bar ≥ −1")
    else: gates[nm] = ("MISSING", f"{nm} missing on A or C")
gates["gauntlet"] = ("MISSING", "missing") if not GT["C"] else (("PASS" if GT["C"][0] == GT["C"][1] == 12 else "FAIL"), f"C {GT['C'][0]}/{GT['C'][1]}")

def fx(x, k, nd=1, suf=""):
    return "—" if not x else f"{x[k]:.{nd}f}{suf}"
L.append("")
L.append("### Fidelity (control = A, same window; D is report-only)")
L.append("")
L.append("| cell | A v23 live | C config C | D v23 restart | C − A | bar | verdict |")
L.append("|---|---|---|---|---|---|---|")
def frow(name, X, key, nd, gate):
    a, c, d = X["A"], X["C"], X["D"]
    diff = f"{c[key]-a[key]:+.{nd}f}" if a and c else "—"
    extra = lambda x: "" if not x else (f" ({x['correct']}/{x['n']})" if 'correct' in x else f" ({x['ok']}/{x['n']})")
    L.append(f"| {name} | {fx(a,key,nd)}{extra(a)} | {fx(c,key,nd)}{extra(c)} | {fx(d,key,nd)}{extra(d)} | {diff} | {gate[1].split('bar ')[-1] if 'bar' in gate[1] else ''} | {gate[0]} |")
frow("GSM8K-200 accuracy %", G, "acc", 1, gates["gsm8k"])
frow("BFCL dev (simple_python+multiple, 600) %", BD, "acc", 2, gates["bfcl_dev"])
frow("BFCL held-out (live_simple+live_multiple) %", BH, "acc", 2, gates["bfcl_heldout"])
gs = lambda t: "—" if not GT[t] else f"{GT[t][0]}/{GT[t][1]}"
L.append(f"| gauntlet | {gs('A')} | {gs('C')} | {gs('D')} | | 12/12 | {gates['gauntlet'][0]} |")
if G["C"] and G["A"]:
    L.append(f"\nGSM8K: C vs A McNemar p = {G['C']['p']}, completion tokens mean C {G['C']['ctok']} (A vs Card D baseline: Δ {G['A']['dpp']} pp, p={G['A']['p']}).")
if G["D"]:
    L.append(f"GSM8K D vs A (same image, two boots): Δ {G['D']['dpp']} pp, p={G['D']['p']} — the same-config noise floor for this instrument.")

# ---------------- speed ----------------
def rt(tag):
    v = []; t = []; a = []
    for r in (1, 2, 3):
        j = jl(f"{R}/rt-{tag}/c1-r{r}.json")
        if j: s = j["summary"]; v.append(s["decode_agg"]); t.append(s["median_ttft_s"]); a.append(s.get("acceptance_len"))
    return None if len(v) < 3 else {"agg": mean(v), "runs": v, "ttft": mean(t), "acc": mean(a)}

def knee6(tag):
    reps = {}
    for k in (1, 2):
        try: txt = open(f"{R}/knee6-{tag}-r{k}.txt").read()
        except Exception: continue
        for mm in re.finditer(r"conc=\s*(\d+):\s*([\d.]+) agg.*?\(runs ([\d.]+)/([\d.]+)\)", txt):
            reps.setdefault(int(mm.group(1)), []).extend([float(mm.group(3)), float(mm.group(4))])
    return reps

def knee(tag, r):
    j = jl(f"{R}/knee-P2f-{tag}-{r}.json")
    if not j: return None
    d = {}
    for x in j["rows"]:
        c, a = (x["conc"], x.get("agg")) if isinstance(x, dict) else (x[0], x[1]); d[int(c)] = a
    return d

def cold(tag):
    try: txt = open(f"{R}/cold-{tag}.txt").read()
    except Exception: return None
    mm = re.search(r"mean_tok_s=([\d.]+)", txt); tt = re.search(r"ttft_s=\[([\d.]+)", txt)
    return None if not mm else (float(mm.group(1)), float(tt.group(1)) if tt else None)

RT = {t: rt(t) for t in "ACD"}; K6 = {t: knee6(t) for t in "ACD"}; KN = {t: knee(t, "r2") for t in "ACD"}; CO = {t: cold(t) for t in "ACD"}
if RT["A"] and RT["C"] and RT["D"]:
    ctl = mean([RT["A"]["agg"], RT["D"]["agg"]]); d = RT["C"]["agg"] / ctl - 1
    gates["rt_c1"] = ("PASS" if d >= -0.02 else "FAIL", f"C {RT['C']['agg']:.1f} vs CTL {ctl:.1f} ({100*d:+.1f}%) bar ≥ −2%")
else: gates["rt_c1"] = ("MISSING", "RT C1 missing on an arm")
k8 = {t: (mean(K6[t].get(8, [])) if len(K6[t].get(8, [])) == 4 else None) for t in "ACD"}
if all(k8[t] is not None for t in "ACD"):
    ctl = mean([k8["A"], k8["D"]]); d = k8["C"] / ctl - 1
    gates["knee6_c8"] = ("PASS" if d >= -0.02 else "FAIL", f"C {k8['C']:.1f} vs CTL {ctl:.1f} ({100*d:+.1f}%) bar ≥ −2% (4 reps/arm)")
else: gates["knee6_c8"] = ("MISSING", "knee6 C8 needs 4 reps on A, C and D")

kv = re.search(r"GPU KV cache size: ([\d,]+) tokens", facts)
bind = re.search(r"boot->ready C (\d+)s", runner); bindD = re.search(r"boot->ready D (\d+)s", runner)
L.append("")
L.append("### Speed (CTL = mean of A and D)")
L.append("")
L.append("| cell | A v23 live | C config C | D v23 restart | C vs CTL | gate |")
L.append("|---|---|---|---|---|---|")
def srow(name, va, vc, vd, nd=1, gate=""):
    ctl = mean([va, vd]) if va is not None and vd is not None else None
    dd = pc(vc / ctl - 1) if (ctl and vc is not None) else "—"
    L.append(f"| {name} | {f(va,nd) if va is not None else '—'} | {f(vc,nd) if vc is not None else '—'} | {f(vd,nd) if vd is not None else '—'} | {dd} | {gate} |")
g_ = lambda t, k: RT[t][k] if RT[t] else None
srow("RT C1 decode_agg (mean of 3)", g_("A", "agg"), g_("C", "agg"), g_("D", "agg"), 1, gates["rt_c1"][0])
srow("RT C1 median TTFT s (≈ TTFT 8K)", g_("A", "ttft"), g_("C", "ttft"), g_("D", "ttft"), 3)
srow("RT C1 acceptance_len", g_("A", "acc"), g_("C", "acc"), g_("D", "acc"), 3)
for cc in (1, 6, 8):
    v = {t: (mean(K6[t].get(cc, [])) if K6[t].get(cc) else None) for t in "ACD"}
    srow(f"knee6 C{cc} (mean of {len(K6['C'].get(cc, []))} reps)", v["A"], v["C"], v["D"], 1, gates["knee6_c8"][0] if cc == 8 else "")
L.append(f"| knee6 C8 raw reps | {K6['A'].get(8)} | {K6['C'].get(8)} | {K6['D'].get(8)} | | |")
for cc in (1, 8, 16):
    v = {t: (KN[t].get(cc) if KN[t] else None) for t in "ACD"}
    srow(f"knee r2 C{cc}", v["A"], v["C"], v["D"])
cv = {t: (CO[t][0] if CO[t] else None) for t in "ACD"}
srow("cold 64K prefill tok/s (N=1)", cv["A"], cv["C"], cv["D"], 0)
L.append(f"| KV tokens | 2,504,891 (v23) | {kv.group(1) if kv else '—'} | | | |")
L.append(f"| bind s | live | {bind.group(1) if bind else '—'} | {bindD.group(1) if bindD else '—'} | | |")

# ---------------- decision ----------------
others = [k for k in gates if k != "perclass"]
others_ok = all(gates[k][0] == "PASS" for k in others)
missing = [k for k in gates if gates[k][0] == "MISSING"]
code_only = bool(pc_fail) and all(c == "code" and d >= -0.05 for c, m, nm, d, bar, cm in pc_fail)
if not is_valid:
    res, why = "FAIL", "INVALID run (mechanism/validity check failed: " + ", ".join(n for n, v in valid if not v) + ")"
elif missing:
    res, why = "FAIL", "INCOMPLETE — missing: " + ", ".join(missing)
elif gates["perclass"][0] == "PASS" and others_ok:
    res, why = "PASS", "every pre-registered gate passed"
elif others_ok and code_only:
    res, why = "JUDGMENT_CALL", "only code per-class cells miss their bar, all within −5% of control — judgment call for James"
else:
    fails = [f"{k}: {gates[k][1]}" for k in gates if gates[k][0] == "FAIL" and k != "perclass"]
    fails += [f"perclass {c}/{m} vs {nm}: {100*d:+.1f}% (bar {bar:.1f}, ctl {cm:.1f})" for c, m, nm, d, bar, cm in pc_fail]
    res, why = "FAIL", "; ".join(fails)
suffix = {"PASS": "P2f-PASS", "FAIL": "P2f-FAIL", "JUDGMENT_CALL": "P2f-JUDGMENT"}[res]

H = [f"**Verdict: {res}** — {why}. NOT promoted (James decides).", f"GATE_RESULT {res}", f"SUFFIX {suffix}", ""]
H.append("### Gate summary")
H.append("")
H.append("| gate | result | detail |"); H.append("|---|---|---|")
for k in ("perclass", "gsm8k", "bfcl_dev", "bfcl_heldout", "gauntlet", "rt_c1", "knee6_c8"):
    H.append(f"| {k} | {gates[k][0]} | {gates[k][1]} |")
H.append("")
H.append("Validity: " + "; ".join(f"{n} {'OK' if v else 'FAIL'}" for n, v in valid))
if pc_fail:
    H.append("")
    H.append("Failing per-class comparisons: " + "; ".join(f"{c}/{m} vs {nm} {100*d:+.1f}% (cand {cm*(1+d):.1f}, bar {bar:.1f}, ctl {cm:.1f}, short of bar by {100*(cm*(1+d)/bar-1):+.1f}%)" for c, m, nm, d, bar, cm in pc_fail))
if res == "JUDGMENT_CALL" or any(c == "code" for c, *_ in pc_fail):
    H.append("")
    H.append("Code arithmetic (acceptance tax vs kernel gain): " + "; ".join(
        f"{k}: decode ×{v['decode_ratio']:.3f} = tokens/step ×{v['tps_ratio']:.3f} (acceptance tax) × step speed ×{v['step_ratio']:.3f} (kernel gain)"
        for k, v in arith.items() if k.startswith("code/")))
print("\n".join(H + [""] + L))
json.dump({"result": res, "why": why, "suffix": suffix, "gates": gates, "valid": valid, "pc_fail": pc_fail, "trigger": trig, "drift": drift,
           "arith": arith, "gsm8k": G, "bfcl_dev": BD, "bfcl_heldout": BH, "gauntlet": GT, "rt": RT, "knee6": K6, "knee_r2": KN, "cold": CO},
          open(f"{R}/verdict.json", "w"), indent=1, default=str)
