#!/usr/bin/env python3
"""cardP2c_verdict.py <results-dir> — Card P2c verdict (bar pre-registered in cardP2c_runner.sh header).
CTL = mean of A (live v23) and D (v23 restart). An arm RECOVERS iff fixx3 code AND structured acceptance within 2.0 pts of CTL
AND RT C1 >= CTL - 2% AND TTFT 8K <= 0.45 s. Missing inputs are UNKNOWN (never defaulted); an UNKNOWN gated value cannot pass."""
import json, sys, os, re, statistics as st
R = sys.argv[1]; ARMS = ["A", "C", "C1", "C2", "D"]; CATS = ["prose", "structured", "code", "shell_ops", "tool_json"]
def J(p):
    try: return json.load(open(os.path.join(R, p)))
    except Exception: return None
def T(p):
    p = os.path.join(R, p); return open(p).read() if os.path.exists(p) else ""
def mean(xs):
    xs = [x for x in xs if x is not None]; return st.mean(xs) if xs else None
def f(x, d=1): return "—" if x is None else f"{x:,.{d}f}"
log = T("runner.log"); A = {}
for a in ARMS:
    x = {}
    rs = [d["summary"] for d in (J(f"rt-{a}/c1-r{r}.json") for r in (1, 2, 3)) if d and "summary" in d]
    x["rt_c1"] = mean([s.get("decode_agg") for s in rs]) if len(rs) == 3 else None
    x["rt_runs"] = [s.get("decode_agg") for s in rs]
    x["ttft"] = mean([s.get("median_ttft_s") for s in rs]) if len(rs) == 3 else None
    x["acc_len"] = mean([s.get("acceptance_len") for s in rs]) if rs else None
    fx = J(f"fixx3-P2c-{a}.json"); x["fx_raw"] = fx
    x["fx"] = {c: (100 * fx["res"][c]["accept"] if fx and c in fx["res"] else None) for c in CATS}
    rot = [J(f"agentfix-P2c-{a}-rot{k}.json") for k in (0, 1, 2)]
    x["rot"] = {}
    for c in CATS:
        v = [d["res"][c]["accept"] for d in rot if d and c in d.get("res", {})]
        x["rot"][c] = (100 * st.mean(v), 100 * (max(v) - min(v))) if v else (None, None)
    facts = T(f"facts-{a}.txt")
    m = re.search(r"GPU KV cache size: ([\d,]+) tokens", facts); x["kv"] = m.group(1) if m else None
    x["replay_line"] = "Decoder SWA bounded replay:" in facts
    x["rehome"] = "rehome done layers=40" in facts
    x["draft_inherit"] = "attention backend for the DeepSeek-V4 DSpark drafter" in facts
    m = re.search(r"TUNEWINDOW (\S+) (\S+)", facts); x["tune"] = f"{m.group(1)}→{m.group(2)}" if m else None
    m = re.findall(r"Preloaded (\d+/\d+)", facts); x["preloaded"] = m[-1] if m else None
    m = re.search(r"STORE_ENTRIES (\d+)", facts); x["store"] = m.group(1) if m else None
    x["autotune_starts"] = facts.count("Autotuning process starts")
    m = re.findall(rf"boot->ready {a} (\d+)s", log); x["boot_s"] = int(m[-1]) if m else None
    g = T(f"gate-{a}.txt"); m = re.search(r"GATE (\w+)", g); x["gate"] = m.group(1) if m else None
    x["err"] = T(f"errcount-{a}.txt").strip() or None
    A[a] = x
def ctl_fx(c):
    v = [A[a]["fx"][c] for a in ("A", "D")]; return None if None in v else st.mean(v)
CT = {"rt_c1": None if None in (A["A"]["rt_c1"], A["D"]["rt_c1"]) else st.mean([A["A"]["rt_c1"], A["D"]["rt_c1"]]),
      "fx": {c: ctl_fx(c) for c in CATS}}
# identical-output subset (prompts whose sha equals BOTH controls)
def ident(a, c):
    try:
        ra = A["A"]["fx_raw"]["res"][c]["rows"]; rd = A["D"]["fx_raw"]["res"][c]["rows"]; rx = A[a]["fx_raw"]["res"][c]["rows"]
    except Exception: return None
    num = den = cn = cd = 0; k = 0
    for i in range(min(len(ra), len(rd), len(rx))):
        if ra[i]["sha"] == rd[i]["sha"] == rx[i]["sha"] and ra[i].get("drafted") and rx[i].get("drafted"):
            k += 1; num += rx[i]["accepted"]; den += rx[i]["drafted"]
            cn += (ra[i]["accepted"] + rd[i]["accepted"]) / 2; cd += (ra[i]["drafted"] + rd[i]["drafted"]) / 2
    return (k, 100 * num / den, 100 * cn / cd) if k and den and cd else (k, None, None)
out = []; P = out.append
P("# Card P2c VERDICT (pre-registered bar; CTL = mean(A, D))\n")
P("| arm | boot s | KV tok | gate | live tune? (starts / store) | RT C1 (runs) | TTFT 8K | RT acc_len | " + " | ".join(CATS) + " |")
P("|---|---|---|---|---|---|---|---|" + "---|" * len(CATS))
for a in ARMS:
    x = A[a]
    P(f"| {a} | {x['boot_s'] or ('live' if a == 'A' else '—')} | {x['kv'] or '—'} | {x['gate'] or ('n/a 0909' if a in 'AD' else '—')} | "
      f"{x['autotune_starts']} / {x['store'] or '—'} ({x['preloaded'] or '—'}) | {f(x['rt_c1'])} ({'/'.join(f(v) for v in x['rt_runs'])}) | {f(x['ttft'], 3)} | {f(x['acc_len'], 3)} | "
      + " | ".join(f(x["fx"][c]) for c in CATS) + " |")
P(f"| CTL | | | | | {f(CT['rt_c1'])} | | | " + " | ".join(f(CT['fx'][c]) for c in CATS) + " |\n")
P("Δ vs CTL (pts / %):\n")
P("| arm | code Δ | structured Δ | tool_json Δ | prose Δ | shell Δ | RT C1 Δ% | TTFT 8K | ident-output code (n, arm vs CTL) | ident-output structured | bar |")
P("|---|---|---|---|---|---|---|---|---|---|---|")
res = {}
for a in ("C", "C1", "C2"):
    x = A[a]
    d = {c: (None if x["fx"][c] is None or CT["fx"][c] is None else x["fx"][c] - CT["fx"][c]) for c in CATS}
    rtd = None if x["rt_c1"] is None or CT["rt_c1"] is None else 100 * (x["rt_c1"] / CT["rt_c1"] - 1)
    ok_code = d["code"] is not None and abs(d["code"]) <= 2.0
    ok_str = d["structured"] is not None and abs(d["structured"]) <= 2.0
    ok_rt = rtd is not None and rtd >= -2.0
    ok_tt = x["ttft"] is not None and x["ttft"] <= 0.45
    valid = (not x["replay_line"]) and x["rehome"] and (a != "C1" or x["gate"] == "False") and (a != "C2" or not x["draft_inherit"])
    measured = x["rt_c1"] is not None and x["fx"]["code"] is not None
    rec = measured and valid and ok_code and ok_str and ok_rt and ok_tt
    res[a] = (rec, measured, valid, d, rtd)
    ic = ident(a, "code"); isd = ident(a, "structured")
    fi = lambda t: "—" if not t or t[1] is None else f"n={t[0]}: {t[1]:.1f} vs {t[2]:.1f}"
    P(f"| {a} | {f(d['code'])} | {f(d['structured'])} | {f(d['tool_json'])} | {f(d['prose'])} | {f(d['shell_ops'])} | {f(rtd)} | {f(x['ttft'], 3)} | {fi(ic)} | {fi(isd)} | "
      + ("RECOVERS" if rec else ("NOT MEASURED" if not measured else ("INVALID" if not valid else "FAIL")))
      + f" (code {'ok' if ok_code else 'x'}, struct {'ok' if ok_str else 'x'}, RT {'ok' if ok_rt else 'x'}, TTFT {'ok' if ok_tt else 'x'}) |")
P("\nMechanism: " + "; ".join(f"{a}: replay_line={A[a]['replay_line']} rehome={A[a]['rehome']} drafter_inherits_target_backend={A[a]['draft_inherit']} gate={A[a]['gate']} errcount={A[a]['err']}" for a in ARMS))
P("\nRotation fixture (agent_fixture_o.sh ×3, mean / spread, %): " + " ; ".join(
    f"{a}: " + ", ".join(f"{c} {f(A[a]['rot'][c][0])}±{f(A[a]['rot'][c][1])}" for c in CATS) for a in ARMS))
recs = [a for a in ("C1", "C2") if res[a][0]]
if recs: v = f"**Verdict: {' and '.join(recs)} RECOVERS** — P2d candidate = {recs[0]}."
else: v = "**Verdict: no override arm recovers** (C1 #56893 gate False, C2 drafter FLASHMLA_SPARSE_DSV41)."
P("\n" + v)
for a in ("C1", "C2"):
    rec, measured, valid, _, _ = res[a]
    P(f"SUFFIX_{a} " + ("RECOVERS" if rec else ("NOTMEASURED" if not measured else ("INVALID" if not valid else "FAIL"))))
print("\n".join(out))
