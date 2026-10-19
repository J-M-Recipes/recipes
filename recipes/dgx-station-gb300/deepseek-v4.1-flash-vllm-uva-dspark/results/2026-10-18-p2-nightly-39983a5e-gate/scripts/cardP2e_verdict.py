#!/usr/bin/env python3
"""cardP2e_verdict.py <results-dir> — Card P2e verdict (bars pre-registered in cardP2e_runner.sh header).
CTL = mean of A (live v23) and D (v23 restart).
RECOVERS iff E fixx3 code AND structured acceptance within 2.0 pts of CTL.
P2f candidate-quality iff RECOVERS AND RT C1 >= CTL-2% AND replay r2 >= CTL-2% AND TTFT 8K <= 0.45 s AND gauntlet 12/12.
Missing inputs are UNKNOWN (never defaulted); an UNKNOWN gated value cannot pass."""
import json, sys, os, re, statistics as st
R = sys.argv[1]; ARMS = ["A", "E", "D"]; CATS = ["prose", "structured", "code", "shell_ops", "tool_json"]
def J(p):
    try: return json.load(open(os.path.join(R, p)))
    except Exception: return None
def T(p):
    p = os.path.join(R, p); return open(p).read() if os.path.exists(p) else ""
def mean(xs):
    xs = [x for x in xs if x is not None]; return st.mean(xs) if xs else None
def f(x, d=1): return "—" if x is None else f"{x:,.{d}f}"
def knee(a, r):
    d = J(f"knee-P2e-{a}-{r}.json")
    if not d: return {}
    out = {}
    for x in d.get("rows", []):
        c, v = (x.get("conc"), x.get("agg")) if isinstance(x, dict) else (x[0], x[1]); out[int(c)] = v
    return out
log = T("runner.log"); A = {}
for a in ARMS:
    x = {}
    rs = [d["summary"] for d in (J(f"rt-{a}/c1-r{r}.json") for r in (1, 2, 3)) if d and "summary" in d]
    x["rt_c1"] = mean([s.get("decode_agg") for s in rs]) if len(rs) == 3 else None
    x["rt_runs"] = [s.get("decode_agg") for s in rs]
    x["ttft"] = mean([s.get("median_ttft_s") for s in rs]) if len(rs) == 3 else None
    x["acc_len"] = mean([s.get("acceptance_len") for s in rs]) if rs else None
    fx = J(f"fixx3-P2e-{a}.json"); x["fx_raw"] = fx
    x["fx"] = {c: (100 * fx["res"][c]["accept"] if fx and c in fx.get("res", {}) else None) for c in CATS}
    rot = [J(f"agentfix-P2e-{a}-rot{k}.json") for k in (0, 1, 2)]
    x["rot"] = {}
    for c in CATS:
        v = [d["res"][c]["accept"] for d in rot if d and c in d.get("res", {})]
        x["rot"][c] = (100 * st.mean(v), 100 * (max(v) - min(v))) if v else (None, None)
    for r in ("r1", "r2"):
        d = J(f"replayc-P2e-{a}-{r}-n4.json"); x[f"rep_{r}"] = d.get("agg_tok_s") if d else None
        x[f"rep_{r}_acc"] = d.get("accept") if d else None
    x["knee_r1"] = knee(a, "r1"); x["knee_r2"] = knee(a, "r2")
    m = re.findall(r"mean_tok_s=(\d+)", T(f"cold-{a}.txt")); x["cold"] = int(m[-1]) if m else None
    m = re.findall(r"ttft_s=\[([\d., ]+)\]", T(f"cold-{a}.txt")); x["cold_ttft"] = m[-1] if m else None
    g = J(f"gauntlet-{a}/summary.json"); x["gaunt"] = (g.get("success_count"), g.get("task_count")) if g else None
    facts = T(f"facts-{a}.txt")
    m = re.search(r"GPU KV cache size: ([\d,]+) tokens", facts); x["kv"] = m.group(1) if m else None
    x["replay_line"] = "Decoder SWA bounded replay:" in facts
    x["rehome"] = "rehome done layers=40" in facts
    x["mega"] = bool(re.search(r"MEGA_ATTN|MegaAttn", facts))
    x["kvfmt"] = sorted(set(re.findall(r"Using DeepSeek's (\S+) KV cache format", facts)))
    m = re.findall(r"Setting kv cache block size to \d+ for (\S+) backend", facts); x["blk"] = m[-1] if m else None
    x["draft_inherit"] = "attention backend for the DeepSeek-V4 DSpark drafter" in facts
    m = re.search(r"TUNEWINDOW (\S+) (\S+)", facts); x["tune"] = f"{m.group(1)}→{m.group(2)}" if m else None
    m = re.findall(r"Preloaded (\d+/\d+)", facts); x["preloaded"] = m[-1] if m else None
    m = re.findall(r"Saved (\d+) configs", facts); x["saved"] = m[-1] if m else None
    m = re.search(r"STORE_ENTRIES (\d+)", facts); x["store"] = m.group(1) if m else None
    m = re.findall(r"env_dir='[^']*/([0-9a-f]{16})'", facts); x["hash"] = m[-1] if m else None
    x["autotune_starts"] = facts.count("Autotuning process starts")
    m = re.findall(r"rehome layer=9 .*?host_avail=([\d.]+)GiB", facts); x["guard9"] = m[-1] if m else None
    m = re.findall(rf"boot->ready {a} (\d+)s", log); x["boot_s"] = int(m[-1]) if m else None
    x["err"] = T(f"errcount-{a}.txt").strip() or None
    A[a] = x
def ctl(key, sub=None):
    v = [(A[a][key] if sub is None else A[a][key].get(sub)) for a in ("A", "D")]
    return None if None in v else st.mean(v)
CT = {"rt_c1": ctl("rt_c1"), "ttft": ctl("ttft"), "rep_r2": ctl("rep_r2"), "rep_r1": ctl("rep_r1"), "cold": ctl("cold"),
      "fx": {c: (None if None in (A["A"]["fx"][c], A["D"]["fx"][c]) else st.mean([A["A"]["fx"][c], A["D"]["fx"][c]])) for c in CATS}}
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
P("# Card P2e VERDICT (pre-registered bars; CTL = mean(A, D))\n")
P("| arm | boot s | KV tok | KV fmt | block-size backends | live tune? (starts / window / saved / store / preloaded) | guard host_avail L9 | RT C1 (runs) | TTFT 8K | RT acc_len | " + " | ".join(CATS) + " |")
P("|---|---|---|---|---|---|---|---|---|---|" + "---|" * len(CATS))
for a in ARMS:
    x = A[a]
    P(f"| {a} | {x['boot_s'] or ('live' if a == 'A' else '—')} | {x['kv'] or '—'} | {','.join(x['kvfmt']) or '—'} | {x['blk'] or '—'} | "
      f"{x['autotune_starts']} / {x['tune'] or '—'} / {x['saved'] or '—'} / {x['store'] or '—'} / {x['preloaded'] or '—'} (key {x['hash'] or '—'}) | {x['guard9'] or '—'} | "
      f"{f(x['rt_c1'])} ({'/'.join(f(v) for v in x['rt_runs'])}) | {f(x['ttft'], 3)} | {f(x['acc_len'], 3)} | " + " | ".join(f(x["fx"][c]) for c in CATS) + " |")
P(f"| CTL | | | | | | | {f(CT['rt_c1'])} | {f(CT['ttft'], 3)} | | " + " | ".join(f(CT['fx'][c]) for c in CATS) + " |\n")
P("| arm | knee r1 C1/C8/C16 | knee r2 C1/C8/C16 | replay r1 (acc) | replay r2 (acc) | cold 64K tok/s (ttft s) | gauntlet | errcount |")
P("|---|---|---|---|---|---|---|---|")
for a in ARMS:
    x = A[a]; k1, k2 = x["knee_r1"], x["knee_r2"]
    P(f"| {a} | {f(k1.get(1))}/{f(k1.get(8))}/{f(k1.get(16))} | {f(k2.get(1))}/{f(k2.get(8))}/{f(k2.get(16))} | {f(x['rep_r1'])} ({x['rep_r1_acc']}) | {f(x['rep_r2'])} ({x['rep_r2_acc']}) | "
      f"{x['cold'] or '—'} ({x['cold_ttft'] or '—'}) | {'/'.join(map(str, x['gaunt'])) if x['gaunt'] else '—'} | {x['err'] or '—'} |")
x = A["E"]
d = {c: (None if x["fx"][c] is None or CT["fx"][c] is None else x["fx"][c] - CT["fx"][c]) for c in CATS}
rtd = None if x["rt_c1"] is None or CT["rt_c1"] is None else 100 * (x["rt_c1"] / CT["rt_c1"] - 1)
r2d = None if x["rep_r2"] is None or CT["rep_r2"] is None else 100 * (x["rep_r2"] / CT["rep_r2"] - 1)
ok_code = d["code"] is not None and abs(d["code"]) <= 2.0
ok_str = d["structured"] is not None and abs(d["structured"]) <= 2.0
ok_rt = rtd is not None and rtd >= -2.0
ok_r2 = r2d is not None and r2d >= -2.0
ok_tt = x["ttft"] is not None and x["ttft"] <= 0.45
ok_g = x["gaunt"] is not None and x["gaunt"][0] == 12 and x["gaunt"][1] == 12
valid = (not x["replay_line"]) and x["rehome"] and not x["mega"] and x["kvfmt"] == ["fp8_ds_mla"]
measured = x["rt_c1"] is not None and x["fx"]["code"] is not None and x["fx"]["structured"] is not None
rec = measured and valid and ok_code and ok_str
cand = rec and ok_rt and ok_r2 and ok_tt and ok_g
ic = ident("E", "code"); isd = ident("E", "structured")
fi = lambda t: "—" if not t or t[1] is None else f"n={t[0]}: {t[1]:.1f} vs CTL {t[2]:.1f}"
P("\nΔ E vs CTL:\n")
P("| code Δ | structured Δ | prose Δ | shell Δ | tool_json Δ | RT C1 Δ% | replay r2 Δ% | TTFT 8K (vs CTL) | gauntlet | ident-output code | ident-output structured |")
P("|---|---|---|---|---|---|---|---|---|---|---|")
P(f"| {f(d['code'])} | {f(d['structured'])} | {f(d['prose'])} | {f(d['shell_ops'])} | {f(d['tool_json'])} | {f(rtd)} | {f(r2d)} | {f(x['ttft'], 3)} ({f(CT['ttft'], 3)}) | "
  f"{'/'.join(map(str, x['gaunt'])) if x['gaunt'] else '—'} | {fi(ic)} | {fi(isd)} |")
P(f"\nBars: RECOVERS = code {'ok' if ok_code else 'x'}, structured {'ok' if ok_str else 'x'} (valid={valid}, measured={measured}); "
  f"candidate-quality adds RT {'ok' if ok_rt else 'x'}, replay r2 {'ok' if ok_r2 else 'x'}, TTFT {'ok' if ok_tt else 'x'}, gauntlet {'ok' if ok_g else 'x'}.")
P("\nMechanism: " + "; ".join(f"{a}: replay_line={A[a]['replay_line']} rehome={A[a]['rehome']} mega_in_facts={A[a]['mega']} kvfmt={A[a]['kvfmt']} drafter_inherit_line={A[a]['draft_inherit']}" for a in ARMS))
P("\nRotation fixture (agent_fixture_o.sh ×3, mean ± spread, %): " + " ; ".join(
    f"{a}: " + ", ".join(f"{c} {f(A[a]['rot'][c][0])}±{f(A[a]['rot'][c][1])}" for c in CATS) for a in ARMS))
if not measured: v, s = "**Verdict: E NOT MEASURED** — see runner.log.", "NOTMEASURED"
elif not valid: v, s = "**Verdict: E INVALID** (mechanism check failed) — see facts-E.txt.", "INVALID"
elif cand: v, s = "**Verdict: E RECOVERS and meets candidate quality — target MegaAttn numerics are the cause; P2f candidate = config C on FLASHMLA_SPARSE_DSV41.**", "RECOVERS-P2f-CAND"
elif rec: v, s = "**Verdict: E RECOVERS acceptance (target MegaAttn numerics are the cause) but misses a candidate-quality bar.**", "RECOVERS-QFAIL"
else: v, s = "**Verdict: E does NOT recover — target MegaAttn is not the cause; config C dead for the agent lane under the 2-pt rule.**", "FAIL"
P("\n" + v)
P(f"SUFFIX_E {s}")
print("\n".join(out))
