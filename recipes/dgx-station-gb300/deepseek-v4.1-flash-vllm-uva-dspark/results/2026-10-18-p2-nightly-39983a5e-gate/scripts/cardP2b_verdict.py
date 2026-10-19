#!/usr/bin/env python3
"""cardP2b_verdict.py <results-dir> — Card P2b verdict (bars pre-registered in cardP2b_runner.sh header).
CTL = mean of arms A (live v23) and D (v23 restart). Prints VERDICT.md to stdout with SUFFIX_B / SUFFIX_C / CANDIDATE lines.
Missing inputs are UNKNOWN (never defaulted to a number); an arm with any UNKNOWN gated value cannot pass."""
import json, sys, os, re, statistics as st
R = sys.argv[1]; ARMS = ["A", "B", "C", "D"]; CATS = ["prose", "structured", "code", "shell_ops", "tool_json"]
def J(p):
    try: return json.load(open(os.path.join(R, p)))
    except Exception: return None
def mean(xs):
    xs = [x for x in xs if x is not None]; return st.mean(xs) if xs else None
def f(x, d=1):
    return "—" if x is None else (f"{x:,.{d}f}")
A = {}
log = open(os.path.join(R, "runner.log")).read() if os.path.exists(os.path.join(R, "runner.log")) else ""
for a in ARMS:
    x = {}
    rt = [J(f"rt-{a}/c1-r{r}.json") for r in (1, 2, 3)]
    rs = [d["summary"] for d in rt if d and "summary" in d]
    x["rt_n"] = len(rs)
    x["rt_c1"] = mean([s.get("decode_agg") for s in rs]) if len(rs) == 3 else None
    x["rt_c1_runs"] = [s.get("decode_agg") for s in rs]
    x["ttft8k"] = mean([s.get("median_ttft_s") for s in rs]) if len(rs) == 3 else None
    x["rt_acc"] = mean([s.get("acceptance_len") for s in rs]) if rs else None
    for r in ("r1", "r2"):
        d = J(f"replayc-P2b-{a}-{r}-n4.json")
        x[f"std_{r}"] = d.get("agg_tok_s") if d else None; x[f"std_{r}_tok"] = d.get("tokens") if d else None
        x[f"std_{r}_acc"] = d.get("accept") if d else None
        x[f"std_{r}_tool"] = f"{d['tool']['n']}/{d['tool']['emitted']}" if d else None
        x[f"std_{r}_wall"] = d.get("wall_s") if d else None
        e = J(f"replayfl-P2b-{a}-{r}.json")
        ok = bool(e) and e.get("all_exact_fl") is True and not e.get("errors")
        x[f"fl_{r}"] = e.get("agg_tok_s") if ok else None; x[f"fl_{r}_raw"] = e.get("agg_tok_s") if e else None
        x[f"fl_{r}_exact"] = e.get("all_exact_fl") if e else None
        for k in ("tpot_mean_ms", "tpot_median_ms", "ttft_median_round0_s", "ttft_median_round1_s", "accept", "per_req_tok_s_mean"):
            x[f"fl_{r}_{k}"] = e.get(k) if e else None
    fx = J(f"fixx3-P2b-{a}.json")
    x["fx"] = {c: (fx["res"][c]["accept"] if fx and c in fx["res"] else None) for c in CATS}
    x["fx_pp"] = {c: ((fx["res"][c]["per_prompt_mean"], fx["res"][c]["per_prompt_sd"]) if fx and c in fx["res"] else (None, None)) for c in CATS}
    x["fx_sha"] = {c: ([r["sha"] for r in fx["res"][c]["rows"]] if fx and c in fx["res"] else []) for c in CATS}
    rot = [J(f"agentfix-P2b-{a}-rot{k}.json") for k in (0, 1, 2)]
    x["rot"] = {}
    for c in CATS:
        v = [d["res"][c]["accept"] for d in rot if d and c in d.get("res", {})]
        x["rot"][c] = (st.mean(v), (max(v) - min(v)) if v else None, len(v)) if v else (None, None, 0)
    cold = open(os.path.join(R, f"cold-{a}.txt")).read() if os.path.exists(os.path.join(R, f"cold-{a}.txt")) else ""
    m = re.search(r"PREFILL target=65536 prompt_tokens=(\d+) ttft_s=\[([^\]]*)\] tok_s=\[([^\]]*)\] mean_tok_s=(\d+)", cold)
    x["cold64"] = f"{m.group(2)} s / {int(m.group(4)):,}" if m else None
    facts = open(os.path.join(R, f"facts-{a}.txt")).read() if os.path.exists(os.path.join(R, f"facts-{a}.txt")) else ""
    m = re.search(r"GPU KV cache size: ([\d,]+) tokens", facts); x["kv"] = m.group(1) if m else None
    x["replay_line"] = "Decoder SWA bounded replay:" in facts; x["rehome"] = "rehome done layers=40" in facts
    m = re.search(r"TUNEWINDOW (\S+) (\S+)", facts); x["tune"] = f"{m.group(1)}→{m.group(2)}" if m else None
    m = re.findall(r"Preloaded (\d+/\d+)", facts); x["preloaded"] = m[-1] if m else None
    m = re.search(r"STORE_ENTRIES (\d+)", facts); x["store"] = m.group(1) if m else None
    m = re.findall(rf"boot->ready {a} (\d+)s", log); x["boot_s"] = int(m[-1]) if m else None
    A[a] = x
def ctl(k, sub=None):
    vals = [(A[a][k][sub] if sub else A[a][k]) for a in ("A", "D")]
    return None if any(v is None for v in vals) else st.mean(vals)
CT = {"rt_c1": ctl("rt_c1"), "fl_r2": ctl("fl_r2"), "std_r2": ctl("std_r2"), "fl_r1": ctl("fl_r1"), "std_r1": ctl("std_r1"),
      "ttft8k": ctl("ttft8k"), "std_r2_tok": ctl("std_r2_tok"), "std_r1_tok": ctl("std_r1_tok")}
CT["fx"] = {c: ctl("fx", c) for c in CATS}
def rel(v, c): return None if v is None or c in (None, 0) else (v / c - 1) * 100
def bars(a, need_ttft):
    x = A[a]; out = []
    v = rel(x["fl_r2"], CT["fl_r2"]); out.append(("FL replay r2 ≥ −2%", v is not None and v >= -2.0, f"{f(x['fl_r2'])} vs {f(CT['fl_r2'])} ({f(v)}%)" + ("" if x["fl_r2_exact"] else f" exact={x['fl_r2_exact']}")))
    worst = None; ok = True; parts = []
    for c in CATS:
        if x["fx"][c] is None or CT["fx"][c] is None: ok = False; parts.append(f"{c} UNKNOWN"); continue
        d = (x["fx"][c] - CT["fx"][c]) * 100; parts.append(f"{c} {d:+.1f}")
        if abs(d) > 2.0: ok = False
    out.append(("fixture_x3 per-class acceptance within 2 pts", ok, ", ".join(parts)))
    v = rel(x["rt_c1"], CT["rt_c1"]); out.append(("RT C1 ≥ −2%", v is not None and v >= -2.0, f"{f(x['rt_c1'])} vs {f(CT['rt_c1'])} ({f(v)}%)"))
    if need_ttft: out.append(("TTFT 8K ≤ 0.45 s", x["ttft8k"] is not None and x["ttft8k"] <= 0.45, f"{f(x['ttft8k'], 3)} s (CTL {f(CT['ttft8k'], 3)})"))
    return out
valid = {"B": A["B"]["replay_line"] and A["B"]["rehome"], "C": (not A["C"]["replay_line"]) and A["C"]["rehome"] and A["C"]["rt_c1"] is not None}
bB = bars("B", False); bC = bars("C", True)
passB = valid["B"] and all(b[1] for b in bB); passC = valid["C"] and all(b[1] for b in bC)
cand = "B" if passB else ("C" if passC else "none")
print(f"**Verdict: candidate {cand}** — " + {"B": "B (store-loaded P2 CAND) meets all P2b bars; promotion still needs a confirm boot + BFCL/GSM8K",
      "C": "B fails; C (--no-swa-bounded-replay) meets all P2b bars incl. TTFT 8K; promotion still needs a confirm boot + BFCL/GSM8K",
      "none": "neither B nor C meets the pre-registered P2b bars — P2 closes"}[cand])
print(f"SUFFIX_B {'P2b-CAND' if passB else ('P2b-FAIL' if valid['B'] else 'P2b-INVALID')}")
print(f"SUFFIX_C {'CAND' if cand == 'C' else ('PASSBARS' if passC else ('FAIL' if valid['C'] else 'INVALID'))}")
print(f"CANDIDATE {cand}")
print()
print("Bars (CTL = mean of A and D):")
for name, bb, v in (("B", bB, valid["B"]), ("C", bC, valid["C"])):
    print(f"- {name} (mechanism valid={v}): " + "; ".join(f"{n}: {'PASS' if p else 'MISS'} [{d}]" for n, p, d in bb))
print()
print("| arm | boot s | KV tok | replay line | rehome | tune window | Preloaded | store | RT C1 (runs) | RT acc_len | TTFT 8K s | std r1 tok/s (tok) | std r2 tok/s (tok) | FL r1 tok/s | FL r2 tok/s | FL r2 TPOT mean/med ms | FL r1 TTFT cold/warm s | FL accept r2 | cold 64K TTFT s / tok/s |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for a in ARMS:
    x = A[a]
    print(f"| {a} | {x['boot_s'] if x['boot_s'] is not None else 'live' if a=='A' else '—'} | {x['kv'] or '—'} | {'yes' if x['replay_line'] else 'no'} | {'yes' if x['rehome'] else 'no'} | {x['tune'] or '—'} | {x['preloaded'] or '—'} | {x['store'] or '—'} | "
          f"{f(x['rt_c1'])} ({', '.join(f(v) for v in x['rt_c1_runs'])}) | {f(x['rt_acc'], 3)} | {f(x['ttft8k'], 3)} | {f(x['std_r1'])} ({x['std_r1_tok']}) | {f(x['std_r2'])} ({x['std_r2_tok']}) | "
          f"{f(x['fl_r1_raw'])} | {f(x['fl_r2_raw'])}{'' if x['fl_r2_exact'] in (True, None) else ' (NOT exact)'} | {f(x['fl_r2_tpot_mean_ms'], 2)}/{f(x['fl_r2_tpot_median_ms'], 2)} | "
          f"{f(x['fl_r1_ttft_median_round0_s'], 3)}/{f(x['fl_r1_ttft_median_round1_s'], 3)} | {f(x['fl_r2_accept'], 3)} | {x['cold64'] or '—'} |")
print()
print("Per-class acceptance % — agent_fixture_x3 (8 prompts/class, token-weighted; per-prompt mean±sd in brackets):")
print("| arm | " + " | ".join(CATS) + " |"); print("|---|" + "---|" * len(CATS))
for a in ARMS:
    print(f"| {a} | " + " | ".join("—" if A[a]["fx"][c] is None else f"{100*A[a]['fx'][c]:.1f} [{100*(A[a]['fx_pp'][c][0] or 0):.0f}±{100*(A[a]['fx_pp'][c][1] or 0):.0f}]" for c in CATS) + " |")
print(f"| CTL | " + " | ".join("—" if CT["fx"][c] is None else f"{100*CT['fx'][c]:.1f}" for c in CATS) + " |")
print()
print("Per-class acceptance % — standard agent_fixture_o.sh × 3 CAT_ORDER rotations (mean, range):")
print("| arm | " + " | ".join(CATS) + " |"); print("|---|" + "---|" * len(CATS))
for a in ARMS:
    print(f"| {a} | " + " | ".join("—" if A[a]["rot"][c][0] is None else f"{100*A[a]['rot'][c][0]:.1f} (±{50*A[a]['rot'][c][1]:.1f}, n={A[a]['rot'][c][2]})" for c in CATS) + " |")
print()
print("Trajectory identity on agent_fixture_x3 (prompts with byte-identical output vs arm A, per class of 8):")
for a in ("B", "C", "D"):
    print(f"- {a} vs A: " + ", ".join(f"{c} {sum(1 for p, q in zip(A[a]['fx_sha'][c], A['A']['fx_sha'][c]) if p == q)}/{len(A['A']['fx_sha'][c])}" for c in CATS))
print()
print("Diagnostics:")
for a in ("B", "C"):
    x = A[a]
    print(f"- {a}: std replay r1 {f(rel(x['std_r1'], CT['std_r1']))}% r2 {f(rel(x['std_r2'], CT['std_r2']))}% vs CTL; r2 tokens {x['std_r2_tok']} vs CTL {f(CT['std_r2_tok'], 0)} ({f(rel(x['std_r2_tok'], CT['std_r2_tok']))}%); "
          f"FL r1 {f(rel(x['fl_r1'], CT['fl_r1']))}% FL r2 {f(rel(x['fl_r2'], CT['fl_r2']))}%; tool turns/emitted r2 {x['std_r2_tool']}; std accept r2 {x['std_r2_acc']} FL accept r2 {x['fl_r2_accept']}")
for a in ("A", "D"):
    x = A[a]; print(f"- {a}: std r2 tool turns/emitted {x['std_r2_tool']}; std accept r2 {x['std_r2_acc']}; FL accept r2 {x['fl_r2_accept']}")
