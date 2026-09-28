#!/usr/bin/env python3
"""verdict2.py <run-dir> — apply harness/protocol-v2.yaml bars to the MOPD run receipts. Fail-closed: missing file = BLOCKED."""
import json, os, sys, glob
D = sys.argv[1]
def L(p):
    try: return json.load(open(os.path.join(D, p)))
    except Exception as e: return {"_err": f"{p}: {e}"}
rows, ok_all = [], True
def bar(name, val, cond, detail, gated=True):
    global ok_all
    st = "BLOCKED" if val is None else ("PASS" if cond else "FAIL")
    if gated and st != "PASS": ok_all = False
    rows.append((name, st if gated else f"{st} (reported)", detail))
def tfrow(name, f, gated, ppl_bar=0.005, flip_bar=0.02):
    t = L(f)
    if "_err" in t: bar(name, None, False, t["_err"], gated); return
    rel = abs(t["ppl_cand"] - t["ppl_ref"]) / t["ppl_ref"]
    bar(f"{name}: TF ppl within 0.5% rel", rel, rel <= ppl_bar, f"ppl {t['ppl_ref']:.4f} -> {t['ppl_cand']:.4f} ({rel*100:.3f}%)", gated)
    bar(f"{name}: TF top-1 flips <= 2.0%", t["top1_flip_rate"], t["top1_flip_rate"] <= flip_bar,
        f"{t['top1_flips']}/{t['positions']} = {t['top1_flip_rate']*100:.3f}% · mean|Δ| {t['mean_abs_dlogprob']:.5f} p99 {t['p99']:.4f} max {t['max']:.3f} · docs {t['docs']}", gated)
def bfclrow(name, a_tag, b_tag, suite, gated):
    a, b = L(f"bfcl-{a_tag}-{suite}-summary.json"), L(f"bfcl-{b_tag}-{suite}-summary.json")
    if "_err" in a or "_err" in b: bar(f"{name} BFCL {suite}", None, False, a.get("_err") or b.get("_err"), gated); return
    ca, cb = a["all"]["acc"], b["all"]["acc"]; d = (cb - ca) * 100
    bar(f"{name} BFCL {suite} {b_tag} >= {a_tag} - 1 pt", d, d >= -1.0,
        f"{a_tag} {a['all']['ok']}/{a['all']['n']} = {ca*100:.2f}% · {b_tag} {b['all']['ok']}/{b['all']['n']} = {cb*100:.2f}% · Δ {d:+.2f} pt · no-call {a['all']['no_call']}/{b['all']['no_call']} · fail kinds {a['all']['fail_kinds']} / {b['all']['fail_kinds']}", gated)
    for s, t in ((a, a_tag), (b, b_tag)):
        er = s["errors"] / max(1, s["n"]); bar(f"errors {t} {suite} <= 0.5%", er, er <= 0.005, f"{s['errors']}/{s['n']} · wall {s['wall_s']} s", gated)

# Q1 residency A/B on MOPD weights (gated — decides whether 'verified' carries over)
tfrow("Q1 residency m-ctrl->m-hot", "tf-compare-mctrl-vs-mhot.json", True)
for suite in ("dev", "heldout"): bfclrow("Q1 residency", "mctrl", "mhot", suite, True)
# Q1 weight swap RL->MOPD (reported)
tfrow("Q1 weights RL(09-24 ctrl)->m-ctrl", "tf-compare-ctrl-vs-mctrl.json", False)   # tf_logprob names the output from the "ref" field inside the jsonl (ctrl), not the file name
old = L("bfcl-rlctrl0924-dev-summary.json"); oldh = L("bfcl-rlctrl0924-heldout-summary.json")   # copied from results/2026-09-24-promotion/receipts/bfcl-ctrl-*
for suite, o in (("dev", old), ("heldout", oldh)):
    n = L(f"bfcl-mctrl-{suite}-summary.json")
    if "_err" in o or "_err" in n: bar(f"Q1 weights BFCL {suite}", None, False, o.get("_err") or n.get("_err"), False); continue
    d = (n["all"]["acc"] - o["all"]["acc"]) * 100
    bar(f"Q1 weights BFCL {suite} RL->MOPD (finding if < -2 pt)", d, d >= -2.0, f"RL {o['all']['ok']}/{o['all']['n']} = {o['all']['acc']*100:.2f}% · MOPD {n['all']['ok']}/{n['all']['n']} = {n['all']['acc']*100:.2f}% · Δ {d:+.2f} pt", False)
r = L("bfcl-rlhot-dev-summary.json")
if "_err" not in r: rows.append(("Q1 rl-hot dev re-run (same-model repeat)", "info", f"{r['all']['ok']}/{r['all']['n']} = {r['all']['acc']*100:.2f}% (09-24 v23: 563/600 = 93.83%)"))
# Q2 ranking portability
p = L("portability.json")
if "_err" in p: bar("Q2 ranking portability", None, False, p["_err"])
else:
    a = p.get("hbm_served_share_rl_ranking_on_mopd"); b = p.get("hbm_served_share_rl_ranking_on_rl_live")
    if b is None: bar("Q2 ranking portability", None, False, "no RL live reference in portability.json")
    else: bar("Q2 RL ranking on MOPD live >= RL ranking on RL live - 3 pt", a - b, (a - b) >= -0.03,
              f"MOPD {a*100:.1f}% vs RL {b*100:.1f}% (oracles {p['hbm_served_share_oracle_mopd']*100:.1f}% / {p['hbm_served_share_oracle_rl_live']*100:.1f}%; Jaccard rank-vs-MOPD-oracle {p['jaccard_rlrank_vs_mopd_oracle']:.3f}, RL-vs-RL {p['jaccard_rlrank_vs_rl_oracle']:.3f}; tokens/layer MOPD {p['mopd_live_tokens_per_layer']} RL {p.get('rl_live_tokens_per_layer')}) -> {p['verdict']}")
# Q3 flood
fr, fc, fh = L("flood-rlhot-summary.json"), L("flood-mctrl-summary.json"), L("flood-mhot-summary.json")
def fl(s): a = s["all"]; return f"solved {a['solved']}/{a['tasks']} · within-rep {a['within_turn_rep']*100:.2f}% · cross-rep {a['cross_turn_rep']*100:.2f}% · flood tasks {a['tasks_with_flood']} · dup tasks {a['tasks_with_any_dup']} · calls {a['calls']} · max/turn {a['max_calls_turn']} · tokens {a['completion_tokens']} · unsolved {a['unsolved']}"
for tag, s in (("rl-hot (RL)", fr), ("m-ctrl (MOPD stock)", fc), ("m-hot (MOPD hotsplit)", fh)):
    rows.append((f"Q3 flood {tag}", "info" if "_err" not in s else "BLOCKED", fl(s) if "_err" not in s else s["_err"]))
if "_err" not in fc and "_err" not in fh:
    dflood = abs(fc["all"]["tasks_with_flood"] - fh["all"]["tasks_with_flood"]); dsol = abs(fc["all"]["solved"] - fh["all"]["solved"])
    bar("Q3 residency sanity: |Δ flood tasks| <= 2 and |Δ solved| <= 3", dflood + dsol, dflood <= 2 and dsol <= 3, f"Δ flood tasks {dflood}, Δ solved {dsol}")

print("# MiMo-V2.6-Pro-MOPD run — verdict (bars from harness/protocol-v2.yaml)\n")
print("| bar | result | detail |\n|---|---|---|")
for r in rows: print(f"| {r[0]} | **{r[1]}** | {r[2]} |")
for f in ("greedy-compare-residency.log", "greedy-compare-weights.log", "tf-compare-residency.log", "tf-compare-weights.log", "ttft-mhot.log", "fixture-mhot.log", "portability.log"):
    p = os.path.join(D, f); print(f"\n{f}: " + (open(p).read().strip()[:1200] if os.path.exists(p) else "missing"))
for p in sorted(glob.glob(os.path.join(D, "power-*.csv"))):
    v = [float(l.split(",")[1]) for l in open(p) if l.count(",") == 1 and l.split(",")[1].strip().replace(".", "", 1).isdigit()]
    if v: print(f"power {os.path.basename(p)}: mean {sum(v)/len(v):.1f} W, {len(v)} samples")
print(f"\nOVERALL (gated bars only): {'PASS — recipe verified on MOPD weights' if ok_all else 'FAIL/BLOCKED — do not update status'}")
