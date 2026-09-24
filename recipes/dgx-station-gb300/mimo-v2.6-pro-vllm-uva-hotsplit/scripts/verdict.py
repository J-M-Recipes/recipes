#!/usr/bin/env python3
"""verdict.py <run-dir> — apply the promotion bars pinned in harness/protocol.yaml to this run's receipts. Fail-closed:
any missing file = BLOCKED, not PASS."""
import json, os, sys, glob
D = sys.argv[1]
def L(p):
    try: return json.load(open(os.path.join(D, p)))
    except Exception as e: return {"_err": f"{p}: {e}"}
rows, ok_all = [], True
def bar(name, val, cond, detail):
    global ok_all
    st = "BLOCKED" if val is None else ("PASS" if cond else "FAIL")
    if st != "PASS": ok_all = False
    rows.append((name, st, detail))

tf = [json.load(open(p)) for p in glob.glob(os.path.join(D, "tf-compare-ctrl-vs-v23.json"))]
if tf:
    t = tf[0]; rel = abs(t["ppl_cand"] - t["ppl_ref"]) / t["ppl_ref"]
    bar("TF ppl within 0.5% rel", rel, rel <= 0.005, f"ppl {t['ppl_ref']:.4f} -> {t['ppl_cand']:.4f} ({rel*100:.3f}%)")
    bar("TF top-1 flips <= 2.0%", t["top1_flip_rate"], t["top1_flip_rate"] <= 0.02,
        f"{t['top1_flips']}/{t['positions']} = {t['top1_flip_rate']*100:.3f}% · mean|Δ| {t['mean_abs_dlogprob']:.5f} p50 {t['p50']:.6f} p99 {t['p99']:.4f} max {t['max']:.3f} · docs {t['docs']} (tok-mismatch {t['docs_token_mismatch']})")
else:
    bar("TF compare", None, False, "missing tf-compare-ctrl-vs-v23.json")

for suite in ("dev", "heldout"):
    a, b = L(f"bfcl-ctrl-{suite}-summary.json"), L(f"bfcl-v23-{suite}-summary.json")
    if "_err" in a or "_err" in b:
        bar(f"BFCL {suite}", None, False, a.get("_err") or b.get("_err")); continue
    ca, cb = a["all"]["acc"], b["all"]["acc"]; d = (cb - ca) * 100
    bar(f"BFCL {suite} v23 >= ctrl - 1 pt", d, d >= -1.0,
        f"ctrl {a['all']['ok']}/{a['all']['n']} = {ca*100:.2f}% · v23 {b['all']['ok']}/{b['all']['n']} = {cb*100:.2f}% · Δ {d:+.2f} pt · no-call {a['all']['no_call']}/{b['all']['no_call']} · fail kinds ctrl {a['all']['fail_kinds']} v23 {b['all']['fail_kinds']}")
    for s, t in ((a, "ctrl"), (b, "v23")):
        er = s["errors"] / max(1, s["n"])
        bar(f"errors {t} {suite} <= 0.5%", er, er <= 0.005, f"{s['errors']}/{s['n']} · wall {s['wall_s']} s")

print("# MiMo-V2.6-Pro finish run — verdict (bars from harness/protocol.yaml)\n")
print("| bar | result | detail |\n|---|---|---|")
for r in rows: print(f"| {r[0]} | **{r[1]}** | {r[2]} |")
g = os.path.join(D, "greedy-compare.log")
print("\nGreedy parity (reported, not gated):", open(g).readline().strip() if os.path.exists(g) else "missing")
lc = os.path.join(D, "longctx-v24-1m.log")
if os.path.exists(lc): print("1M row (v24):", [l.strip() for l in open(lc) if "needle" in l or "decode" in l or "DONE" in l])
for p in sorted(glob.glob(os.path.join(D, "power-*.csv"))):
    v = [float(l.split(",")[1]) for l in open(p) if l.count(",") == 1 and l.split(",")[1].strip().replace(".", "", 1).isdigit()]
    if v: print(f"power {os.path.basename(p)}: mean {sum(v)/len(v):.1f} W, {len(v)} samples")
print(f"\nOVERALL: {'PROMOTE (experimental -> verified)' if ok_all else 'DO NOT PROMOTE'}")
