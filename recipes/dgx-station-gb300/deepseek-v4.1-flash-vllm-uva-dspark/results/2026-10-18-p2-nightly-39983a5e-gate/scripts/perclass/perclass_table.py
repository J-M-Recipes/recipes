#!/usr/bin/env python3
"""perclass_table.py — summarize perclass_decode.py runs into the baseline table, and (with --control) apply the gate.

Baseline mode : python3 perclass_table.py --outdir DIR --arm v23            -> DIR/perclass-table-<arm>.{md,json}
Gate mode     : python3 perclass_table.py --outdir CAND_DIR --arm CAND --control-dir CTL_DIR --control CTL
Reads perclass-<arm>-r{1,2}-{c1,c8}.json, perclass-<arm>-nat-c1-nat.json (optional), agentfix-<arm>-x3.json (optional).
Spread per cell = |r1 - r2| / mean(r1, r2) (same-config, same-window; the noise floor).
Gate bar per class per cell (draft rule) = control mean x (1 - max(2%, 1.5 x control spread)). Candidate value = its r1/r2 mean.
"""
import argparse, json, os

ORDER = ["prose", "structured", "code", "shell_ops", "tool_json"]


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def summarize(d, arm):
    R = {r: {m: load(f"{d}/perclass-{arm}-{r}-{m}.json") for m in ("c1", "c8")} for r in ("r1", "r2")}
    nat = load(f"{d}/perclass-{arm}-nat-c1-nat.json"); x3 = load(f"{d}/agentfix-{arm}-x3.json")
    out = {}
    for c in ORDER:
        row = {}
        for m, key in (("c1", "decode_tok_s_pooled"), ("c8", "agg_decode_tok_s")):
            v = [R[r][m]["res"][c][key] if R[r][m] and c in R[r][m]["res"] else None for r in ("r1", "r2")]
            ok = [x for x in v if x is not None]; mean = sum(ok) / len(ok) if ok else None
            row[m] = {"r1": v[0], "r2": v[1], "mean": round(mean, 2) if mean else None,
                      "spread": round(abs(v[0] - v[1]) / mean, 4) if len(ok) == 2 and mean else None}
            row[m]["accept"] = [R[r][m]["res"][c]["accept"] if R[r][m] and c in R[r][m]["res"] else None for r in ("r1", "r2")]
            row[m]["tokens"] = [R[r][m]["res"][c]["tokens"] if R[r][m] and c in R[r][m]["res"] else None for r in ("r1", "r2")]
            row[m]["n"] = [R[r][m]["res"][c]["n_run"] if R[r][m] and c in R[r][m]["res"] else None for r in ("r1", "r2")]
            row[m]["tpot_ms"] = [R[r][m]["res"][c]["tpot_ms_median"] if R[r][m] and c in R[r][m]["res"] else None for r in ("r1", "r2")]
            row[m]["foreign_tokens"] = [R[r][m]["res"][c]["foreign_tokens"] if R[r][m] and c in R[r][m]["res"] else None for r in ("r1", "r2")]
            row[m]["max_running"] = [R[r][m]["res"][c]["max_running_seen"] if R[r][m] and c in R[r][m]["res"] else None for r in ("r1", "r2")]
            row[m]["errors"] = [R[r][m]["res"][c]["n_errors"] if R[r][m] and c in R[r][m]["res"] else None for r in ("r1", "r2")]
            row[m]["exact_fl"] = [R[r][m]["res"][c]["exact_fl"] if R[r][m] and c in R[r][m]["res"] else None for r in ("r1", "r2")]
        if nat and c in nat["res"]:
            row["nat"] = {k: nat["res"][c][k] for k in ("ct_median", "tokens", "decode_tok_s_pooled", "accept", "tool_call_frac", "n_run")}
        if x3 and c in x3.get("res", {}):
            row["x3"] = {"accept": x3["res"][c]["accept"], "tok_s": x3["res"][c]["tok_s"], "tokens": x3["res"][c]["tokens"]}
        out[c] = row
    fl = (R["r1"]["c1"] or {}).get("fl")
    return out, fl


def f(x, nd=1):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True); ap.add_argument("--arm", required=True)
    ap.add_argument("--control-dir"); ap.add_argument("--control")
    a = ap.parse_args()
    S, fl = summarize(a.outdir, a.arm)
    L = [f"## per-class decode wall-clock — {a.arm}", "", f"FL (forced tokens/prompt): {fl}", "",
         "| class | n C1/C8 | C1 tok/s r1 | r2 | mean | spread | C8 agg tok/s r1 | r2 | mean | spread | accept C1 r1/r2 | accept C8 r1/r2 | accept x3 fixture | tokens C1/C8 per run | natural len median |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in ORDER:
        r = S[c]; c1 = r["c1"]; c8 = r["c8"]
        L.append(f"| {c} | {c1['n'][0]}/{c8['n'][0]} | {f(c1['r1'])} | {f(c1['r2'])} | {f(c1['mean'])} | {pct(c1['spread'])} | "
                 f"{f(c8['r1'])} | {f(c8['r2'])} | {f(c8['mean'])} | {pct(c8['spread'])} | "
                 f"{f(c1['accept'][0], 3)}/{f(c1['accept'][1], 3)} | {f(c8['accept'][0], 3)}/{f(c8['accept'][1], 3)} | "
                 f"{f(r.get('x3', {}).get('accept'), 3)} | {c1['tokens'][0]}/{c8['tokens'][0]} | {f(r.get('nat', {}).get('ct_median'), 0)} |")
    hyg = []
    for c in ORDER:
        for m in ("c1", "c8"):
            x = S[c][m]
            if any(e for e in x["errors"] if e): hyg.append(f"{c} {m} errors {x['errors']}")
            if any(v is not None and abs(v) > 0 for v in x["foreign_tokens"]): hyg.append(f"{c} {m} foreign tokens {x['foreign_tokens']}")
            exp = 1 if m == "c1" else 8
            if any(v is not None and v > exp for v in x["max_running"]): hyg.append(f"{c} {m} max running {x['max_running']} > {exp}")
            if any(v is False for v in x["exact_fl"]): hyg.append(f"{c} {m} not exact FL")
    L += ["", "Hygiene: " + ("; ".join(hyg) if hyg else "no errors, no foreign tokens, running never above expected, all requests exact FL")]
    res = {"arm": a.arm, "fl": fl, "summary": S, "hygiene": hyg}
    if a.control:
        C, _ = summarize(a.control_dir, a.control)
        L += ["", f"### gate: {a.arm} vs control {a.control} (bar = ctl mean x (1 - max(2%, 1.5 x ctl spread)))", "",
              "| class | cell | ctl mean | ctl spread | bar | cand mean | Δ vs ctl | verdict |", "|---|---|---|---|---|---|---|---|"]
        verdict = {}
        for c in ORDER:
            for m in ("c1", "c8"):
                cm = C[c][m]["mean"]; sp = C[c][m]["spread"]; cand = S[c][m]["mean"]
                if cm is None or sp is None or cand is None:
                    L.append(f"| {c} | {m} | — | — | — | — | — | MISSING |"); verdict[f"{c}/{m}"] = "MISSING"; continue
                bar = cm * (1 - max(0.02, 1.5 * sp)); v = "PASS" if cand >= bar else "FAIL"; verdict[f"{c}/{m}"] = v
                L.append(f"| {c} | {m} | {cm:.1f} | {pct(sp)} | {bar:.1f} | {cand:.1f} | {100 * (cand / cm - 1):+.1f}% | {v} |")
        res["gate"] = verdict
        L += ["", f"GATE: {'PASS' if all(v == 'PASS' for v in verdict.values()) else 'FAIL'} ({sum(v == 'PASS' for v in verdict.values())}/{len(verdict)} cells)"]
    tag = a.arm if not a.control else f"{a.arm}-vs-{a.control}"
    open(f"{a.outdir}/perclass-table-{tag}.md", "w").write("\n".join(L) + "\n")
    json.dump(res, open(f"{a.outdir}/perclass-table-{tag}.json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
