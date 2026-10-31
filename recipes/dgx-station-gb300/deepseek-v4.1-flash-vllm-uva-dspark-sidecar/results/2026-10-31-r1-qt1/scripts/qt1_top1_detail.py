#!/usr/bin/env python3
"""qt1_top1_detail.py — where does the P4 top-1 excess live?

P4 compares the fraction of emitted tokens that equal the reported argmax with the mean of the
reported max probability. A sampler that is distribution-preserving in the PIT sense can still show
this if the emitted token is over-drawn from the top-1 cell, and the two live hypotheses are:

  H-draft   a draft-favouring acceptance path: accepted draft tokens ARE the drafter's argmax, so the
            excess should be concentrated where the target is confident (top max-p quartile) and
            where draft and target agree most often;
  H-calib   a global calibration offset in the reported max prob (reporting bug, not sampling), which
            would be flat across confidence.

Splits by class and by max-p quartile, with a paired binomial z per cell (p_i = max prob of position i,
so the variance is sum p(1-p), not n*p(1-p) with a pooled p).

Usage: qt1_top1_detail.py --dir RUN --arms T1.0a,T1.0b
"""
import argparse, json, math, os, sys


def load(d, arms):
    rows = []
    for a in arms:
        p = os.path.join(d, "gen-%s.jsonl" % a)
        for line in open(p):
            r = json.loads(line)
            tl, tids = r["top_logprobs"], r["token_ids"]
            for j, tid in enumerate(tids):
                if j >= len(tl) or not tl[j]:
                    continue
                lps = sorted((float(v["logprob"]) for v in tl[j].values()), reverse=True)
                ids = [int(k) for k, v in sorted(tl[j].items(), key=lambda kv: -float(kv[1]["logprob"]))]
                pm = math.exp(lps[0])
                rows.append({"arm": a, "cls": r["cls"], "pm": pm, "hit": int(tid == ids[0]),
                             "logpm": -lps[0]})
    return rows


def cell(rows):
    n = len(rows)
    if not n:
        return None
    obs = sum(r["hit"] for r in rows) / n
    exp = sum(r["pm"] for r in rows) / n
    var = sum(r["pm"] * (1 - r["pm"]) for r in rows) / (n * n)
    return {"n": n, "obs": obs, "exp": exp, "diff": obs - exp,
            "z": (obs - exp) / math.sqrt(var) if var > 0 else None,
            "mean_max_p": exp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--arms", default="T1.0a,T1.0b")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = load(a.dir, [x for x in a.arms.split(",") if x])
    out = {"overall": cell(rows), "by_arm": {}, "by_class": {}, "by_confidence_quartile": {}}
    for arm in sorted(set(r["arm"] for r in rows)):
        out["by_arm"][arm] = cell([r for r in rows if r["arm"] == arm])
    for c in sorted(set(r["cls"] for r in rows)):
        out["by_class"][c] = cell([r for r in rows if r["cls"] == c])
    s = sorted(rows, key=lambda r: r["pm"])
    q = len(s) // 4
    for i, name in enumerate(["Q1_low_conf", "Q2", "Q3", "Q4_high_conf"]):
        seg = s[i * q:(i + 1) * q] if i < 3 else s[3 * q:]
        out["by_confidence_quartile"][name] = cell(seg)
        out["by_confidence_quartile"][name]["max_p_range"] = [seg[0]["pm"], seg[-1]["pm"]]
    # the same split by log-domain width of the top-2 gap (draft/target agreement proxy)
    def gap(r):
        return r["pm"]
    txt = json.dumps(out, indent=1)
    if a.out:
        with open(a.out, "w") as f:
            f.write(txt)
    print(txt)


if __name__ == "__main__":
    sys.exit(main())
