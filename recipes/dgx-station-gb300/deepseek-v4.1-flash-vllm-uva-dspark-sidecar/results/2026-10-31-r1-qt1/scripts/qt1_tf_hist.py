#!/usr/bin/env python3
"""qt1_tf_hist.py — P7 against the HISTORICAL h2-REF receipts (no new h2-REF boot needed).

The h2-REF A/A2 receipts store {tokens, logprobs, top1} per doc, and the new Pg61 capture was made by
feeding those exact token arrays back, so the positions line up. Both components of the P7 bar are
computable from what exists:

  mean|dlp|      | decode-path-free forced logprob (Pg61) - forced logprob (h2-REF) |
  top-1 flip     argmax of the Pg61 top-20 vs the h2-REF top1 id

jaccard20 is NOT computable (the historical receipts never stored a top-20), and it is not part of
the bar. The repeat floor for this lane is measured separately (Pg61a vs Pg61b) with the same code.

Usage: qt1_tf_hist.py --ref tf-G1-A.jsonl --cand tf20-Pg61a.jsonl [--ref2 tf-G1-A2.jsonl]
"""
import argparse, json, sys


def load_hist(p):
    return {json.loads(l)["id"]: json.loads(l) for l in open(p) if l.strip()}


def load_cand(p):
    return {json.loads(l)["doc"]: json.loads(l) for l in open(p) if l.strip()}


def compare(hist, cand):
    dl, n, flips, agree = [], 0, 0, 0
    per_doc = []
    for doc in sorted(set(hist) & set(cand)):
        h, c = hist[doc], cand[doc]
        if list(h["tokens"]) != list(c["tokens"]):
            per_doc.append({"doc": doc, "ERROR": "token arrays differ"})
            continue
        dh, dc = [], 0
        for i, (lp_h, lp_c) in enumerate(zip(h["logprobs"], [p["lp"] for p in c["positions"]])):
            if lp_h is None or lp_c is None:
                continue
            dh.append(abs(float(lp_h) - float(lp_c)))
            top = c["positions"][i].get("top20") or []
            h1 = h.get("top1") or []
            if top and i < len(h1):
                dc += int(int(top[0][0]) != int(h1[i]))
        n += len(dh)
        dl += dh
        flips += dc
        per_doc.append({"doc": doc, "positions": len(dh), "mean_abs_dlp": sum(dh) / max(len(dh), 1),
                        "top1_flips": dc})
    s = sorted(dl)
    return {"positions": n, "docs": len(per_doc),
            "mean_abs_dlp": sum(dl) / max(len(dl), 1),
            "p99_abs_dlp": s[int(0.99 * (len(s) - 1))] if s else None,
            "max_abs_dlp": max(dl) if dl else None,
            "top1_flips": flips, "top1_flip_rate": flips / max(n, 1),
            "per_doc": per_doc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = compare(load_hist(a.ref), load_cand(a.cand))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(r, f, indent=1)
    r.pop("per_doc")
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    sys.exit(main())
