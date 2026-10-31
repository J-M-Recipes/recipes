#!/usr/bin/env python3
"""qt1_tf.py — QT1-C capture: teacher-forced top-20 over a FIXED token corpus. RUNS ON THE BOX.

Why the corpus comes from the reference receipt: the historical h2-REF captures (G1/tf-G1-A.jsonl,
G1f/tf-G1-Pg.jsonl) store only {tokens, logprobs, top1}. Feeding those SAME token arrays back through
prompt_logprobs=20 reproduces the identical positions, so the new Pg61 capture lines up with h2-REF
position for position — no corpus file, no re-tokenisation, no drift.

Output: one JSONL row per doc, {doc, id, tokens, n, positions:[{lp, rank, top20:[[id, logprob]...]}]},
which is what qt1_analyze.py --tf-a/--tf-b consumes.

Usage:
  qt1_tf.py capture --ref tf-G1-A.jsonl --out tf20-Pg61a.jsonl [--conc 1] [--maxpos 0]
  qt1_tf.py compare --a tf20-Pg61a.jsonl --b tf20-Pg61b.jsonl      (repeat floor for this lane)
"""
import argparse, json, sys, time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "http://127.0.0.1:30006"
MODEL = "dsv41-flash-uva"


def post(path, body, timeout=900):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def capture_doc(doc, maxpos=0):
    ids = list(doc["tokens"])
    if maxpos:
        ids = ids[:maxpos]
    body = {"model": MODEL, "prompt": ids, "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 20}
    d = post("/v1/completions", body)
    pl = d["choices"][0].get("prompt_logprobs") or []
    pos = []
    for i in range(1, min(len(ids), len(pl))):
        tid = ids[i]
        ent = pl[i] or {}
        own = ent.get(str(tid)) or ent.get(tid)
        pairs = sorted(((int(t), float(v["logprob"])) for t, v in ent.items()), key=lambda x: -x[1])
        pos.append({"lp": float(own["logprob"]) if own else None,
                    "rank": int(own.get("rank", 0)) if own else None,
                    "top20": [[t, p] for t, p in pairs[:20]]})
    return {"doc": doc.get("id"), "id": doc.get("id"), "tokens": ids, "n": len(pos), "positions": pos}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    c = sub.add_parser("capture")
    c.add_argument("--ref", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--maxpos", type=int, default=0)
    c.add_argument("--conc", type=int, default=1)
    m = sub.add_parser("compare")
    m.add_argument("--a", required=True)
    m.add_argument("--b", required=True)
    a = ap.parse_args()

    if a.cmd == "compare":
        sys.path.insert(0, __file__.rsplit("/", 1)[0])
        import qt1_analyze as Q
        r = Q.tf_compare(a.a, a.b)
        print(json.dumps(r, indent=1))
        return 0

    docs = [json.loads(l) for l in open(a.ref) if l.strip()]
    out, errs = [], []

    def job(doc):
        for attempt in range(3):
            try:
                out.append(capture_doc(doc, a.maxpos))
                return
            except Exception as e:
                last = e
                time.sleep(1.5 * (attempt + 1))
        errs.append((doc.get("id"), repr(last)[:160]))

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.conc) as ex:
        list(ex.map(job, docs))
    out.sort(key=lambda r: r["doc"])
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({"capture": a.out, "docs": len(out), "n_errors": len(errs),
                      "positions": sum(r["n"] for r in out), "secs": round(time.time() - t0, 1),
                      "errors": errs[:4]}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
