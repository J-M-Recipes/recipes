#!/usr/bin/env python3
"""qt1_rescore.py — QT1-B: re-score the emitted sequence on the PREFILL path. RUNS ON THE BOX.

stdlib only. For each gen record, send prompt_ids + emitted server_token_ids as one PROMPT with
prompt_logprobs=20, max_tokens=1. The logprob the prefill path assigns to emitted token j is the
distribution at absolute index P+j; compare it with the decode path's returned values. Different
kernels run here (prefill/GEMM vs decode/GEMV, stock attention vs the peer v3 path), so this is the
first numerics check that actually covers the decode graphs.

Writes rescore-<arm>.jsonl with positions[j] = {lp, rank, top20:[[id, logprob],...]} aligned to the
emitted tokens. Token ids here are SERVER ids; the analysis only compares logprobs and argmax/rank
sets on this side, so no interning is needed.

Usage: python3 qt1_rescore.py --dir RUN --arm T1.0a [--conc 4]
"""
import argparse, hashlib, json, os, sys, time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "http://127.0.0.1:30006"
MODEL = "dsv41-flash-uva"


def intern(tok):
    """Same deterministic identity as qt1_gen.py (sha1 of the decoded string)."""
    return int(hashlib.sha1(tok.encode("utf-8", "surrogatepass")).hexdigest()[:12], 16)


def post(path, body, timeout=900):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def rescore(rec):
    ids = rec["prompt_ids"] + rec["server_token_ids"]
    P = len(rec["prompt_ids"])
    body = {"model": MODEL, "prompt": ids, "max_tokens": 1, "temperature": 0.0,
            "prompt_logprobs": 20, "return_token_ids": True}
    d = post("/v1/completions", body)
    pls = d["choices"][0].get("prompt_logprobs") or []
    pos = []
    for j, tid in enumerate(rec["server_token_ids"]):
        k = P + j
        if k >= len(pls) or not pls[k]:
            pos.append({"lp": None}); continue
        ent = pls[k]
        pairs = sorted(((intern(v.get("decoded_token", "")), float(v["logprob"]))
                        for t, v in ent.items()), key=lambda x: -x[1])
        own = next((v for t, v in ent.items() if int(t) == tid), None)
        rank = None
        if own is not None:
            rank = int(own.get("rank", 0)) or None
        pos.append({"lp": (float(own["logprob"]) if own is not None else None), "rank": rank,
                    "top20": [[t, p] for t, p in pairs[:20]]})
    return {"prompt_id": rec["prompt_id"], "arm": rec["arm"], "cls": rec["cls"], "positions": pos}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--conc", type=int, default=4)
    a = ap.parse_args()
    gen = [json.loads(l) for l in open(os.path.join(a.dir, "gen-%s.jsonl" % a.arm)) if l.strip()]
    pids = {json.loads(l)["id"]: json.loads(l)["ids"]
            for l in open(os.path.join(a.dir, "promptids.jsonl")) if l.strip()}
    for r in gen:
        r["prompt_ids"] = pids[r["prompt_id"]]
    out = os.path.join(a.dir, "rescore-%s.jsonl" % a.arm)
    done, errs = [], 0

    def job(r):
        nonlocal errs
        for attempt in range(4):
            try:
                done.append(rescore(r))
                return
            except Exception as e:
                last = e
                time.sleep(1.5 * (attempt + 1))
        errs += 1
        if errs < 6:
            print("ERR", r["prompt_id"], repr(last)[:160], flush=True)

    with ThreadPoolExecutor(max_workers=a.conc) as ex:
        list(ex.map(job, gen))
    order = {r["prompt_id"]: i for i, r in enumerate(gen)}
    done.sort(key=lambda r: order[r["prompt_id"]])
    with open(out, "w") as f:
        for r in done:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({"arm": a.arm, "rescored": len(done), "errors": errs, "out": out}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
