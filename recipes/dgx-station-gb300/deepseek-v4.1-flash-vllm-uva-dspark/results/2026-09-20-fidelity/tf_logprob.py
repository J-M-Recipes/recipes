#!/usr/bin/env python3
"""tf_logprob.py — teacher-forced per-token logprobs from an OpenAI-compatible vLLM endpoint.

capture:  python3 tf_logprob.py capture <tag> <corpus.jsonl>      -> tf-<tag>.jsonl  (one line per doc: tokens, logprobs, top1)
compare:  python3 tf_logprob.py compare tf-<ref>.jsonl tf-<cand>.jsonl  -> summary + tf-compare-<ref>-vs-<cand>.json

Mechanism: /v1/completions with max_tokens=1, prompt_logprobs=1 (vLLM extension). The server returns, for every prompt
position, the logprob of the ACTUAL next token and the top-1 token. No sampling, no speculation involved — prefill only —
so the comparison measures the serving path's numerics (offload layout, hook, kernels), not the drafter.
Corpus line: {"id": "...", "text": "..."}. Positions beyond MAXPOS per doc are dropped (server context is 1M; we cap cost).
Env: BASE_URL MODEL API_KEY MAXPOS (4096) CONC (4)
"""
from __future__ import annotations
import json, os, sys, time, math
from concurrent.futures import ThreadPoolExecutor
import urllib.request

BASE = os.getenv("BASE_URL", "http://127.0.0.1:30006/v1")
MODEL = os.getenv("MODEL", "dsv41-flash-uva")
KEY = os.getenv("API_KEY", "none")
MAXPOS = int(os.getenv("MAXPOS", "4096"))
CONC = int(os.getenv("CONC", "4"))


def capture_one(doc):
    payload = {"model": MODEL, "prompt": doc["text"], "max_tokens": 1, "temperature": 0,
               "prompt_logprobs": 1, "logprobs": 1, "truncate_prompt_tokens": MAXPOS}
    req = urllib.request.Request(f"{BASE}/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        j = json.load(r)
    ch = j["choices"][0]
    pl = ch.get("prompt_logprobs")
    if pl is None:
        raise RuntimeError("server returned no prompt_logprobs (flag unsupported on this build?)")
    toks, lps, top1 = [], [], []
    for pos in pl:
        if not pos:  # first position has no logprob
            continue
        # pos: {token_id: {"logprob","rank","decoded_token"}}. vLLM returns the top-1 entry (rank 1) and, when the
        # actual next token is not top-1, a second entry for the actual token (rank > 1). One entry => actual is top-1.
        items = [(int(t), d.get("rank", 1), d["logprob"]) for t, d in pos.items()]
        best = min(items, key=lambda x: x[1])
        actual = max(items, key=lambda x: x[1]) if len(items) > 1 else best
        toks.append(actual[0]); lps.append(actual[2]); top1.append(best[0])
    return {"id": doc["id"], "n": len(toks), "tokens": toks, "logprobs": lps, "top1": top1,
            "prompt_tokens": j.get("usage", {}).get("prompt_tokens"), "secs": round(time.time() - t0, 1)}


def capture_safe(doc):
    """One retry, then record the failure instead of aborting the whole capture."""
    for attempt in (1, 2):
        try:
            return capture_one(doc)
        except Exception as ex:  # noqa: BLE001
            err = f"{type(ex).__name__}: {str(ex)[:200]}"
            if attempt == 2:
                return {"id": doc["id"], "n": 0, "tokens": [], "logprobs": [], "top1": [], "error": err, "secs": 0}
            time.sleep(2)


def capture(tag, corpus):
    docs = [json.loads(l) for l in open(corpus)]
    out = f"tf-{tag}.jsonl"; n = 0; t0 = time.time(); errs = 0
    with open(out, "w") as f, ThreadPoolExecutor(CONC) as ex:
        for rec in ex.map(capture_safe, docs):
            if rec.get("error"): errs += 1; print(f"[{tag}] {rec['id']} ERROR {rec['error']}", flush=True)
            rec["tag"] = tag; rec["model"] = MODEL
            f.write(json.dumps(rec) + "\n"); f.flush(); n += rec["n"]
            print(f"[{tag}] {rec['id']} positions={rec['n']} {rec['secs']}s", flush=True)
    print(f"[{tag}] wrote {out}: docs={len(docs)} errors={errs} positions={n} wall={time.time()-t0:.0f}s")


def compare(fa, fb):
    A = {r["id"]: r for r in map(json.loads, open(fa))}
    B = {r["id"]: r for r in map(json.loads, open(fb))}
    ta, tb = next(iter(A.values()))["tag"], next(iter(B.values()))["tag"]
    dl, flips, n, tokmis, per = [], 0, 0, 0, {}
    for k in A:
        if k not in B: continue
        a, b = A[k], B[k]
        if a.get("error") or b.get("error") or not a["n"] or not b["n"]: continue
        m = min(a["n"], b["n"])
        if a["tokens"][:m] != b["tokens"][:m]:
            tokmis += 1; continue  # tokenization must match; otherwise the doc is not comparable
        d = [abs(x - y) for x, y in zip(a["logprobs"][:m], b["logprobs"][:m])]
        fl = sum(1 for x, y in zip(a["top1"][:m], b["top1"][:m]) if x != y)
        dl.extend(d); flips += fl; n += m
        per[k] = {"positions": m, "mean_abs_dlp": sum(d) / m, "max_abs_dlp": max(d), "top1_flips": fl}
    dl.sort()
    q = lambda p: dl[min(len(dl) - 1, int(len(dl) * p))]
    ppl_a = math.exp(-sum(A[k]["logprobs"][:per[k]["positions"]][i] for k in per for i in range(per[k]["positions"])) / n)
    ppl_b = math.exp(-sum(B[k]["logprobs"][:per[k]["positions"]][i] for k in per for i in range(per[k]["positions"])) / n)
    summ = {"ref": ta, "cand": tb, "positions": n, "docs": len(per), "docs_token_mismatch": tokmis,
            "mean_abs_dlogprob": sum(dl) / n, "p50": q(.5), "p99": q(.99), "max": dl[-1],
            "top1_flips": flips, "top1_flip_rate": flips / n, "ppl_ref": ppl_a, "ppl_cand": ppl_b, "per_doc": per}
    out = f"tf-compare-{ta}-vs-{tb}.json"; json.dump(summ, open(out, "w"), indent=1)
    print(f"TF Δlogprob {ta} → {tb}: positions={n} docs={len(per)} (token-mismatch docs={tokmis}) "
          f"mean|Δ|={summ['mean_abs_dlogprob']:.5f} p50={summ['p50']:.5f} p99={summ['p99']:.4f} max={summ['max']:.3f} "
          f"top-1 flips={flips} ({flips/n*100:.3f}%) ppl {ppl_a:.4f} → {ppl_b:.4f}; wrote {out}")


if __name__ == "__main__":
    if sys.argv[1] == "capture": capture(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "compare": compare(sys.argv[2], sys.argv[3])
    else: print(__doc__)
