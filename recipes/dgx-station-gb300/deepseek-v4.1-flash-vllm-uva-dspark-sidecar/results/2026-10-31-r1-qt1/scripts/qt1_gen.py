#!/usr/bin/env python3
"""qt1_gen.py — QT1 generation arm. RUNS ON THE BOX, loopback only (dark-window isolation).

stdlib only. Writes gen-<arm>.jsonl + promptids.jsonl into --out.

For each prompt row (private Hermes transcripts, per-class fixture):
  1. chat prepass, max_tokens=1, return_token_ids -> the exact prompt token ids the chat template
     produces (cache in promptids.jsonl; identical for every arm).
  2. /v1/completions on those ids with the arm's temperature, top_p=1, top_k=-1, seed, logprobs=20,
     return_token_ids=true. The GPU runner's rejection sampler returns top-20 logprobs computed from
     the TARGET logits at each verified decode position, plus the emitted token's own logprob.

Token identity: vLLM returns completions top_logprobs keyed by DECODED STRINGS, while emitted tokens
come back as ids. This script interns every decoded string to a stable integer for the run, so the
analysis can compare "is the emitted token one of these 20" without a tokenizer. server_token_ids are
kept verbatim for provenance and for the prefill re-score.

Usage: python3 qt1_gen.py --arm T1.0a --temp 1.0 --seed-base 100 --max-tokens 128 --out ~/dsv41/qt1/run1
"""
import argparse, hashlib, json, os, threading, urllib.request, sys
from concurrent.futures import ThreadPoolExecutor

BASE = "http://127.0.0.1:30006"
MODEL = "dsv41-flash-uva"
PROMPTS = os.environ.get("QT1_PROMPTS", os.path.expanduser("~/dsv41/bench/perclass_prompts.jsonl"))

_seen = set()


def intern(tok):
    """Deterministic token identity: sha1 of the decoded string.

    Must be the SAME function in qt1_rescore.py: the prefill re-score only ever sees token ids plus
    decoded strings, so it has to land on the same integers the gen pass used. A per-process counter
    would make every decode-vs-prefill argmax comparison silently disagree (top1_agree -> 0).
    """
    if tok not in _seen:
        _seen.add(tok)
    return int(hashlib.sha1(tok.encode("utf-8", "surrogatepass")).hexdigest()[:12], 16)


def post(path, body, timeout=900):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def prompt_ids(row, cache):
    pid = row["id"]
    if pid in cache:
        return cache[pid]
    d = post("/v1/chat/completions", {"model": MODEL, "messages": row["messages"], "max_tokens": 1,
                                     "temperature": 0.0, "chat_template_kwargs": {"thinking": False},
                                     "return_token_ids": True})
    ids = d["prompt_token_ids"]
    # the chat prepass already rendered any tools/system framing into these ids, so the completions
    # re-score below sees exactly the chat context.
    cache[pid] = ids
    return ids


def one(row, ids, arm, T, seed_base, max_tokens):
    idx = row.get("_idx", 0)
    body = {"model": MODEL, "prompt": ids, "max_tokens": max_tokens, "temperature": T,
            "top_p": 1.0, "top_k": -1, "seed": seed_base + idx,
            "logprobs": 20, "return_token_ids": True, "ignore_eos": False}
    d = post("/v1/completions", body)
    ch = d["choices"][0]
    lp = ch.get("logprobs") or {}
    tids = ch.get("token_ids") or []
    tops = lp.get("top_logprobs") or []
    tlps = lp.get("token_logprobs") or []
    toks = lp.get("tokens") or []
    n = min(len(tids), len(tops))
    rec_tops, rec_ids, rec_lp, rec_ss, ns = [], [], [], [], []
    for j in range(n):
        entry = tops[j] or {}
        ii = [(intern(k), float(v)) for k, v in entry.items() if isinstance(v, (int, float))]
        ii.sort(key=lambda x: -x[1])
        rec_tops.append({str(i): {"logprob": p} for i, p in ii})
        ns.append(len(ii))
        rec_ids.append(intern(toks[j]) if j < len(toks) else None)     # emitted token, interned
        rec_lp.append(float(tlps[j]) if j < len(tlps) and tlps[j] is not None else None)
        rec_ss.append(int(tids[j]))
    return {"prompt_id": row["id"], "cls": row["cls"], "arm": arm, "temperature": T,
            "seed": seed_base + idx, "max_tokens": max_tokens,
            "prompt_tokens": len(ids), "text": ch.get("text", "")[:4000],
            "token_ids": rec_ids, "server_token_ids": rec_ss, "token_logprobs": rec_lp,
            "top_logprobs": rec_tops, "n_entries": ns, "stats": d.get("metrics") or {}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--temp", type=float, required=True)
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--prompts", default=PROMPTS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = [json.loads(l) for l in open(a.prompts) if l.strip()]
    if a.limit:
        rows = rows[:a.limit]
    for i, r in enumerate(rows):
        r["_idx"] = i

    cache = {}
    pfile = os.path.join(a.out, "promptids.jsonl")
    if os.path.exists(pfile):
        cache = {json.loads(l)["id"]: json.loads(l)["ids"] for l in open(pfile) if l.strip()}
    missing = [r for r in rows if r["id"] not in cache]
    if missing:
        cols = {}
        with ThreadPoolExecutor(max_workers=a.conc) as ex:
            for r, ids in zip(missing, ex.map(lambda r: prompt_ids(r, cache), missing)):
                cols[r["id"]] = ids
        with open(pfile, "a") as f:
            for r in missing:
                f.write(json.dumps({"id": r["id"], "cls": r["cls"], "ids": cache[r["id"]]}) + "\n")
    print("prompt ids cached: %d (fetched %d)" % (len(cache), len(missing)), flush=True)

    out = os.path.join(a.out, "gen-%s.jsonl" % a.arm)
    done, errs = [], []
    lock = threading.Lock()

    def job(r):
        try:
            rec = one(r, cache[r["id"]], a.arm, a.temp, a.seed_base, a.max_tokens)
            with lock:
                done.append(rec)
                if len(done) % 20 == 0:
                    print("  %s: %d/%d" % (a.arm, len(done), len(rows)), flush=True)
        except Exception as e:
            with lock:
                errs.append((r["id"], repr(e)[:200]))

    with ThreadPoolExecutor(max_workers=a.conc) as ex:
        list(ex.map(job, rows))
    order = {r["id"]: i for i, r in enumerate(rows)}
    done.sort(key=lambda r: order[r["prompt_id"]])
    with open(out, "w") as f:
        for rec in done:
            f.write(json.dumps(rec) + "\n")
    print(json.dumps({"arm": a.arm, "records": len(done), "errors": errs[:5], "n_errors": len(errs),
                      "positions": sum(len(r["token_ids"]) for r in done),
                      "vocab_interned": len(_seen),
                      "out": out}, indent=1))
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
