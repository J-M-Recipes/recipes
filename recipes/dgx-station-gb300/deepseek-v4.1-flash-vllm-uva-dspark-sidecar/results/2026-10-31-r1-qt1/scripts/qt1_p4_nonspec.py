#!/usr/bin/env python3
"""qt1_p4_nonspec.py — discriminator for QT1's P4 argmax excess (Milo, 2026-10-31). Box-side, stdlib, loopback.

Takes the low-confidence positions (reported p_max < 0.75) from a QT1 gen file and replays each as
prompt = chat prompt ids + the emitted prefix up to that position, max_tokens=1, T=1, top_p=1, top_k=-1,
logprobs=20, return_tokens_as_token_ids=true (no string identity). The token at that position is then sampled by the
target with NO drafts in flight, and the request reports its own distribution. Compare the argmax rate with the
reported mean p_max, exactly as P4 does.
  excess here too      -> sampler-vs-reported-distribution mismatch (not speculation)
  no excess here       -> the speculative accept path is what favours the argmax/draft
Usage: python3 qt1_p4_nonspec.py --gen run1/gen-T1.0a.jsonl --prompts run1/promptids.jsonl --per-rec 25 \
         --seeds 2 --out p4ns.jsonl --conc 4
"""
import argparse, json, math, random, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor
BASE, MODEL = "http://127.0.0.1:30006", "dsv41-flash-uva"

def post(body):
    req = urllib.request.Request(BASE + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True); ap.add_argument("--prompts", required=True)
    ap.add_argument("--per-rec", type=int, default=25); ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--pmax", type=float, default=0.75); ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--out", required=True); ap.add_argument("--rng", type=int, default=7)
    a = ap.parse_args()
    P = {json.loads(l)["id"]: json.loads(l)["ids"] for l in open(a.prompts)}
    rnd = random.Random(a.rng); jobs = []
    for line in open(a.gen):
        r = json.loads(line)
        cand = []
        for j, tops in enumerate(r["top_logprobs"]):
            if tops and math.exp(max(v["logprob"] for v in tops.values())) < a.pmax:
                cand.append(j)
        for j in rnd.sample(cand, min(a.per_rec, len(cand))):
            for s in range(a.seeds):
                jobs.append((r["prompt_id"], r["cls"], j, P[r["prompt_id"]] + r["server_token_ids"][:j], 50000 + len(jobs)))
    lock = threading.Lock(); out = open(a.out, "w"); n = [0, 0]
    def run(job):
        pid, cls, j, ids, seed = job
        body = {"model": MODEL, "prompt": ids, "max_tokens": 1, "temperature": 1.0, "top_p": 1.0, "top_k": -1,
                "seed": seed, "logprobs": 20, "return_tokens_as_token_ids": True, "return_token_ids": True}
        for attempt in range(3):
            try:
                d = post(body); break
            except Exception as e:
                err = repr(e)[:120]
        else:
            with lock: n[1] += 1
            return
        ch = d["choices"][0]; lp = ch["logprobs"]
        tops = {int(k.split(":")[1]): float(v) for k, v in (lp["top_logprobs"][0] or {}).items()}
        em = int(ch["token_ids"][0])
        with lock:
            out.write(json.dumps({"pid": pid, "cls": cls, "pos": j, "seed": seed, "emitted": em, "tops": tops}) + "\n")
            n[0] += 1
    with ThreadPoolExecutor(a.conc) as ex:
        list(ex.map(run, jobs))
    out.close()
    print(json.dumps({"jobs": len(jobs), "ok": n[0], "failed": n[1], "out": a.out}))

if __name__ == "__main__":
    main()
