#!/usr/bin/env python3
"""No-dark W3 6.1: per-position draft survival on live Pg63, per concurrency, T=0 and T=1.
Metric deltas of vllm:spec_decode_num_accepted_tokens_per_pos / num_drafts around each cell."""
import json, re, sys, threading, time, urllib.request, statistics as st
ROOT = "http://gb300:30006"; MODEL = "deepseek-v4.1-flash"
OUT = "~/hermes/research/dsv41-gb300-owen-transfer-20261121/cards/receipts/survival.json"
PROMPTS = [
 "Write a Python function that parses an ISO-8601 duration string like P3DT4H5M into seconds, with tests.",
 "Explain how a refrigerator moves heat from inside to outside, step by step, for a curious teenager.",
 "Write a 400-word short story about a lighthouse keeper who finds a message in a bottle.",
 "Return a JSON array of 12 objects describing fictional books with fields title, author, year, genre, pages.",
 "A train leaves at 9:40, travels 212 km at 80 km/h, stops 12 minutes, then 95 km at 60 km/h. When does it arrive? Show steps.",
 "Summarize the causes and consequences of the 1929 stock market crash in about 300 words.",
 "Rewrite this as a formal business email: hey, the shipment is late again, we need it by friday or we cancel.",
 "Write a bash script that finds the 10 largest files under a directory and prints them in human-readable sizes.",
]
def metrics():
    m = urllib.request.urlopen(ROOT + "/metrics", timeout=10).read().decode()
    pos = {int(p): float(v) for p, v in re.findall(r'^vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^}]*position="(\d+)"\} ([0-9.e+]+)', m, re.M)}
    g = lambda n: float(re.search(rf'^vllm:{n}_total\{{[^}}]*\}} ([0-9.e+]+)', m, re.M).group(1))
    return {"pos": pos, "drafts": g("spec_decode_num_drafts"), "dtok": g("spec_decode_num_draft_tokens"),
            "acc": g("spec_decode_num_accepted_tokens"), "gen": g("generation_tokens")}
def one(prompt, temp, out):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 600, "temperature": temp,
            "chat_template_kwargs": {"thinking": False}}
    r = urllib.request.Request(ROOT + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time(); u = json.loads(urllib.request.urlopen(r, timeout=600).read())["usage"]; out.append((u["completion_tokens"], time.time() - t))
def cell(conc, temp, secs=75):
    m0 = metrics(); t0 = time.time(); done = []; stop = [False]
    def worker(i):
        k = i
        while not stop[0]:
            one(PROMPTS[k % len(PROMPTS)], temp, done); k += conc
    ths = [threading.Thread(target=worker, args=(i,)) for i in range(conc)]
    [t.start() for t in ths]; time.sleep(secs); stop[0] = True; [t.join() for t in ths]
    m1 = metrics(); wall = time.time() - t0
    nd = m1["drafts"] - m0["drafts"]
    surv = {p: round((m1["pos"].get(p, 0) - m0["pos"].get(p, 0)) / nd, 4) for p in range(8)} if nd else {}
    return {"conc": conc, "temp": temp, "wall": round(wall, 1), "drafts": nd, "draft_tokens": m1["dtok"] - m0["dtok"],
            "accepted": m1["acc"] - m0["acc"], "gen_tokens": m1["gen"] - m0["gen"],
            "mean_drafts_per_step": round((m1["dtok"] - m0["dtok"]) / nd, 2) if nd else None,
            "accept_len": round(1 + (m1["acc"] - m0["acc"]) / nd, 3) if nd else None,
            "agg_tok_s": round((m1["gen"] - m0["gen"]) / wall, 1), "requests": len(done), "survival_pos0_7": surv}
res = []
for conc, temp in [(1, 0), (1, 1.0), (8, 0), (8, 1.0), (16, 1.0), (32, 1.0)]:
    r = cell(conc, temp); res.append(r); print(json.dumps(r), flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
