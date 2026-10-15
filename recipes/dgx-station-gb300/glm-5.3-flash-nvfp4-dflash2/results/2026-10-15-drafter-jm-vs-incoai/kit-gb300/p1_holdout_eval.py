#!/usr/bin/env python3
"""P3 fixture (c): our P1 holdout, 300 stratified prompts, thinking OFF, greedy, 300 tokens, per-bucket acceptance.

Runs inside the serving container (needs transformers + aiohttp). Prompts are rendered exactly the way P1
generated its training targets (p1/build/generate_v2.py: prompt_messages + tools + chat_template_mm.jinja,
enable_thinking=False, add_generation_prompt=True) and sent as raw text to /v1/completions, temperature 0.
Buckets are run one after another so /metrics deltas are per bucket.
Acceptance length is reported two ways:
  accept_len_completion = completion_tokens / drafts   (diag_matrix.py convention, weekend RESULTS tables)
  mean_acceptance_length = 1 + accepted / drafts       (canada-quant holdout_client.py convention)
"""
import argparse, asyncio, json, random, re, time, urllib.request
import aiohttp
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:30021")
ap.add_argument("--holdout", required=True)
ap.add_argument("--template", required=True)
ap.add_argument("--model-dir", required=True)
ap.add_argument("--served-name", default="glm-5.3-flash")
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--seed", type=int, default=20261015)
ap.add_argument("--concurrency", type=int, default=8)
ap.add_argument("--max-tokens", type=int, default=300)
ap.add_argument("--out", required=True)
ap.add_argument("--label", default="")
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True)
template = open(a.template).read()
rows = [json.loads(l) for l in open(a.holdout)]
buckets = {}
for r in rows:
    buckets.setdefault(r["bucket"], []).append(r)
order = ["chat", "agent", "code_stem", "structured"]
total = len(rows)
rng = random.Random(a.seed)
picked = {}
for b in order:
    pool = sorted(buckets[b], key=lambda r: r["id"])
    k = round(a.n * len(pool) / total)
    picked[b] = rng.sample(pool, k)


def prompt_messages(rec):
    conv = rec["conversations"]
    if (rec.get("meta") or {}).get("scaffold_not_target"):
        return [m for m in conv if m.get("role") in ("system", "user")]
    return [dict(m) for m in conv]


def render(rec):
    kw = dict(tokenize=False, add_generation_prompt=True, chat_template=template, enable_thinking=False)
    if rec.get("tools"):
        kw["tools"] = rec["tools"]
    return tok.apply_chat_template(prompt_messages(rec), **kw)


def metrics():
    txt = urllib.request.urlopen(a.url + "/metrics", timeout=30).read().decode()
    m = {"drafts": 0.0, "draft_tokens": 0.0, "accepted": 0.0, "per_pos": {}}
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        mm = re.match(r"(vllm:spec_decode_num_\w+)(\{[^}]*\})?\s+([0-9.eE+-]+)", line)
        if not mm:
            continue
        name, labels, val = mm.group(1), mm.group(2) or "", float(mm.group(3))
        if name == "vllm:spec_decode_num_drafts_total": m["drafts"] += val
        elif name == "vllm:spec_decode_num_draft_tokens_total": m["draft_tokens"] += val
        elif name == "vllm:spec_decode_num_accepted_tokens_total": m["accepted"] += val
        elif name.startswith("vllm:spec_decode_num_accepted_tokens_per_pos"):
            pos = re.search(r'position="(\d+)"', labels)
            if pos:
                m["per_pos"][int(pos.group(1))] = m["per_pos"].get(int(pos.group(1)), 0.0) + val
    return m


async def run_bucket(name, recs):
    reqs = []
    sem = asyncio.Semaphore(a.concurrency)

    async def one(sess, rec):
        prompt = render(rec)
        body = {"model": a.served_name, "prompt": prompt, "max_tokens": a.max_tokens, "temperature": 0.0, "top_p": 1.0,
                "skip_special_tokens": False}
        async with sem:
            t0 = time.time()
            try:
                async with sess.post(a.url + "/v1/completions", json=body, timeout=aiohttp.ClientTimeout(total=1200)) as resp:
                    j = await resp.json()
                    st = resp.status
                u = j.get("usage") or {}
                ch = (j.get("choices") or [{}])[0]
                reqs.append({"id": rec["id"], "source": rec["source"], "status": st, "prompt_tokens": u.get("prompt_tokens"),
                             "completion_tokens": u.get("completion_tokens") or 0, "finish": ch.get("finish_reason"),
                             "wall_s": round(time.time() - t0, 3), "text_head": (ch.get("text") or "")[:160],
                             "error": None if st == 200 else j})
            except Exception as e:  # record, do not invent
                reqs.append({"id": rec["id"], "source": rec["source"], "status": None, "completion_tokens": 0, "error": repr(e)})

    before = metrics(); t0 = time.time()
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=a.concurrency + 8)) as sess:
        await asyncio.gather(*(one(sess, r) for r in recs))
    dt = time.time() - t0
    time.sleep(1.0)
    after = metrics()
    drafts = after["drafts"] - before["drafts"]; dtok = after["draft_tokens"] - before["draft_tokens"]
    acc = after["accepted"] - before["accepted"]
    per_pos = {p: after["per_pos"].get(p, 0) - before["per_pos"].get(p, 0) for p in sorted(after["per_pos"])}
    comp = sum(r["completion_tokens"] for r in reqs)
    return {
        "bucket": name, "n": len(recs), "ok": sum(1 for r in reqs if r.get("status") == 200),
        "concurrency": a.concurrency, "max_tokens": a.max_tokens, "wall_s": round(dt, 2),
        "completion_tokens": comp, "agg_tok_s": round(comp / dt, 1) if dt else None,
        "drafts": drafts, "draft_tokens": dtok, "accepted_tokens": acc,
        "accept_len_completion": round(comp / drafts, 3) if drafts else None,
        "mean_acceptance_length": round(1 + acc / drafts, 3) if drafts else None,
        "draft_token_acceptance_rate": round(acc / dtok, 4) if dtok else None,
        "per_position_acceptance_rate": {p: round(v / drafts, 4) for p, v in per_pos.items()} if drafts else {},
        "sources": sorted({r["source"] for r in recs}),
        "requests": sorted(reqs, key=lambda r: r["id"]),
    }


async def main():
    out = {"label": a.label, "seed": a.seed, "n": a.n, "picked_counts": {b: len(picked[b]) for b in order},
           "regime": "thinking OFF, temperature 0, /v1/completions with chat_template_mm.jinja render (P1 generation path)",
           "buckets": {}}
    for b in order:
        res = await run_bucket(b, picked[b])
        out["buckets"][b] = res
        print("BUCKET " + json.dumps({k: v for k, v in res.items() if k != "requests"}), flush=True)
    tot_d = sum(out["buckets"][b]["drafts"] for b in order)
    tot_c = sum(out["buckets"][b]["completion_tokens"] for b in order)
    tot_a = sum(out["buckets"][b]["accepted_tokens"] for b in order)
    out["overall"] = {"accept_len_completion": round(tot_c / tot_d, 3) if tot_d else None,
                      "mean_acceptance_length": round(1 + tot_a / tot_d, 3) if tot_d else None,
                      "completion_tokens": tot_c, "drafts": tot_d}
    json.dump(out, open(a.out, "w"), indent=2)
    print("RESULT " + json.dumps({"label": a.label, "overall": out["overall"],
                                  "by_bucket": {b: out["buckets"][b]["accept_len_completion"] for b in order}}), flush=True)

asyncio.run(main())
