#!/usr/bin/env python3
"""divergence_margin.py A.json B.json OUT.json
For every prompt whose greedy text differs between lane A and lane B, find the first
divergent *token*, then teacher-force the shared prefix on the live server and read the
model's own logprob of A's token vs B's token at that position (echo=true, prompt_logprobs).
Same prefix -> same distribution -> the two logprobs come from one context.
|dlogp| ~ 0 means the decode paths split at a near-tie (fp noise); large |dlogp| means one
decode path picked a token the model itself rates clearly worse.
Also proves the instrument: scores one sequence twice and asserts byte-identical logprobs.
"""
import json, os, sys, urllib.request

base = os.getenv("BASE_URL", "http://127.0.0.1:30001").rstrip("/")
model = os.getenv("MODEL", "glm-5.3-big")
H = {"Authorization": f"Bearer {os.environ['API_KEY']}", "Content-Type": "application/json"}
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from greedy_equiv import PROMPTS  # noqa: E402  (same 20 prompts)


def post(path, payload):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(), headers=H)
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def ids_for(question, text):
    reasoning, _, content = text.partition("\u241f")
    msgs = [{"role": "user", "content": question}]
    kw = {"chat_template_kwargs": {"reasoning_effort": "low"}}  # must match greedy_equiv.py
    pre = post("/tokenize", {"model": model, "messages": msgs, "add_generation_prompt": True, **kw})["tokens"]
    full = post("/tokenize", {"model": model, "messages": msgs + [
        {"role": "assistant", "reasoning_content": reasoning, "content": content}],
        "add_generation_prompt": False, **kw})["tokens"]
    assert full[:len(pre)] == pre, "generation prompt is not a prefix of the full render"
    return pre, full[len(pre):]


def score(ids):
    d = post("/v1/completions", {"model": model, "prompt": ids, "max_tokens": 1, "echo": True,
                                 "logprobs": 1, "temperature": 0})
    return d["choices"][0]["prompt_logprobs"]


A, B = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
results = []
for k in sorted(A, key=int):
    if A[k] == B.get(k):
        continue
    q = PROMPTS[int(k)]
    pre, a = ids_for(q, A[k])
    pre2, b = ids_for(q, B[k])
    assert pre == pre2
    j = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    if j >= min(len(a), len(b)):
        results.append({"prompt": int(k), "note": "one text is a prefix of the other", "len_a": len(a), "len_b": len(b)})
        continue
    ctx = pre + a[:j]
    la = score(ctx + [a[j]])[-1]
    lb = score(ctx + [b[j]])[-1]
    ea = la[str(a[j])]; eb = lb[str(b[j])]
    results.append({
        "prompt": int(k), "tok_index": j, "shared_prefix_tokens": len(ctx),
        "a_tok": ea["decoded_token"], "a_logp": ea["logprob"], "a_rank": ea["rank"],
        "b_tok": eb["decoded_token"], "b_logp": eb["logprob"], "b_rank": eb["rank"],
        "dlogp_a_minus_b": ea["logprob"] - eb["logprob"],
    })
    r = results[-1]
    print(f"prompt {k:>2} tok {j:>4}: A {r['a_tok']!r:>14} lp={r['a_logp']:.4f} r{r['a_rank']} | "
          f"B {r['b_tok']!r:>14} lp={r['b_logp']:.4f} r{r['b_rank']} | dlogp={r['dlogp_a_minus_b']:+.4f}")

# instrument proof: same sequence twice must be identical
if results and "tok_index" in results[0]:
    k = str(results[0]["prompt"]); pre, a = ids_for(PROMPTS[int(k)], A[k])
    s1 = score(pre + a[:64]); s2 = score(pre + a[:64])
    lp1 = [next(iter(x.values()))["logprob"] if x else None for x in s1]
    lp2 = [next(iter(x.values()))["logprob"] if x else None for x in s2]
    print("INSTRUMENT repeat-identical:", lp1 == lp2, "max|d|=", max(abs((p or 0) - (q or 0)) for p, q in zip(lp1, lp2)))

ds = [abs(r["dlogp_a_minus_b"]) for r in results if "dlogp_a_minus_b" in r]
summary = {"sites": len(ds), "mean_abs_dlogp": sum(ds) / len(ds) if ds else None,
           "max_abs_dlogp": max(ds) if ds else None,
           "a_rank1": sum(r.get("a_rank") == 1 for r in results), "b_rank1": sum(r.get("b_rank") == 1 for r in results),
           "server_lane": os.getenv("LANE", "?")}
print("SUMMARY", json.dumps(summary))
json.dump({"summary": summary, "sites": results}, open(sys.argv[3], "w"), indent=1)
