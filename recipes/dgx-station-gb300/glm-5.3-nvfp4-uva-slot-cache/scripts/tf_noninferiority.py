#!/usr/bin/env python3
"""tf_noninferiority.py REF.json OUT.json -- formal decode noninferiority for a slot-cache lane.

REF.json is a lane's greedy output set (greedy_equiv.py, 20 prompts). On the LIVE server, teacher-force each
full generated sequence via /v1/completions (echo=true, logprobs=1, max_tokens=1) and read the model's logprob of
EVERY generated token under the same prefix. Scoring the SAME reference text on two servers makes the
comparison apples-to-apples: if the two lanes compute the same function, per-token logprobs are byte-identical;
if a lane's numerics drifted, the distribution of |dlogp| tells you by how much and where.

Run it twice: once on the reference server (self-score), once on the candidate. Then compare the two OUT.json:
  python3 tf_noninferiority.py --compare OUT_ref.json OUT_cand.json
reports n tokens, mean/p50/p95/p99/max |dlogp|, fraction of tokens with |dlogp| > 0.01 / 0.1 / 1.0, and the
per-prompt max. Noninferiority bar (this recipe): p99 |dlogp| <= 0.05 and max <= 0.5 over >= 3000 tokens.
"""
import json, os, sys, urllib.request

if sys.argv[1] == "--compare":
    a, b = json.load(open(sys.argv[2])), json.load(open(sys.argv[3]))
    d = []; per = {}
    for k in a["prompts"]:
        la, lb = a["prompts"][k]["logprobs"], b["prompts"][k]["logprobs"]
        n = min(len(la), len(lb)); dd = [abs(x - y) for x, y in zip(la[:n], lb[:n])]
        d += dd; per[k] = max(dd) if dd else 0.0
    d.sort(); n = len(d)
    q = lambda p: d[min(n - 1, int(p * n))] if n else float("nan")
    out = {"tokens": n, "mean": sum(d) / n if n else None, "p50": q(0.5), "p95": q(0.95), "p99": q(0.99), "max": d[-1] if n else None,
           "frac_gt_0.01": sum(x > 0.01 for x in d) / n if n else None, "frac_gt_0.1": sum(x > 0.1 for x in d) / n if n else None,
           "frac_gt_1.0": sum(x > 1.0 for x in d) / n if n else None, "per_prompt_max": per,
           "bar": {"p99_le_0.05": q(0.99) <= 0.05, "max_le_0.5": (d[-1] if n else 9) <= 0.5, "tokens_ge_3000": n >= 3000}}
    out["NONINFERIOR"] = all(out["bar"].values())
    print(json.dumps(out, indent=1)); sys.exit(0 if out["NONINFERIOR"] else 1)

base = os.getenv("BASE_URL", "http://127.0.0.1:30001").rstrip("/")
model = os.getenv("MODEL", "glm-5.3-big")
H = {"Authorization": f"Bearer {os.environ['API_KEY']}", "Content-Type": "application/json"}
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from greedy_equiv import PROMPTS  # noqa: E402

def post(path, payload):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(), headers=H)
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)

def render(prompt):
    # same chat render the greedy run used: chat template applied server-side via /v1/chat/completions is not
    # echoable, so we use the tokenizer-rendered prompt via /tokenize with the same messages + effort=low.
    t = post("/tokenize", {"model": model, "messages": [{"role": "user", "content": prompt}], "add_generation_prompt": True,
                            "chat_template_kwargs": {"reasoning_effort": "low"}})
    return t["tokens"]

ref = json.load(open(sys.argv[1]))
out = {"server_lane": os.getenv("LANE", "?"), "ref": sys.argv[1], "prompts": {}}
tot = 0
for k in sorted(ref, key=int):
    text = ref[k]
    ptoks = render(PROMPTS[int(k)])
    gt = post("/tokenize", {"model": model, "prompt": text, "add_special_tokens": False})["tokens"]
    r = post("/v1/completions", {"model": model, "prompt": ptoks + gt, "max_tokens": 1, "temperature": 0, "echo": True, "logprobs": 1})
    lp = r["choices"][0]["logprobs"]["token_logprobs"]
    gen_lp = [x for x in lp[len(ptoks):len(ptoks) + len(gt)] if x is not None]
    out["prompts"][k] = {"gen_tokens": len(gt), "logprobs": gen_lp}
    tot += len(gen_lp)
    print(f"prompt {k}: {len(gen_lp)} scored tokens", flush=True)
# instrument self-check: rescore prompt 0 and require byte-identical
k0 = sorted(ref, key=int)[0]; ptoks = render(PROMPTS[int(k0)])
gt = post("/tokenize", {"model": model, "prompt": ref[k0], "add_special_tokens": False})["tokens"]
r = post("/v1/completions", {"model": model, "prompt": ptoks + gt, "max_tokens": 1, "temperature": 0, "echo": True, "logprobs": 1})
lp2 = [x for x in r["choices"][0]["logprobs"]["token_logprobs"][len(ptoks):len(ptoks) + len(gt)] if x is not None]
out["instrument_repeat_identical"] = lp2 == out["prompts"][k0]["logprobs"]
out["total_scored_tokens"] = tot
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(f"SCORED {tot} tokens on {out['server_lane']}; instrument repeat-identical: {out['instrument_repeat_identical']}")
