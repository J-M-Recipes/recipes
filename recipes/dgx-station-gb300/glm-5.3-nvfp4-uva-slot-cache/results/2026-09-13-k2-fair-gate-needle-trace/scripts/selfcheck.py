#!/usr/bin/env python3
"""selfcheck.py X.json — score lane X's own greedy text on the live server; report fraction of
tokens at rank 1 (greedy-faithfulness of the render+prefill path). Tests /tokenize with and
without reasoning_effort=low to see whether the template changes."""
import json, os, sys, urllib.request
base = os.getenv("BASE_URL", "http://127.0.0.1:30001").rstrip("/"); model = os.getenv("MODEL", "glm-5.3-big")
H = {"Authorization": f"Bearer {os.environ['API_KEY']}", "Content-Type": "application/json"}
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from greedy_equiv import PROMPTS
def post(path, payload):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(), headers=H)
    with urllib.request.urlopen(req, timeout=600) as r: return json.load(r)
msgs = [{"role": "user", "content": PROMPTS[0]}]
t0 = post("/tokenize", {"model": model, "messages": msgs, "add_generation_prompt": True})["tokens"]
t1 = post("/tokenize", {"model": model, "messages": msgs, "add_generation_prompt": True, "chat_template_kwargs": {"reasoning_effort": "low"}})["tokens"]
print("gen-prompt tokens: default", len(t0), "| effort=low", len(t1), "| identical:", t0 == t1)
d = post("/detokenize", {"model": model, "tokens": t1}); print("RENDER(low):", repr(d.get("prompt"))[:400])
X = json.load(open(sys.argv[1])); tot = r1 = 0; worst = []
for k in sorted(X, key=int)[:int(os.getenv("N", "6"))]:
    reasoning, _, content = X[k].partition("\u241f"); m = [{"role": "user", "content": PROMPTS[int(k)]}]
    kw = {"chat_template_kwargs": {"reasoning_effort": "low"}}
    pre = post("/tokenize", {"model": model, "messages": m, "add_generation_prompt": True, **kw})["tokens"]
    full = post("/tokenize", {"model": model, "messages": m + [{"role": "assistant", "reasoning_content": reasoning, "content": content}], "add_generation_prompt": False, **kw})["tokens"]
    assert full[:len(pre)] == pre
    lp = post("/v1/completions", {"model": model, "prompt": full, "max_tokens": 1, "echo": True, "logprobs": 1, "temperature": 0})["choices"][0]["prompt_logprobs"]
    gen = lp[len(pre):]; ranks = [next(iter(x.values()))["rank"] for x in gen]
    n1 = sum(r == 1 for r in ranks); tot += len(ranks); r1 += n1
    bad = [(i, r) for i, r in enumerate(ranks) if r != 1]
    print(f"prompt {k:>2}: {n1}/{len(ranks)} rank1; non-rank1 at {bad[:8]}")
print(f"TOTAL rank1 {r1}/{tot} = {r1/tot:.3f}")
