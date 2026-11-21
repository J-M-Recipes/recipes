#!/usr/bin/env python3
"""Determinism check: 32 concurrent greedy requests (distinct prompts, logprobs) run twice back to back.
At C32 each MoE call carries 32x(1+3)=128 rows -> sidecar >64-row (v2/b12x) path. Compares text + per-token logprobs.
usage: det_check.py ROOT OUT.json"""
import json, sys, threading, urllib.request
root, outp = sys.argv[1], sys.argv[2]
TOPICS = ["volcanoes", "the printing press", "honeybees", "black holes", "the Roman aqueducts", "jazz", "photosynthesis",
          "the Silk Road", "glaciers", "the steam engine", "octopuses", "the immune system", "chess openings", "tides",
          "the Apollo program", "coral reefs", "bread baking", "the Great Wall", "hurricanes", "radio", "penguins",
          "the Renaissance", "earthquakes", "vaccines", "the Internet", "rainforests", "clocks", "the Moon", "bridges",
          "salt", "elephants", "lightning"]
def run():
    res = [None] * 32
    def one(i):
        body = {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": f"Write three paragraphs about {TOPICS[i]}."}],
                "max_tokens": 160, "temperature": 0, "logprobs": True, "chat_template_kwargs": {"thinking": False}}
        r = urllib.request.Request(root + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        d = json.loads(urllib.request.urlopen(r, timeout=600).read())["choices"][0]
        res[i] = (d["message"]["content"], [t["logprob"] for t in (d.get("logprobs") or {}).get("content") or []])
    ts = [threading.Thread(target=one, args=(i,)) for i in range(32)]; [t.start() for t in ts]; [t.join() for t in ts]
    return res
a, b = run(), run()
same_text = sum(1 for x, y in zip(a, b) if x[0] == y[0])
same_lp = 0; maxd = 0.0; ndiff = 0; ntok = 0
for x, y in zip(a, b):
    n = min(len(x[1]), len(y[1])); ntok += n
    if x[0] == y[0] and x[1] == y[1]: same_lp += 1
    for p, q in zip(x[1][:n], y[1][:n]):
        if p != q: ndiff += 1; maxd = max(maxd, abs(p - q))
r = {"identical_text": same_text, "identical_text_and_logprobs": same_lp, "of": 32, "tokens_compared": ntok,
     "logprob_diffs": ndiff, "max_abs_dlogprob": maxd}
json.dump({"summary": r, "a": a, "b": b}, open(outp, "w")); print(json.dumps(r))
