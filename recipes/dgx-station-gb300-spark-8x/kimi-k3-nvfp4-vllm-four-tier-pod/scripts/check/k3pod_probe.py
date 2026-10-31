#!/usr/bin/env python3
"""Ask a running K3 server for prompt logprobs + a short greedy continuation; save JSON.
Usage: k3pod_probe.py PORT OUT.json"""
import json, sys, urllib.request

PORT, OUT = int(sys.argv[1]), sys.argv[2]
PROMPTS = [
    "The history of the transistor begins",
    "Kimi K3 runs across one DGX Station and eight DGX Sparks. The experts",
    "def fibonacci(n):\n    ",
]
res = []
for p in PROMPTS:
    body = {"model": "k3", "prompt": p, "max_tokens": 8, "temperature": 0, "logprobs": 20,
            "prompt_logprobs": 20}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    c = d["choices"][0]
    res.append({"prompt": p, "text": c["text"], "prompt_logprobs": c.get("prompt_logprobs"),
                "top_logprobs": c["logprobs"]["top_logprobs"], "tokens": c["logprobs"]["tokens"]})
    print(repr(p[:40]), "->", repr(c["text"]))
json.dump(res, open(OUT, "w"))
print("wrote", OUT)
