#!/usr/bin/env python3
"""R2 greedy identity probe: 40 raw-completion prompts, T=0, 128 tokens, single stream (greedy_agree.py prompts).
usage: greedy_probe.py ROOT OUT.json"""
import json, os, sys, urllib.request
topics = ["the history of the Roman aqueducts", "how a transformer attention head works", "a recipe for sourdough bread", "why the sky is blue", "the rules of chess for a beginner", "a haiku sequence about winter", "the plot of a heist movie", "how to change a bicycle tire", "the economics of solar power", "a letter of complaint to an airline", "Python code to parse a CSV file", "the difference between TCP and UDP", "a product description for a hiking boot", "the causes of the French Revolution", "a bedtime story about a fox", "how vaccines train the immune system", "the geology of the Grand Canyon", "a SQL query explanation for a join", "the life cycle of a star", "tips for learning Spanish"]
prompts = []
for t in topics:
    prompts.append(f"Write a short, detailed explanation of {t}.\n\n")
    prompts.append(f"Here are five surprising facts about {t}:\n1.")
root, out = sys.argv[1], sys.argv[2]
res = []
for p in prompts:
    body = {"model": os.environ.get("MODEL", "deepseek-v4.1-flash-spark-vllm"), "prompt": p, "max_tokens": 128, "temperature": 0, "seed": 1}
    req = urllib.request.Request(root + "/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    res.append(d["choices"][0]["text"])
json.dump({"root": root, "prompts": prompts, "outputs": res}, open(out, "w"))
print("wrote", out, len(res))
