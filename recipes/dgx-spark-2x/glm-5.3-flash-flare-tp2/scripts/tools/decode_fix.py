#!/usr/bin/env python3
"""Aug-29 quant_faceoff decode fixtures, thinking-aware: rate = completion_tokens / (end - first delta of ANY kind).
GLM-5.3-Flash's template always thinks (enable_thinking is a no-op), so quant_faceoff's content-only TTFT breaks decode."""
import json, sys, time, urllib.request, statistics
B = sys.argv[1]; OUT = sys.argv[2]
M = json.load(urllib.request.urlopen(B + "/models"))["data"][0]["id"]
FIX = {"structured_count": "Print integers 1 through 300 separated only by commas. Do not omit any integer.",
       "prose": "Write 450 words explaining why measurement methodology matters in systems engineering. Plain prose, no headings or lists."}
def run(prompt, seed, effort=None):
    body = {"model": M, "messages": [{"role": "user", "content": prompt}], "max_tokens": 4096, "temperature": 0.0, "seed": seed,
            "stream": True, "stream_options": {"include_usage": True}}
    if effort: body["reasoning_effort"] = effort
    req = urllib.request.Request(B + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = firstc = None; usage = {}; rc = c = 0
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            if not raw.startswith(b"data: "): continue
            p = raw[6:].strip()
            if p == b"[DONE]": break
            d = json.loads(p)
            if d.get("usage"): usage = d["usage"]
            for ch in d.get("choices", []):
                dd = ch.get("delta", {})
                if dd.get("reasoning_content"):
                    rc += 1; first = first or time.perf_counter()
                if dd.get("content"):
                    c += 1; first = first or time.perf_counter(); firstc = firstc or time.perf_counter()
    t1 = time.perf_counter(); ct = usage.get("completion_tokens", 0)
    return {"completion_tokens": ct, "ttft_any_s": round((first or t1) - t0, 3), "first_content_s": round((firstc or t1) - t0, 3),
            "total_s": round(t1 - t0, 3), "decode_tok_s": round(ct / max(1e-4, t1 - (first or t1)), 1), "reasoning_deltas": rc, "content_deltas": c}
rows = []
for effort in (None, "low"):
    for name, prompt in FIX.items():
        for run_i in range(2):
            r = run(prompt, 9000 + run_i, effort); r.update(fixture=name, run=run_i + 1, effort=effort or "default(max)"); rows.append(r); print(json.dumps(r), flush=True)
summ = {}
for effort in ("default(max)", "low"):
    for name in FIX:
        v = [x["decode_tok_s"] for x in rows if x["fixture"] == name and x["effort"] == effort]
        summ[f"{name}|{effort}"] = round(statistics.median(v), 1)
json.dump({"rows": rows, "summary": summ}, open(OUT, "w"), indent=1)
print("RESULT", json.dumps(summ))
