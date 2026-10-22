#!/usr/bin/env python3
"""Quick smoke + decode-rate probe for an OpenAI-compatible lane (stdlib only)."""
import json, sys, time, urllib.request, concurrent.futures as cf
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:30007/v1"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "GLM-5.3-Flash"
C = int(sys.argv[3]) if len(sys.argv) > 3 else 1
MAXTOK = int(sys.argv[4]) if len(sys.argv) > 4 else 512
PROMPT = sys.argv[5] if len(sys.argv) > 5 else "Write a detailed, multi-paragraph explanation of how a CPU branch predictor works, including history tables and misprediction costs."

def one(i):
    body = {"model": MODEL, "messages": [{"role": "user", "content": f"[{i}] " + PROMPT}],
            "max_tokens": MAXTOK, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None; n = 0; text = []; usage = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            ev = json.loads(line[5:])
            if ev.get("usage"): usage = ev["usage"]
            for ch in ev.get("choices", []):
                dd = ch.get("delta", {}); d = (dd.get("content") or "") + (dd.get("reasoning_content") or "")
                if d:
                    if first is None: first = time.time()
                    n += 1; text.append(d)
    t1 = time.time()
    comp = usage["completion_tokens"] if usage else n
    ttft = (first - t0) if first else None
    dec = comp / (t1 - first) if first and t1 > first else 0
    return {"i": i, "ttft": ttft, "completion_tokens": comp, "decode_tok_s": dec, "wall": t1 - t0, "chunks": n, "head": "".join(text)[:160]}

with cf.ThreadPoolExecutor(C) as ex:
    res = list(ex.map(one, range(C)))
for r in res: print(json.dumps(r))
agg = sum(r["decode_tok_s"] for r in res)
print(f"C={C} per-stream mean {agg/C:.1f} tok/s, aggregate {agg:.1f} tok/s, ttft mean {sum(r['ttft'] or 0 for r in res)/C:.2f}s")
