#!/usr/bin/env python3
"""Warm + measure steady-state decode at C1/C4/C8 (streaming, first-delta TTFT excluded)."""
import json, os, time, threading, urllib.request

base = os.getenv("BASE_URL", "http://127.0.0.1:30001/v1").rstrip("/")
model = os.getenv("MODEL", "glm-5.3-big")
key = os.environ["API_KEY"]
H = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
NONCE = str(int(time.time()))

def one(idx, max_tokens, out):
    payload = {"model": model, "stream": True, "temperature": 0, "max_tokens": max_tokens,
               "stream_options": {"include_usage": True},
               "chat_template_kwargs": {"reasoning_effort": "low"},
               "messages": [{"role": "user", "content": f"[{NONCE}-{idx}] Write a detailed essay on the history of the Roman aqueducts, at least 600 words."}]}
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(payload).encode(), headers=H)
    t0 = time.monotonic(); first = None; n = 0; usage = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"): continue
            d = json.loads(line[5:])
            if d.get("usage"): usage = d["usage"]
            ch = d.get("choices") or []
            if ch and (ch[0].get("delta") or {}):
                delta = ch[0]["delta"]
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                    if first is None: first = time.monotonic()
                    n += 1
    t1 = time.monotonic()
    toks = usage["completion_tokens"] if usage else n
    out.append({"ttft": round((first or t1) - t0, 2), "tokens": toks, "decode_s": round(t1 - (first or t0), 2)})

def run(c, max_tokens):
    out = []; ts = [threading.Thread(target=one, args=(i, max_tokens, out)) for i in range(c)]
    t0 = time.monotonic(); [t.start() for t in ts]; [t.join() for t in ts]; wall = time.monotonic() - t0
    tot = sum(o["tokens"] for o in out)
    per = [o["tokens"] / o["decode_s"] for o in out if o["decode_s"] > 0]
    return {"C": c, "agg_tok_s": round(tot / wall, 2), "per_stream_tok_s": round(sum(per) / len(per), 2),
            "mean_ttft": round(sum(o["ttft"] for o in out) / len(out), 2), "tokens_total": tot}

# warm each shape
for c in (1, 4, 8):
    run(c, 64)
for c in (1, 4, 8):
    print("BENCH", json.dumps(run(c, 512)), flush=True)
