#!/usr/bin/env python3
"""speed_reps.py OUT.json — C1 speed instrument per campaign skill:
8 prompts x 512 tok, temp 0, streamed. Rep 0 = warm (discarded), then 3 scored reps.
Per prompt: median decode tok/s and spread% across scored reps. Overall: median of medians.
"""
import json, os, sys, time, statistics, urllib.request

base = os.getenv("BASE_URL", "http://127.0.0.1:30001").rstrip("/")
model = os.getenv("MODEL", "glm-5.3-big")
H = {"Authorization": f"Bearer {os.environ['API_KEY']}", "Content-Type": "application/json"}
MAXTOK = int(os.getenv("MAX_TOKENS", "512")); REPS = int(os.getenv("REPS", "3"))
PROMPTS = [
    ("prose", "Describe a lighthouse keeper's last night on duty, in vivid detail."),
    ("prose", "Explain how a mixture-of-experts transformer routes tokens, for a curious engineer."),
    ("code", "Write a Python module implementing an LRU cache with TTL, with docstrings and a small test."),
    ("code", "Write a Rust function that parses a simple INI file into a HashMap, with error handling."),
    ("reason", "A train leaves at 3:10 pm going 60 mph; another leaves the same station at 3:40 pm going 80 mph. When does the second catch the first? Show all steps."),
    ("reason", "List and justify four ways to reduce p95 latency in a Python web service under bursty load."),
    ("list", "Give twenty distinct startup ideas in the home-energy space, one line each with a one-clause risk."),
    ("dialog", "Write a short dialogue between a skeptical CFO and an engineer proposing a $2M GPU purchase."),
]


def run(q):
    p = {"model": model, "max_tokens": MAXTOK, "temperature": 0, "stream": True,
         "stream_options": {"include_usage": True},
         "messages": [{"role": "user", "content": q}]}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(p).encode(), headers=H)
    t0 = time.time(); ttft = None; usage = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            if not line.startswith(b"data:"):
                continue
            s = line[5:].strip()
            if s == b"[DONE]":
                break
            d = json.loads(s)
            if d.get("usage"):
                usage = d["usage"]
            ch = d.get("choices") or []
            if ttft is None and ch:
                delta = ch[0].get("delta") or {}
                if any(delta.get(key) for key in ("content", "reasoning_content", "reasoning")):
                    ttft = time.time() - t0
    wall = time.time() - t0
    ct = usage["completion_tokens"] if usage else None
    ttft = ttft or 0.0
    return {"wall": wall, "ttft": ttft, "completion_tokens": ct, "prompt_tokens": usage["prompt_tokens"] if usage else None,
            "decode_tok_s": (ct - 1) / (wall - ttft) if ct and wall > ttft else None}


reps = []
for rep in range(REPS + 1):
    row = []
    for kind, q in PROMPTS:
        r = run(q); r["kind"] = kind; row.append(r)
        print(f"rep{rep} {kind:>6} ttft={r['ttft']:.3f}s ct={r['completion_tokens']} tok/s={r['decode_tok_s']:.2f}", flush=True)
    reps.append(row)
scored = reps[1:]
per = []
for i, (kind, q) in enumerate(PROMPTS):
    v = [rep[i]["decode_tok_s"] for rep in scored]
    t = [rep[i]["ttft"] for rep in scored]
    per.append({"kind": kind, "median_tok_s": statistics.median(v), "spread_pct": 100 * (max(v) - min(v)) / statistics.median(v),
                "ttft_median": statistics.median(t), "reps": v})
overall = statistics.median([p["median_tok_s"] for p in per])
summary = {"lane": os.getenv("LANE", "?"), "max_tokens": MAXTOK, "scored_reps": REPS, "prompts": len(PROMPTS),
           "c1_median_tok_s": overall, "max_spread_pct": max(p["spread_pct"] for p in per),
           "ttft_median": statistics.median(p["ttft_median"] for p in per)}
for p in per:
    print(f"{p['kind']:>6} median={p['median_tok_s']:.2f} spread={p['spread_pct']:.1f}% ttft={p['ttft_median']:.3f}")
print("SUMMARY", json.dumps(summary))
json.dump({"summary": summary, "per_prompt": per, "raw": reps}, open(sys.argv[1], "w"), indent=1)
