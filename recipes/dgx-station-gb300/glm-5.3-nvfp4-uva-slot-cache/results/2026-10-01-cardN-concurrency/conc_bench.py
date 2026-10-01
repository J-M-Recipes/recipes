#!/usr/bin/env python3
"""conc_bench.py OUT.json — concurrency instrument for the big-GLM slot-cache lane.
For each C in CONCS: rep 0 warm (discarded), then REPS scored reps of C simultaneous streamed 512-token
requests (distinct prompts, nonce-suffixed so the prefix cache cannot help). Reports aggregate tok/s,
per-stream tok/s (median across streams), TTFT median, and spread across reps. Stdlib only.
"""
import json, os, sys, time, threading, statistics, uuid, urllib.request

base = os.getenv("BASE_URL", "http://127.0.0.1:30001").rstrip("/")
model = os.getenv("MODEL", "glm-5.3-big")
H = {"Authorization": f"Bearer {os.environ['API_KEY']}", "Content-Type": "application/json"}
MAXTOK = int(os.getenv("MAX_TOKENS", "512")); REPS = int(os.getenv("REPS", "3"))
CONCS = [int(x) for x in os.getenv("CONCS", "1,2,4").split(",")]
LANE = os.getenv("LANE", "x")
PROMPTS = [
    "Describe a lighthouse keeper's last night on duty, in vivid detail.",
    "Write a Python module implementing an LRU cache with TTL, with docstrings and a small test.",
    "A train leaves at 3:10 pm going 60 mph; another leaves the same station at 3:40 pm going 80 mph. When does the second catch the first? Show all steps.",
    "Write a short dialogue between a skeptical CFO and an engineer proposing a $2M GPU purchase.",
    "Explain how a mixture-of-experts transformer routes tokens, for a curious engineer.",
    "Write a Rust function that parses a simple INI file into a HashMap, with error handling.",
    "List and justify four ways to reduce p95 latency in a Python web service under bursty load.",
    "Give twenty distinct startup ideas in the home-energy space, one line each with a one-clause risk.",
]


def one(q, out, i):
    p = {"model": model, "max_tokens": MAXTOK, "temperature": 0, "stream": True,
         "stream_options": {"include_usage": True},
         "messages": [{"role": "user", "content": q + f"\n[{uuid.uuid4().hex[:8]}]"}]}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(p).encode(), headers=H)
    t0 = time.time(); ttft = None; n = None
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for line in r:
                if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                    continue
                d = json.loads(line[6:])
                if ttft is None and d.get("choices") and (d["choices"][0].get("delta") or {}).get("content"):
                    ttft = time.time() - t0
                if d.get("usage"):
                    n = d["usage"].get("completion_tokens")
        t1 = time.time()
        if ttft is None: ttft = t1 - t0
        out[i] = {"ok": True, "tokens": n or 0, "wall": t1 - t0, "ttft": ttft,
                  "decode_tps": ((n or 0) / max(t1 - t0 - ttft, 1e-6))}
    except Exception as e:  # noqa
        out[i] = {"ok": False, "err": str(e)[:200], "tokens": 0, "wall": time.time() - t0, "ttft": 0, "decode_tps": 0}


res = {"lane": LANE, "max_tokens": MAXTOK, "reps": REPS, "concs": {}}
for C in CONCS:
    reps = []
    for rep in range(REPS + 1):
        out = [None] * C
        th = [threading.Thread(target=one, args=(PROMPTS[i % len(PROMPTS)], out, i)) for i in range(C)]
        t0 = time.time(); [t.start() for t in th]; [t.join() for t in th]; wall = time.time() - t0
        toks = sum(o["tokens"] for o in out)
        row = {"agg_tps": toks / wall, "per_stream_median": statistics.median(o["decode_tps"] for o in out),
               "ttft_median": statistics.median(o["ttft"] for o in out), "wall": wall, "tokens": toks,
               "errors": sum(1 for o in out if not o["ok"]), "warm": rep == 0}
        print(f"[{LANE}] C{C} rep{rep}{' (warm)' if rep == 0 else ''}: agg {row['agg_tps']:.1f} tok/s  per-stream {row['per_stream_median']:.1f}  ttft {row['ttft_median']:.2f}s  errors {row['errors']}", flush=True)
        reps.append(row)
    scored = [r for r in reps if not r["warm"]]
    aggs = [r["agg_tps"] for r in scored]
    res["concs"][str(C)] = {"reps": reps,
                            "agg_median": statistics.median(aggs),
                            "spread_pct": (max(aggs) - min(aggs)) / statistics.median(aggs) * 100,
                            "per_stream_median": statistics.median(r["per_stream_median"] for r in scored),
                            "ttft_median": statistics.median(r["ttft_median"] for r in scored),
                            "errors": sum(r["errors"] for r in scored)}
    s = res["concs"][str(C)]
    print(f"[{LANE}] C{C} SCORED: agg {s['agg_median']:.1f} (spread {s['spread_pct']:.0f}%) per-stream {s['per_stream_median']:.1f} ttft {s['ttft_median']:.2f}s errors {s['errors']}", flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
