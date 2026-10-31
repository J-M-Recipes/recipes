#!/usr/bin/env python3
"""G1 decode timer: per-token decode time of a running server at batch 1.

Two independent methods, which must agree:
  A  wall difference: (t[max_tokens=1+D] - t[max_tokens=1]) / D, ignore_eos, temperature 0.
     Immune to stream chunking; includes the scheduler/detokenizer loop per step.
  B  streaming: median inter-chunk gap over D tokens.
Usage: g1_bench.py PORT [D] [reps]
"""
import json, statistics, sys, time, urllib.request

PORT = int(sys.argv[1]); D = int(sys.argv[2]) if len(sys.argv) > 2 else 256
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
URL = f"http://127.0.0.1:{PORT}/v1/completions"
PROMPT = "The history of the transistor begins"


def post(max_tokens, stream=False):
    body = json.dumps({"model": "k3", "prompt": PROMPT, "max_tokens": max_tokens,
                       "temperature": 0, "ignore_eos": True, "stream": stream}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        if not stream:
            out = json.load(r)
            return time.perf_counter() - t0, out["usage"]["completion_tokens"], out["choices"][0]["text"]
        stamps = []
        for line in r:
            if line.startswith(b"data: ") and not line.startswith(b"data: [DONE]"):
                stamps.append(time.perf_counter())
        return stamps


post(16); post(16)                                     # warm
a = []
for _ in range(REPS):
    t1, n1, _ = post(1)
    tD, nD, text = post(1 + D)
    assert nD == 1 + D, f"got {nD} tokens"
    a.append((tD - t1) / D * 1e3)
gaps = []
for _ in range(max(2, REPS // 2)):
    s = post(1 + D, stream=True)
    gaps += [(y - x) * 1e3 for x, y in zip(s[1:], s[2:])]
res = {"port": PORT, "D": D, "reps": REPS,
       "A_ms_per_token_median": round(statistics.median(a), 4), "A_all": [round(x, 4) for x in a],
       "B_gap_ms_median": round(statistics.median(gaps), 4),
       "B_gap_ms_p10": round(statistics.quantiles(gaps, n=10)[0], 4),
       "B_gap_ms_p90": round(statistics.quantiles(gaps, n=10)[-1], 4),
       "sample_text": text[:160]}
print(json.dumps(res, indent=1))
