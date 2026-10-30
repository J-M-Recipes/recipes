#!/usr/bin/env python3
"""caller_probe.py <base_url> <model> <out.json> — what a caller sees on a live OpenAI-compatible lane (stdlib only).

Six measures, chat completions, streaming, thinking on the server's default budget (no chat_template_kwargs),
temperature 0, from a LAN client:
  1 single stream, 400-token reply: decode tok/s = (tokens - 1) / (last chunk - first chunk), median of 3
  2 40-token turn, complete: wall time request -> last byte, median of 5
  3 8 concurrent streams x 400 tokens: aggregate tok/s = all tokens / (last chunk - first send), 3 reps
  4 first token, idle: TTFT of a short prompt, median of 5
  5 quiet single-stream step: median inter-chunk gap of measure 1 (a chunk = one engine step; spec decode emits
    several tokens per step)
  6 worst-10 % inter-chunk gap (p90) of a single 1,500-token stream, counting only gaps inside the window in which
    a fresh ~33K-token prompt is being read (send -> its first token), plus that prompt's TTFT and the same
    stream's p90 gap with no concurrent read (control); 3 reps, median
Token counts come from the server's usage block (stream_options.include_usage). Unique nonces defeat prefix cache.
"""
import json, random, statistics as st, sys, threading, time, urllib.request

BASE, MODEL, OUT = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]
TOPICS = ["the history of the printing press", "how bridges carry load", "the water cycle", "how compilers work",
          "the life of honeybees", "how GPS finds a position", "the economics of shipping containers",
          "how vaccines train the immune system"]


def stream(prompt, max_tokens, t_send=None):
    body = json.dumps({"model": MODEL, "stream": True, "temperature": 0, "max_tokens": max_tokens,
                       "stream_options": {"include_usage": True},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = t_send or time.time(); times = []; usage = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices") or []:
                delta = c.get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                    times.append(time.time())
    t_end = time.time()
    toks = (usage or {}).get("completion_tokens") or len(times)
    gaps = [b - a for a, b in zip(times, times[1:])]
    return {"t_send": t0, "ttft": (times[0] - t0) if times else None, "first": times[0] if times else None,
            "last": times[-1] if times else None, "end": t_end, "tokens": toks, "chunks": len(times), "gaps": gaps,
            "prompt_tokens": (usage or {}).get("prompt_tokens"), "times": times}


def nonce():
    return f"[{random.getrandbits(48):012x}] "


def p(xs, q):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


res = {"base": "<lane>", "model": MODEL, "started": time.strftime("%F %T %Z")}
# 1 + 5
s1 = [stream(nonce() + f"Write about {TOPICS[i]} in plain prose. Keep going.", 400) for i in range(3)]
rates = [(r["tokens"] - 1) / (r["last"] - r["first"]) for r in s1]
res["m1_single_stream_400_tok_s"] = {"median": st.median(rates), "reps": rates, "tokens": [r["tokens"] for r in s1]}
res["m5_quiet_step_ms"] = {"median_gap_ms": 1000 * st.median([g for r in s1 for g in r["gaps"]]),
                           "p90_gap_ms": 1000 * p([g for r in s1 for g in r["gaps"]], 0.9)}
# 2
walls = []
for i in range(5):
    t0 = time.time(); r = stream(nonce() + "Say hello and name one fruit.", 40); walls.append(r["end"] - t0)
res["m2_40_token_turn_s"] = {"median": st.median(walls), "reps": walls}
# 4
tt = [stream(nonce() + "Reply with one word: ok", 8)["ttft"] for _ in range(5)]
res["m4_first_token_idle_s"] = {"median": st.median(tt), "reps": tt}
# 3
aggs = []
for rep in range(3):
    out = [None] * 8; t0 = time.time()
    def run(i):
        out[i] = stream(nonce() + f"Write about {TOPICS[i]} in plain prose. Keep going.", 400, t0)
    th = [threading.Thread(target=run, args=(i,)) for i in range(8)]; [t.start() for t in th]; [t.join() for t in th]
    tot = sum(r["tokens"] for r in out); span = max(r["last"] for r in out) - t0
    aggs.append({"aggregate_tok_s": tot / span, "per_stream_tok_s": [(r["tokens"] - 1) / (r["last"] - r["first"]) for r in out]})
res["m3_8_streams"] = {"median_aggregate": st.median(a["aggregate_tok_s"] for a in aggs), "reps": aggs}
# 6  (fixed 2026-10-30: victim long enough to straddle the read; gaps counted only inside the read window)
filler = " ".join(f"Section {k}: the committee reviewed item {k * 7 % 997} and recorded its findings in the ledger."
                  for k in range(1780))
def windowed(v, lo, hi):
    ts = v["times"]; return [b - a for a, b in zip(ts, ts[1:]) if b > lo and a < hi]
reps = []
for rep in range(3):
    ctl = stream(nonce() + f"Write about {TOPICS[5]} in plain prose. Keep going.", 1500)
    box = {}
    def long_read():
        box["long"] = stream(nonce() + filler + "\n\nIn one sentence, what did the committee do?", 32)
    def victim():
        box["victim"] = stream(nonce() + f"Write about {TOPICS[6]} in plain prose. Keep going.", 1500)
    tv = threading.Thread(target=victim); tv.start(); time.sleep(1.0)
    tl = threading.Thread(target=long_read); tl.start(); tl.join(); tv.join()
    v, L = box["victim"], box["long"]
    w = windowed(v, L["t_send"], L["first"])
    assert v["last"] > L["first"], "victim ended before the read finished"
    reps.append({"window_gaps": len(w), "window_p90_ms": 1000 * p(w, 0.9), "window_max_ms": 1000 * max(w),
                 "window_median_ms": 1000 * st.median(w), "control_p90_ms": 1000 * p(ctl["gaps"], 0.9),
                 "long_prompt_tokens": L["prompt_tokens"], "long_ttft_s": L["ttft"]})
res["m6_gap_during_33k_read"] = {"median_window_p90_ms": st.median(r["window_p90_ms"] for r in reps),
                                 "median_long_ttft_s": st.median(r["long_ttft_s"] for r in reps), "reps": reps}
res["ended"] = time.strftime("%F %T %Z")
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k.startswith("m")}, indent=1, default=lambda x: round(x, 3)))
