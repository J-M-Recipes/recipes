#!/usr/bin/env python3
"""long_greedy.py <outfile> — long-generation greedy set for the DFlash/KDA checkpoint bug (sglang #37817/#37818).
8 prompts that reliably run long (effort low, max_tokens 2500, T=0, ignore_eos off). Saves {idx: {"text","tokens","tok_s"}}.
Compare: long_greedy.py --compare a.json b.json  -> identical count, first-divergence token index per prompt.
The bug: DFlash misses a Mamba/KDA state checkpoint when accepted tokens cross a tracking boundary (interval 64/128/256),
leaving full-attn KV and linear state at different positions. Signature = DFlash text diverging from AR text (same image,
same weights, T=0) at some point after the first few hundred tokens, on the pinned image; and NOT diverging on v0.5.20.
Env: BASE_URL MODEL MAXTOK(2500) EFFORT(low)"""
import json, os, sys, time, urllib.request
if sys.argv[1] == "--compare":
    a, b = json.load(open(sys.argv[2])), json.load(open(sys.argv[3])); same = 0; rows = []
    for k in sorted(a, key=int):
        x, y = a[k]["text"], b.get(k, {}).get("text", "")
        if x == y: same += 1; rows.append((int(k), -1, a[k]["tokens"], b.get(k, {}).get("tokens"))); continue
        i = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
        rows.append((int(k), i, a[k]["tokens"], b.get(k, {}).get("tokens")))
    print(f"LONG_GREEDY identical={same}/{len(a)}")
    for k, i, la, lb in rows: print(f"  prompt {k}: {'identical' if i < 0 else f'first divergence at char {i}'} (tokens {la} vs {lb})")
    sys.exit(0)
BASE = os.getenv("BASE_URL", "http://127.0.0.1:30001/v1").rstrip("/"); MODEL = os.getenv("MODEL", "glm-5.3-flash")
MAXTOK = int(os.getenv("MAXTOK", "2500")); EFFORT = os.getenv("EFFORT", "low")
PROMPTS = [
 "Write a complete, self-contained tutorial (at least 1500 words) on building a small HTTP server in Rust with tokio, including full code listings, error handling, and a section on testing.",
 "Explain, in exhaustive detail with worked numerical examples, how a RAID6 array reconstructs data after two simultaneous disk failures. Cover Reed-Solomon arithmetic over GF(2^8) step by step.",
 "Write a long technical design document for a persistent, crash-safe job queue on top of SQLite: schema, state machine, locking, idempotency, retries, and a full Python implementation.",
 "Produce a detailed, chapter-by-chapter outline and then the full first chapter (2000+ words) of a novel about an engineer who maintains a mountain-town water system.",
 "Give a complete derivation of the backpropagation equations for a two-layer MLP with softmax cross-entropy, then implement it in numpy with a training loop and unit tests.",
 "Write an extensive comparison (1500+ words) of speculative decoding methods: draft models, MTP heads, block diffusion drafts, and self-speculation; include acceptance-rate math and when each wins.",
 "Write a full bash script (with comments explaining every section) that safely rotates and archives logs across a fleet of hosts over ssh, with dry-run mode, locking, and a summary report.",
 "Explain the TCP congestion control algorithms Reno, CUBIC and BBR in depth, with state diagrams described in prose, worked examples of window evolution, and a discussion of fairness.",
]
out = {}
for i, p in enumerate(PROMPTS):
    body = {"model": MODEL, "messages": [{"role": "user", "content": p}], "temperature": 0, "max_tokens": MAXTOK,
            "chat_template_kwargs": {"reasoning_effort": EFFORT}}
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"{BASE}/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=1800))
    dt = time.time() - t0; ch = r["choices"][0]; txt = (ch["message"].get("reasoning_content") or "") + "\n<<<CONTENT>>>\n" + (ch["message"].get("content") or "")
    n = r["usage"]["completion_tokens"]; out[str(i)] = {"text": txt, "tokens": n, "tok_s": round(n / dt, 1), "finish": ch.get("finish_reason")}
    print(f"[{i}] tokens={n} {n/dt:.1f} tok/s finish={ch.get('finish_reason')}", flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1); print(f"LONG_GREEDY saved {len(out)} to {sys.argv[1]}")
