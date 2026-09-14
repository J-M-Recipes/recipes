#!/usr/bin/env python3
"""depth_knee.py (v2) — single-stream DECODE tok/s vs prompt depth on the :30006 vLLM lane, cache-independent.

v1 relied on a prefix-cache hit for the second call and silently included a re-prefill at some depths
(106K and 425K reconstructed exactly as prefill+decode). v2 streams and times FIRST token -> LAST token,
so the number is pure decode regardless of cache state. TTFT (prefill) is reported separately from the same request.

Per depth: 1 warm request (max_tokens=1) then REPS streamed requests of MAXTOK tokens (ignore_eos).
decode_tok_s = (tokens_after_first) / (t_last - t_first). Writes /home/milo/dsv41/depth-<tag>.json.

Env: TAG (required), DEPTHS ("8000 65536 131072 262144 524288"), REPS (2), MAXTOK (192).
"""
import json, os, random, time, urllib.request, uuid

B = "http://127.0.0.1:30006/v1/chat/completions"
MODEL = "dsv41-flash-uva"
tag = os.environ["TAG"]
DEPTHS = [int(x) for x in os.getenv("DEPTHS", "8000 65536 131072 262144 524288").split()]
REPS = int(os.getenv("REPS", "2"))
MAXTOK = int(os.getenv("MAXTOK", "192"))
WORDS = "apple river stone cloud iron velvet copper meadow lantern orbit cedar prism harbor tundra quartz fable".split()


def req(body):
    return urllib.request.Request(B, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})


def mk(depth):
    rnd = random.Random(depth)
    return f"NONCE {uuid.uuid4().hex}\n" + " ".join(rnd.choice(WORDS) for _ in range(int(depth * 0.72))) + "\nContinue the list of words."


def stream_once(base):
    """Returns (ttft_s, decode_tok_s, n_tokens, prompt_tokens)."""
    body = {**base, "max_tokens": MAXTOK, "ignore_eos": True, "stream": True,
            "stream_options": {"include_usage": True}}
    t0 = time.monotonic(); t_first = None; t_last = None; n = 0; ptok = None
    with urllib.request.urlopen(req(body), timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"): continue
            data = line[5:].strip()
            if data == "[DONE]": break
            ev = json.loads(data)
            if ev.get("usage"): ptok = ev["usage"].get("prompt_tokens"); continue
            ch = ev.get("choices") or []
            if not ch: continue
            d = ch[0].get("delta") or {}
            if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                now = time.monotonic(); n += 1
                if t_first is None: t_first = now
                t_last = now
    # a delta may carry several tokens (spec decode emits accepted runs); use usage completion_tokens when present
    return t_first - t0, n, t_last - t_first, ptok


rows = []
urllib.request.urlopen(req({"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}), timeout=300).read()
for d in DEPTHS:
    base = {"model": MODEL, "messages": [{"role": "user", "content": mk(d)}], "temperature": 0,
            "chat_template_kwargs": {"thinking": False}}
    r1 = json.load(urllib.request.urlopen(req({**base, "max_tokens": 1}), timeout=3600))
    ptok = r1["usage"]["prompt_tokens"]
    runs = []
    for _ in range(REPS):
        ttft, nchunks, span, _p = stream_once(base)
        # decode tokens = MAXTOK - 1 (first token arrives with TTFT); chunks < tokens when spec-decode batches deltas
        dec = (MAXTOK - 1) / span if span > 0 else float("nan")
        runs.append({"ttft_s": round(ttft, 2), "chunks": nchunks, "decode_tok_s": round(dec, 2)})
    dec_mean = sum(r["decode_tok_s"] for r in runs) / len(runs)
    rows.append({"depth_target": d, "prompt_tokens": ptok, "decode_tok_s": round(dec_mean, 2), "runs": runs})
    print(f"[{tag}] DEPTH {ptok:>8} tok: ttft {runs[0]['ttft_s']:6.1f}s  decode {dec_mean:6.2f} tok/s  (runs {'/'.join(str(r['decode_tok_s']) for r in runs)}; chunks {'/'.join(str(r['chunks']) for r in runs)})", flush=True)
json.dump({"tag": tag, "rows": rows, "maxtok": MAXTOK, "method": "stream first->last token", "ts": time.strftime("%Y-%m-%dT%H:%M:%S")},
          open(f"/home/milo/dsv41/depth-{tag}.json", "w"), indent=1)
