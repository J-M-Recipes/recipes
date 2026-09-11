#!/usr/bin/env python3
"""cold_prefill_probe.py — engine-agnostic cold-prefill probe for any OpenAI-compatible endpoint.

Method (catid / al-engr convention): a nonce at the START of the prompt (so no prefix-cache hit —
a suffix nonce still lets early KV blocks reuse cache), max_tokens=1, rate = prompt_tokens / request time.
That is effective TTFT-prefill, not a kernel counter. Runs one warm-up request first.

Env: BASE_URL (default http://127.0.0.1:30003/v1), MODEL (dsf-vision-exp), API_KEY (required),
     SIZES ("8000 32000 64000 128000 256000"), N samples per size (3), THINKING (0/1, default 0).
Prints one PREFILL line per size; publish TTFT + tok/s and label as cold prefill.

For thinking models (GLM-5.3, DeepSeek-V4 with thinking on) this measures prefill+thinking unless
thinking is disabled via chat_template_kwargs — keep THINKING=0 or count the first delta of any kind.
"""
import json, os, random, time, urllib.request, uuid

base = os.getenv("BASE_URL", "http://127.0.0.1:30003/v1")
model = os.getenv("MODEL", "dsf-vision-exp")
H = {"Authorization": f"Bearer {os.environ['API_KEY']}", "Content-Type": "application/json"}
SIZES = [int(x) for x in os.getenv("SIZES", "8000 32000 64000 128000 256000").split()]
N = int(os.getenv("N", "3"))
THINK = os.getenv("THINKING", "0") == "1"
WORDS = "apple river stone cloud iron velvet copper meadow lantern orbit cedar prism harbor tundra quartz fable".split()


def post(p, timeout=3600):
    t0 = time.monotonic()
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(p).encode(), headers=H)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r), time.monotonic() - t0


post({"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1})  # warm-up
for target in SIZES:
    rates, samples = [], []
    for _ in range(N):
        rnd = random.Random()
        words = int(target * 0.72)  # ~1.4 tok/word for random words on DeepSeek/GLM tokenizers; verify via prompt_tokens
        body = f"NONCE {uuid.uuid4().hex}\n" + " ".join(rnd.choice(WORDS) for _ in range(words))
        p = {"model": model, "messages": [{"role": "user", "content": body + "\nReply with one word."}],
             "max_tokens": 1, "temperature": 0, "chat_template_kwargs": {"thinking": THINK}}
        r, dt = post(p)
        pt = r["usage"]["prompt_tokens"]
        rates.append(pt / dt); samples.append((pt, dt))
    print(f"PREFILL target={target} prompt_tokens={samples[0][0]} ttft_s={[round(d, 2) for _, d in samples]} "
          f"tok_s={[round(x) for x in rates]} mean_tok_s={sum(rates) / len(rates):.0f}", flush=True)
