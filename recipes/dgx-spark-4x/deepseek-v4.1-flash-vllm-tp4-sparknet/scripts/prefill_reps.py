#!/usr/bin/env python3
"""Cold-prefill repeats with a unique leading nonce (no prefix-cache hits).
usage: prefill_reps.py ROOT MODEL REPS LEN [LEN ...]   -> prefill_reps.json in cwd
Prompt = token ids of '[pre <nonce>] ' + random filler words, trimmed to exactly LEN; /v1/completions max_tokens 1, T=0."""
import json, random, sys, time, urllib.request, statistics as st
root, model, reps = sys.argv[1], sys.argv[2], int(sys.argv[3]); lens = [int(x) for x in sys.argv[4:]]
WORDS = "amber basin cedar delta ember fjord garnet harbor iris juniper kestrel lumen meadow nimbus orchid pylon quartz raven".split()
def post(path, body, timeout=1800):
    r = urllib.request.Request(root + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        return json.loads(f.read())
out = {}
for n in lens:
    rows = []
    for i in range(reps):
        rnd = random.Random(time.time_ns())
        nonce = "".join(rnd.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(12))
        text = f"[pre {nonce}] " + " ".join(f"{rnd.choice(WORDS)}{rnd.randint(0, 999)}" for _ in range(int(n * 0.6)))
        ids = post("/tokenize", {"model": model, "prompt": text, "add_special_tokens": False})["tokens"][:n]
        t0 = time.time()
        r = post("/v1/completions", {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0})
        dt = time.time() - t0
        rows.append({"prompt_tokens": r["usage"]["prompt_tokens"], "seconds": round(dt, 3), "tok_s": round(r["usage"]["prompt_tokens"] / dt, 1)})
        print(json.dumps({"len": n, **rows[-1]}), flush=True)
    out[str(n)] = {"reps": rows, "median_tok_s": round(st.median(x["tok_s"] for x in rows), 1)}
json.dump(out, open("prefill_reps.json", "w"), indent=1)
print(json.dumps({k: v["median_tok_s"] for k, v in out.items()}))
