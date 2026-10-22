#!/usr/bin/env python3
"""glm_needles.py BASE LEN... -- chat-API needle-in-haystack for GLM on Flare (thinking on, effort low).
Filler = numbered records; one needle line at ~60% depth; nonce prefix defeats prefix cache. Tokens from usage."""
import json, sys, time, uuid, random, urllib.request
B = sys.argv[1]; LENS = [int(x) for x in sys.argv[2:]]
M = json.load(urllib.request.urlopen(B + "/models"))["data"][0]["id"]
WORDS = "copper juniper atlas river quartz meadow lantern harbor cobalt willow ember granite falcon orchid".split()
def prompt(target):
    nonce = uuid.uuid4().hex; code = f"{random.randint(100000, 999999)}"
    lines = []; i = 0; chars = int(target * 3.6)
    rnd = random.Random(nonce)
    while sum(map(len, lines)) < chars:
        lines.append(f"Record {i:07d}: " + " ".join(rnd.choice(WORDS) for _ in range(8)) + ".\n"); i += 1
    pos = int(len(lines) * 0.6)
    lines.insert(pos, f"IMPORTANT: the vault access code is {code}.\n")
    return f"NONCE {nonce}\n" + "".join(lines) + "\nWhat is the vault access code? Reply with only the number.", code
for L in LENS:
    p, code = prompt(L)
    body = {"model": M, "messages": [{"role": "user", "content": p}], "max_tokens": 2048, "reasoning_effort": "low"}
    t0 = time.time()
    try:
        req = urllib.request.Request(B + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=3600))
        msg = r["choices"][0]["message"]; ans = (msg.get("content") or "").strip()
        row = {"target": L, "prompt_tokens": r["usage"]["prompt_tokens"], "completion_tokens": r["usage"]["completion_tokens"],
               "wall_s": round(time.time() - t0, 1), "expected": code, "answer": ans[:60], "pass": code in ans}
    except Exception as e:
        row = {"target": L, "error": str(e)[:300], "wall_s": round(time.time() - t0, 1), "pass": False}
    print(json.dumps(row), flush=True)
