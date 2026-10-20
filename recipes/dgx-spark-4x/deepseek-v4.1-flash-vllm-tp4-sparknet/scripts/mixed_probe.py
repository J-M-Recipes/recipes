#!/usr/bin/env python3
"""Mixed-traffic probe: TTFT of a short chat request issued while a cold ~93K prefill is in flight.
usage: mixed_probe.py ROOT MODEL NTOK      (ROOT = http://127.0.0.1:8002)
1. baseline short-chat TTFT x3 (idle server)
2. start a cold NTOK-token prompt (unique leading nonce, token ids, max_tokens 1) in a thread
3. after 1.0 s send the short streaming chat; record its TTFT/total and the long prefill time. x2 reps (fresh nonce each)."""
import json, random, sys, threading, time, urllib.request

root, model, ntok = sys.argv[1], sys.argv[2], int(sys.argv[3])
WORDS = "amber basin cedar delta ember fjord garnet harbor iris juniper kestrel lumen meadow nimbus orchid pylon".split()

def post(path, body, timeout=1800):
    r = urllib.request.Request(root + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(r, timeout=timeout)

def short_ttft(tag):
    body = {"model": model, "temperature": 0, "max_tokens": 32, "stream": True,
            "messages": [{"role": "user", "content": f"[{tag}] Write one sentence about the sea."}]}
    t0 = time.time(); first = None
    with post("/v1/chat/completions", body) as f:
        for line in f:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            d = json.loads(line[5:])
            ch = d.get("choices") or []
            if first is None and ch and (ch[0].get("delta", {}).get("content") or ch[0].get("delta", {}).get("reasoning_content")):
                first = time.time() - t0
    return {"ttft_s": round(first, 3) if first else None, "total_s": round(time.time() - t0, 3)}

def long_ids(seed):
    rnd = random.Random(seed)
    nonce = "".join(rnd.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(12))
    text = f"[mixed {nonce}] " + " ".join(f"{rnd.choice(WORDS)}{rnd.randint(0, 999)}" for _ in range(int(ntok * 0.6)))
    with post("/tokenize", {"model": model, "prompt": text, "add_special_tokens": False}) as f:
        ids = json.loads(f.read())["tokens"]
    return ids[:ntok]

out = {"ntok": ntok, "baseline": [], "mixed": []}
for i in range(3):
    out["baseline"].append(short_ttft(f"base{i}-{time.time()}"))
for rep in range(2):
    ids = long_ids(int(time.time() * 1000) + rep)
    res = {}
    def big():
        t0 = time.time()
        with post("/v1/completions", {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0}) as f:
            u = json.loads(f.read())["usage"]
        res["long"] = {"prompt_tokens": u["prompt_tokens"], "seconds": round(time.time() - t0, 3)}
    th = threading.Thread(target=big); th.start()
    time.sleep(1.0)
    s = short_ttft(f"mix{rep}-{time.time()}")
    th.join()
    out["mixed"].append({"short": s, **res})
    print(json.dumps(out["mixed"][-1]), flush=True)
print(json.dumps(out))
json.dump(out, open("mixed.json", "w"), indent=1)
