#!/usr/bin/env python3
"""Phase-locked yield probe: cold NTOK prompt (max_tokens 1); a short streamed chat (thinking off) is sent on the first
/metrics poll that shows >= FRAC of the long prompt computed. Records TTFT (first delta of any kind), first content,
long prompt completion. usage: phase_trace.py ROOT NTOK FRAC SHORT_TOK OUT.json"""
import json, random, re, sys, threading, time, urllib.request
root, ntok, frac, short_tok, outp = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
M = "deepseek-v4.1-flash"
W = "amber basin cedar delta ember fjord garnet harbor iris juniper kestrel lumen meadow nimbus orchid pylon".split()
def post(path, body, timeout=1800):
    return urllib.request.urlopen(urllib.request.Request(root + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=timeout)
rnd = random.Random(time.time()); nonce = "".join(rnd.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(12))
def ids_for(n, tag):
    text = f"[{tag} {nonce}] " + " ".join(f"{rnd.choice(W)}{rnd.randint(0, 999)}" for _ in range(int(n * 0.6) + 50))
    with post("/tokenize", {"model": M, "prompt": text, "add_special_tokens": False}) as f:
        return json.loads(f.read())["tokens"][:n]
long_ids = ids_for(ntok, "long")
short_ids = ids_for(short_tok, "short") if short_tok > 64 else None
def prompt_tokens():
    m = urllib.request.urlopen(root + "/metrics", timeout=5).read().decode()
    return sum(float(v) for v in re.findall(r'^vllm:prompt_tokens_total\{[^}]*\} ([0-9.e+]+)', m, re.M))
ev = {}; T0 = time.time()
def big():
    with post("/v1/completions", {"model": M, "prompt": long_ids, "max_tokens": 1, "temperature": 0}) as f: f.read()
    ev["long_done"] = round(time.time() - T0, 3)
# prompt_tokens_total only moves when a prefill completes, so the phase lock is in time: a solo cold run of the
# same length first (fresh nonce), then the short arrives at FRAC x that solo wall.
t = time.time()
with post("/v1/completions", {"model": M, "prompt": ids_for(ntok, "cal"), "max_tokens": 1, "temperature": 0}) as f: f.read()
ev["solo_s"] = round(time.time() - t, 3); time.sleep(2)
T0 = time.time(); tb = threading.Thread(target=big); tb.start()
time.sleep(frac * ev["solo_s"])
ev["phase_source"] = "time"
if short_ids is None:
    body = {"model": M, "messages": [{"role": "user", "content": f"[{nonce}] Write one sentence about the sea."}], "max_tokens": 32,
            "temperature": 0, "stream": True, "chat_template_kwargs": {"thinking": False}}
    path = "/v1/chat/completions"
else:
    body = {"model": M, "prompt": short_ids, "max_tokens": 16, "temperature": 0, "stream": True}; path = "/v1/completions"
ev["short_sent"] = round(time.time() - T0, 3)
with post(path, body) as f:
    for line in f:
        line = line.decode().strip()
        if not line.startswith("data:") or line.endswith("[DONE]"): continue
        d = json.loads(line[5:]); ch = d.get("choices") or []
        if not ch: continue
        txt = ch[0].get("text") or (ch[0].get("delta") or {}).get("content") or (ch[0].get("delta") or {}).get("reasoning")
        if txt and "short_first" not in ev: ev["short_first"] = round(time.time() - T0, 3)
ev["short_done"] = round(time.time() - T0, 3)
tb.join()
ev["ttft"] = round(ev["short_first"] - ev["short_sent"], 3) if "short_first" in ev else None
ev.update({"ntok": ntok, "frac": frac, "short_tok": short_tok})
json.dump(ev, open(outp, "w")); print(json.dumps(ev))
