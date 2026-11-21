#!/usr/bin/env python3
"""Instrumented mixed probe: one cold NTOK prompt (token ids, max_tokens 1) + a short streamed chat sent at ARRIVE s,
while /metrics is polled every ~40 ms. Records waiting/running counts and prompt-token progress so we can see whether
the short request WAITS (scheduler) or RUNS slowly (step time).
usage: mixed_trace.py ROOT MODEL NTOK ARRIVE OUT.json"""
import json, random, re, sys, threading, time, urllib.request
root, model, ntok, arrive, outp = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
MAXTOK = int(sys.argv[6]) if len(sys.argv) > 6 else 32
THINK = (sys.argv[7] if len(sys.argv) > 7 else "on") == "on"
W = "amber basin cedar delta ember fjord garnet harbor iris juniper kestrel lumen meadow nimbus orchid pylon".split()
def post(path, body, timeout=1800):
    r = urllib.request.Request(root + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(r, timeout=timeout)
rnd = random.Random(time.time())
nonce = "".join(rnd.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(12))
text = f"[trace {nonce}] " + " ".join(f"{rnd.choice(W)}{rnd.randint(0, 999)}" for _ in range(int(ntok * 0.6)))
with post("/tokenize", {"model": model, "prompt": text, "add_special_tokens": False}) as f:
    ids = json.loads(f.read())["tokens"][:ntok]
T0 = None; ev = {}; samples = []; stop = False
pat = re.compile(r'^(vllm:(num_requests_running|num_requests_waiting|prompt_tokens_total|generation_tokens_total|num_requests_waiting_by_reason)(\{[^}]*\})?) ([0-9.eE+-]+)', re.M)
def poll():
    while not stop:
        t = time.time() - T0
        try:
            m = urllib.request.urlopen(root + "/metrics", timeout=5).read().decode()
            d = {}
            for full, name, lab, val in pat.findall(m):
                key = name + (re.sub(r'engine="0",model_name="[^"]*",?', "", lab or "") if "reason" in (lab or "") else "")
                d[key] = d.get(key, 0) + float(val)
            samples.append({"t": round(t, 3), **d})
        except Exception as e:
            samples.append({"t": round(t, 3), "err": str(e)[:80]})
        time.sleep(0.04)
def big():
    t = time.time()
    with post("/v1/completions", {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0}) as f:
        u = json.loads(f.read())["usage"]
    ev["long_done"] = round(time.time() - T0, 3); ev["long_tokens"] = u["prompt_tokens"]
def short():
    body = {"model": model, "temperature": 0, "max_tokens": MAXTOK, "stream": True, "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": f"[{nonce}-s] Write one sentence about the sea." if MAXTOK <= 32 else f"[{nonce}-s] Write a 300-word story about a lighthouse keeper."}]}
    if not THINK: body["chat_template_kwargs"] = {"thinking": False}
    ev["short_sent"] = round(time.time() - T0, 3)
    with post("/v1/chat/completions", body) as f:
        for line in f:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            d = json.loads(line[5:]); ch = d.get("choices") or []
            if ch and (ch[0].get("delta", {}).get("content") or ch[0].get("delta", {}).get("reasoning")):
                ev.setdefault("chunk_t", []).append(round(time.time() - T0, 3))
            if d.get("usage"): ev["short_completion_tokens"] = d["usage"].get("completion_tokens")
            if "short_first" not in ev and ch and (ch[0].get("delta", {}).get("content") or ch[0].get("delta", {}).get("reasoning_content") or ch[0].get("delta", {}).get("reasoning")):
                ev["short_first"] = round(time.time() - T0, 3)
    ev["short_done"] = round(time.time() - T0, 3)
T0 = time.time()
tp = threading.Thread(target=poll, daemon=True); tp.start()
tb = threading.Thread(target=big); tb.start()
time.sleep(arrive); short(); tb.join(); time.sleep(0.3); stop = True; tp.join()
ev["short_ttft"] = round(ev["short_first"] - ev["short_sent"], 3)
ev["short_total"] = round(ev["short_done"] - ev["short_sent"], 3)
if ev.get("short_completion_tokens") and ev["short_done"] > ev["short_first"]:
    ev["short_decode_tok_s"] = round((ev["short_completion_tokens"] - 1) / (ev["short_done"] - ev["short_first"]), 1)
ev["maxtok"] = MAXTOK; ev["think"] = THINK
ct = ev.pop("chunk_t", [])
if ct and ev.get("short_completion_tokens"):
    # decode rate while the long prefill was still running: tokens (pro-rated by stream chunks) between first token and long_done
    tpc = ev["short_completion_tokens"] / len(ct)
    during = [t for t in ct if t <= ev["long_done"]]
    if len(during) > 1 and during[-1] > during[0]:
        ev["decode_during_prefill_tok_s"] = round((len(during) - 1) * tpc / (during[-1] - during[0]), 1)
    ev["tokens_during_prefill"] = round(len(during) * tpc)
json.dump({"ev": ev, "samples": samples}, open(outp, "w"))
print(json.dumps(ev))
