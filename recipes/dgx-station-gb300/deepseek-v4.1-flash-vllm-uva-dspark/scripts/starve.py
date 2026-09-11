#!/usr/bin/env python3
"""Mixed-workload starvation probe: fire a long cold prefill, then N short requests staggered behind it.
Reports each short request's TTFT vs its solo baseline. Env: BASE_URL, MODEL, LONG (tokens target), SHORTS (count), GAP (s)."""
import json, os, random, threading, time, urllib.request, uuid
base = os.getenv("BASE_URL", "http://127.0.0.1:30006/v1"); model = os.getenv("MODEL", "dsv41-flash-uva")
LONG = int(os.getenv("LONG", "480000")); SHORTS = int(os.getenv("SHORTS", "4")); GAP = float(os.getenv("GAP", "5"))
H = {"Authorization": "Bearer x", "Content-Type": "application/json"}
W = "apple river stone cloud iron velvet copper meadow lantern orbit cedar prism harbor tundra quartz fable".split()

def stream_ttft(msgs, max_tokens=48):
    p = {"model": model, "messages": msgs, "max_tokens": max_tokens, "temperature": 0, "stream": True,
         "chat_template_kwargs": {"thinking": False}}
    t0 = time.monotonic(); first = None; n = 0
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(p).encode(), headers=H)
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]": continue
            d = json.loads(line[6:]); ch = d["choices"][0]["delta"] if d.get("choices") else {}
            if ch.get("content"):
                n += 1
                if first is None: first = time.monotonic() - t0
    return first, time.monotonic() - t0, n

def short_msgs():
    return [{"role": "user", "content": f"NONCE {uuid.uuid4().hex}. In one sentence, what does `ls -la` show?"}]

def long_msgs():
    rnd = random.Random(); body = f"NONCE {uuid.uuid4().hex}\n" + " ".join(rnd.choice(W) for _ in range(int(LONG * 0.72)))
    return [{"role": "user", "content": body + "\nReply with one word."}]

stream_ttft(short_msgs(), 8)  # warm
solo = [stream_ttft(short_msgs())[0] for _ in range(3)]
print(f"SOLO short TTFT: {[round(x,3) for x in solo]}  (mean {sum(solo)/3:.3f}s)", flush=True)

res = {}
def long_job():
    res["long"] = stream_ttft(long_msgs(), 8)
def short_job(i):
    res[f"s{i}"] = stream_ttft(short_msgs())

t = threading.Thread(target=long_job); t.start(); tl0 = time.monotonic()
shorts = []
for i in range(SHORTS):
    time.sleep(GAP)
    th = threading.Thread(target=short_job, args=(i,)); th.start(); shorts.append((th, time.monotonic() - tl0))
for th, _ in shorts: th.join()
t.join()
lf, lw, _ = res["long"]
print(f"LONG  target={LONG} TTFT={lf:.1f}s wall={lw:.1f}s", flush=True)
for i, (_, off) in enumerate(shorts):
    f, w, n = res[f"s{i}"]
    print(f"SHORT#{i} fired at +{off:.0f}s into long prefill: TTFT={f:.2f}s (solo {sum(solo)/3:.2f}s, x{f/(sum(solo)/3):.0f}) wall={w:.2f}s tokens={n}", flush=True)
