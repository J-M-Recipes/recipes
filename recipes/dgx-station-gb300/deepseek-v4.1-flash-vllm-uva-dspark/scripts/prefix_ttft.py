#!/usr/bin/env python3
"""E5: prefix-cache TTFT. Request 1 = cold (nonce on system prompt); 2-6 = identical system, nonce only in last user msg."""
import json, os, time, urllib.request, uuid
base = os.getenv("BASE_URL", "http://127.0.0.1:30006/v1")
model = os.getenv("MODEL", "dsv41-flash-uva")
H = {"Authorization": "Bearer x", "Content-Type": "application/json"}
preamble = open("/tmp/preamble.txt").read()
ctx0 = json.load(open("/tmp/ctx0.json"))

def stream_ttft(msgs, max_tokens=16):
    p = {"model": model, "messages": msgs, "max_tokens": max_tokens, "temperature": 0, "stream": True,
         "chat_template_kwargs": {"thinking": False}}
    t0 = time.monotonic(); first = None; n = 0
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(p).encode(), headers=H)
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            d = json.loads(line[6:])
            ch = d["choices"][0]["delta"] if d.get("choices") else {}
            if ch.get("content"):
                n += 1
                if first is None:
                    first = time.monotonic() - t0
    return first, time.monotonic() - t0, n

def nostream_usage(msgs, max_tokens=1):
    p = {"model": model, "messages": msgs, "max_tokens": max_tokens, "temperature": 0, "stream": False,
         "chat_template_kwargs": {"thinking": False}}
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(p).encode(), headers=H)
    r = json.load(urllib.request.urlopen(req, timeout=600))
    return r["usage"]

def metrics_prefix():
    t = urllib.request.urlopen(base.replace("/v1", "") + "/metrics", timeout=10).read().decode()
    d = {}
    for l in t.splitlines():
        if l.startswith("vllm:prefix_cache"):
            k = l.split("{")[0] if "{" in l else l.split()[0]
            d[k] = float(l.split()[-1])
        if "gpu_prefix_cache" in l or "prefix_cache" in l:
            parts = l.split()
            if parts and not l.startswith("#"):
                d[parts[0].split("{")[0]] = float(parts[-1])
    return d

# usage once on warm-shape body without nonce
usage_msgs = [{"role": "system", "content": preamble}] + ctx0
u = nostream_usage(usage_msgs)
print(f"USAGE prompt_tokens={u.get('prompt_tokens')} completion_tokens={u.get('completion_tokens')} total={u.get('total_tokens')}", flush=True)
m0 = metrics_prefix()
print(f"METRICS_BEFORE {m0}", flush=True)

times = []
for i in range(6):
    nonce = uuid.uuid4().hex
    if i == 0:
        sys_content = preamble + f"\nNONCE {nonce}"
        msgs = [{"role": "system", "content": sys_content}] + ctx0
        kind = "COLD"
    else:
        msgs = [{"role": "system", "content": preamble}] + list(ctx0)
        if msgs and msgs[-1]["role"] == "user":
            msgs[-1] = dict(msgs[-1])
            msgs[-1]["content"] = msgs[-1]["content"] + f"\nNONCE {nonce}"
        else:
            msgs.append({"role": "user", "content": f"NONCE {nonce}. Reply with one word."})
        kind = "WARM"
    first, wall, n = stream_ttft(msgs)
    times.append((kind, first, wall, n))
    print(f"REQ#{i} {kind} TTFT={first:.3f}s wall={wall:.3f}s tokens={n}", flush=True)

warm = [t[1] for t in times[1:] if t[1] is not None]
cold = times[0][1]
print(f"COLD {cold:.3f}s  WARM mean={sum(warm)/len(warm):.3f}s  [{', '.join(f'{x:.3f}' for x in warm)}]  speedup={cold/(sum(warm)/len(warm)):.1f}x", flush=True)
m1 = metrics_prefix()
print(f"METRICS_AFTER {m1}", flush=True)
