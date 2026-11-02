#!/usr/bin/env python3
"""pc3_probe.py — does streaming break prefix reuse? (Card PC3). Agent-shaped turn 2 at 64K, 3 sessions each:
  NS  : non-streaming both turns (PC/PC2 shape, known ~99.9% reuse)
  S   : stream=True, no stream_options
  SU  : stream=True + stream_options.include_usage (KV1 shape, saw 0 reuse)
  SU1 : turn 1 streamed+usage, turn 2 non-streaming
Reuse read from server Prefill lines (#cached-token) per request window, plus turn-2 TTFT. JSON to argv[1]."""
import calendar, json, os, re, subprocess, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pc_probe import ctx  # noqa: E402
BASE = os.getenv("BASE_URL", "http://127.0.0.1:30001/v1"); CONT = os.environ["CONT"]; H = {"Content-Type": "application/json"}
Q = "\n\nReply with the single word OK."


def req(messages, stream, usage):
    p = {"model": "glm-5.3-flash", "messages": messages, "max_tokens": 48, "temperature": 0.0,
         "chat_template_kwargs": {"reasoning_effort": "low"}, "stream": stream}
    if stream and usage:
        p["stream_options"] = {"include_usage": True}
    t0 = time.time(); ttft = None; content = ""; cached = None
    r = urllib.request.urlopen(urllib.request.Request(BASE + "/chat/completions", json.dumps(p).encode(), H), timeout=600)
    if not stream:
        j = json.load(r); content = j["choices"][0]["message"].get("content") or ""
        cached = ((j.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens"); ttft = time.time() - t0
    else:
        for raw in r:
            ln = raw.decode(errors="replace").strip()
            if not ln.startswith("data:") or ln[5:].strip() == "[DONE]":
                continue
            j = json.loads(ln[5:])
            if j.get("usage"):
                cached = (j["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")
            for ch in j.get("choices") or []:
                d = ch.get("delta") or {}
                if (d.get("content") or d.get("reasoning_content")) and ttft is None:
                    ttft = time.time() - t0
                content += d.get("content") or ""
    return {"t0": t0, "t1": time.time(), "ttft": round(ttft or -1, 2), "content": content, "usage_cached": cached}


def server_cached(t0, t1):
    out = subprocess.run(["docker", "logs", CONT], capture_output=True, text=True)
    s = (out.stdout + out.stderr).replace("\r", "\n"); new = cached = 0
    for ln in s.splitlines():
        m = re.match(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", ln)
        if not m or "Prefill batch" not in ln:
            continue
        ts = calendar.timegm(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
        if t0 - 1 <= ts <= t1 + 1:
            a = re.search(r"#new-token: (\d+)", ln); b = re.search(r"#cached-token: (\d+)", ln)
            new += int(a.group(1)) if a else 0; cached += int(b.group(1)) if b else 0
    return round(cached / max(1, new + cached), 4)


res = []
for name, (s1, u1, s2, u2) in {"NS": (False, False, False, False), "S": (True, False, True, False),
                               "SU": (True, True, True, True), "SU1": (True, True, False, False)}.items():
    for i in range(3):
        c = ctx(f"PC3-{name}-{i}", 64000)
        a = req([{"role": "user", "content": c + Q}], s1, u1); time.sleep(1)
        msgs = [{"role": "user", "content": c + Q}, {"role": "assistant", "content": a["content"] or "OK"},
                {"role": "user", "content": "Now reply with the single word YES."}]
        b = req(msgs, s2, u2); time.sleep(2)
        row = {"mode": name, "i": i, "t2_ttft": b["ttft"], "t2_server_cached_frac": server_cached(b["t0"], b["t1"]),
               "t2_usage_cached": b["usage_cached"], "t1_content": a["content"][:60]}
        print(f"[pc3] {row}", flush=True); res.append(row)
json.dump(res, open(sys.argv[1], "w"), indent=1)
