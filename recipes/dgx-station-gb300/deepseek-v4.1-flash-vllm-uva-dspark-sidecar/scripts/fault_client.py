#!/usr/bin/env python3
"""fault_client.py <base> <model> <n> <out.json> — n concurrent streaming chats (T=0, long), records every chunk's wall
time so the fault card can check that no token arrives after the sidecar stall + watchdog trip."""
import json, sys, threading, time, urllib.request
base, model, n, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
res = [None] * n
def run(i):
    body = json.dumps({"model": model, "stream": True, "temperature": 0, "max_tokens": 6000,
                       "chat_template_kwargs": {"thinking": False},
                       "messages": [{"role": "user", "content": f"[{i}] Write a very long, detailed history of the printing press, chapter by chapter."}]}).encode()
    times, err, fin = [], None, None
    try:
        req = urllib.request.Request(base + "/v1/chat/completions", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            for line in r:
                if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                    d = json.loads(line[6:])
                    ch = d.get("choices") or [{}]
                    if ch[0].get("delta", {}).get("content"):
                        times.append(time.time())
                    fin = ch[0].get("finish_reason") or fin
    except Exception as e:
        err = f"{type(e).__name__}: {e}"[:200]
    res[i] = {"chunks": len(times), "first": times[0] if times else None, "last": times[-1] if times else None,
              "finish": fin, "error": err, "times_tail": times[-5:]}
ts = [threading.Thread(target=run, args=(i,)) for i in range(n)]
[t.start() for t in ts]; [t.join() for t in ts]
json.dump(res, open(out, "w"), indent=1)
for i, r in enumerate(res):
    print(i, r["chunks"], "last", f"{r['last']:.3f}" if r["last"] else None, "finish", r["finish"], "err", r["error"])
print("max_last", max((r["last"] or 0) for r in res))
