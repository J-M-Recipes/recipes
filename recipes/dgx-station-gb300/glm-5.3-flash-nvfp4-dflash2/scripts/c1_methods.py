#!/usr/bin/env python3
"""C1 decode by prompt/method, 3 runs each. reasoning_effort from EFFORT (default low)."""
import json, time, urllib.request, os, sys
U = os.getenv("BASE_URL", "http://127.0.0.1:30001/v1") + "/chat/completions"; H = {"Content-Type": "application/json"}
TAG = os.getenv("TAG", "x")
EFFORT = os.getenv("EFFORT", "low")
def run(label, prompt, mt):
    p = {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": prompt}], "max_tokens": mt, "temperature": 0.0,
         "chat_template_kwargs": {"reasoning_effort": EFFORT}}
    rs = []; rt = None; last = None
    for _ in range(3):
        t0 = time.time(); r = json.load(urllib.request.urlopen(urllib.request.Request(U, data=json.dumps(p).encode(), headers=H), timeout=600)); dt = time.time() - t0
        u = r["usage"]; rs.append(round(u["completion_tokens"] / dt, 1)); rt = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
        last = r
    m = (last["choices"][0]["message"] if last else {})
    rc = m.get("reasoning_content") or ""
    c = m.get("content") or ""
    print(f"[{TAG}] {label}: runs {rs} median {sorted(rs)[1]} (reasoning_tokens={rt} reasoning_len={len(rc)} content_len={len(c)})", flush=True)
run(f"recipe-method history-essay 512 effort={EFFORT}", "Write a detailed essay about the history of computing.", 512)
run(f"failure-ledger prose 300 effort={EFFORT}", "Write a 350-word essay on why engineers should keep failure ledgers. No headings, no lists.", 300)
run(f"code 400 effort={EFFORT}", "Write a Python module with three functions: parse an nginx log line, aggregate status codes, and print a table. Include docstrings.", 400)
run(f"shell-ops 200 effort={EFFORT}", "Give me the exact bash commands to: find files over 1GB under /var, list docker containers with their images, and show the ten largest directories in /home. Commands only.", 200)
