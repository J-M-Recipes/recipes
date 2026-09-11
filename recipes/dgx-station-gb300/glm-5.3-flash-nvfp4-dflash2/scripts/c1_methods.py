#!/usr/bin/env python3
"""C1 decode by prompt/method, 3 runs each. Reconciles the recipe's 234 tok/s method (history essay, 512 tok, thinking default)."""
import json, time, urllib.request, os, sys
U = os.getenv("BASE_URL", "http://127.0.0.1:30001/v1") + "/chat/completions"; H = {"Content-Type": "application/json"}
TAG = os.getenv("TAG", "x")
def run(label, prompt, mt, think):
    p = {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": prompt}], "max_tokens": mt, "temperature": 0.0}
    if think is not None: p["chat_template_kwargs"] = {"enable_thinking": think}
    rs = []; rt = None
    for _ in range(3):
        t0 = time.time(); r = json.load(urllib.request.urlopen(urllib.request.Request(U, data=json.dumps(p).encode(), headers=H), timeout=600)); dt = time.time() - t0
        u = r["usage"]; rs.append(round(u["completion_tokens"] / dt, 1)); rt = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
    print(f"[{TAG}] {label}: runs {rs} median {sorted(rs)[1]} (reasoning_tokens={rt})", flush=True)
run("recipe-method history-essay 512 think-default", "Write a detailed essay about the history of computing.", 512, None)
run("history-essay 512 think-OFF", "Write a detailed essay about the history of computing.", 512, False)
run("failure-ledger prose 300 think-OFF", "Write a 350-word essay on why engineers should keep failure ledgers. No headings, no lists.", 300, False)
run("code 400 think-OFF", "Write a Python module with three functions: parse an nginx log line, aggregate status codes, and print a table. Include docstrings.", 400, False)
run("shell-ops 200 think-OFF", "Give me the exact bash commands to: find files over 1GB under /var, list docker containers with their images, and show the ten largest directories in /home. Commands only.", 200, False)
