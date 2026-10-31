#!/usr/bin/env python3
"""Warm a freshly booted K3 pod before real traffic.

vLLM JIT-compiles a FlashAttention kernel (FlashAttentionForwardSm100) the first time a prompt spans more than one
prefill chunk; that first request pays ~4-5 s once per engine process. Send one prompt longer than a chunk, then a
short decode, and print the timings. Usage: warmup.py PORT [CHUNK_TOKENS=1024]
"""
import json
import sys
import time
import urllib.request

port = sys.argv[1]
chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
url = f"http://127.0.0.1:{port}/v1/completions"
sent = "The transistor was invented at Bell Labs in 1947 by Bardeen, Brattain and Shockley. "
prompt = sent * (chunk // 16 + 8)          # ~20 tokens per sentence -> comfortably more than one chunk


def call(p, n):
    b = json.dumps({"model": "k3", "prompt": p, "max_tokens": n, "temperature": 0}).encode()
    t0 = time.time()
    d = json.load(urllib.request.urlopen(urllib.request.Request(url, b, {"Content-Type": "application/json"}),
                                         timeout=600))
    return time.time() - t0, d["usage"]["prompt_tokens"]


for i in range(2):
    dt, n = call(prompt, 1)
    print(f"warmup multi-chunk prompt {i + 1}: {n} tokens, {dt:.2f} s", flush=True)
dt, _ = call("Hello", 16)
print(f"warmup decode 16 tokens: {dt:.2f} s", flush=True)
