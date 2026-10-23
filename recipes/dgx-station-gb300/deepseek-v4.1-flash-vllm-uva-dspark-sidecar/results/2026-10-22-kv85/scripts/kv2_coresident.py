#!/usr/bin/env python3
"""kv2_coresident.py — capacity stress that actually fills the KV pool on the live lane (no reboot).
4 independent ~1M-token prompts, each decoding 16K tokens with ignore_eos, so all four stay resident at once.
Samples vllm:kv_cache_usage_perc (/metrics, 2 s) and GB300 memory.used (1 s). Output: JSON summary on stdout + file.
Run on the box: python3 kv2_coresident.py OUTDIR
"""
import json, os, subprocess, sys, threading, time, urllib.request, random
sys.path.insert(0, "~/dsv41/m3/sc5")
from sc5_probes import stream_chat, filler  # stdlib-only instruments used by SC5/KV1

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
BASE = "http://127.0.0.1:30006"; MODEL = "dsv41-flash-uva"; GB300 = "GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55"
N, WORDS, MAXTOK = 4, int(os.environ.get("WORDS", "880000")), int(os.environ.get("MAXTOK", "16000"))
stop = threading.Event(); kv_samples = []; mem_samples = []; results = [None] * N

def sample_kv():
    while not stop.is_set():
        try:
            txt = urllib.request.urlopen(BASE + "/metrics", timeout=5).read().decode()
            for line in txt.splitlines():
                if line.startswith("vllm:kv_cache_usage_perc{"):
                    kv_samples.append((time.time(), float(line.rsplit(" ", 1)[1])))
                if line.startswith("vllm:num_requests_running{"):
                    pass
        except Exception:
            pass
        time.sleep(2)

def sample_mem():
    while not stop.is_set():
        try:
            o = subprocess.run(["nvidia-smi", f"--id={GB300}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
            mem_samples.append((time.time(), int(o)))
        except Exception:
            pass
        time.sleep(1)

def worker(i):
    seed = 7000 + i
    prompt = f"DOC {i} NONCE {random.randrange(10**12)}\n" + filler(WORDS, seed) + "\n\nContinue the list of words."
    t0 = time.time()
    r = stream_chat(BASE, {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": MAXTOK,
                           "ignore_eos": True, "temperature": 0, "chat_template_kwargs": {"thinking": False}}, timeout=3600)
    results[i] = {"i": i, "t_send": t0, "t_end": time.time(), "http": r["http"], "error": r["error"], "prompt_tokens": r["prompt_tokens"],
                  "completion_tokens": r["completion_tokens"], "ttft_s": r["ttft_s"] and round(r["ttft_s"], 1),
                  "decode_tok_s": r.get("decode_tok_s") and round(r["decode_tok_s"], 1), "finish": r["finish_reason"]}

ths = [threading.Thread(target=sample_kv, daemon=True), threading.Thread(target=sample_mem, daemon=True)]
for t in ths: t.start()
mem0 = None
time.sleep(2); mem0 = mem_samples[-1][1] if mem_samples else None
ws = []
for i in range(N):
    t = threading.Thread(target=worker, args=(i,)); t.start(); ws.append(t); time.sleep(2)
for t in ws: t.join()
stop.set(); time.sleep(2)
peak_kv = max((v for _, v in kv_samples), default=None); peak_mem = max((v for _, v in mem_samples), default=None)
# co-residency: time window during which all 4 had sent and none had finished
last_send = max(r["t_send"] + (r["ttft_s"] or 0) for r in results if r); first_end = min(r["t_end"] for r in results if r)
summ = {"n": N, "words": WORDS, "max_tokens": MAXTOK, "results": results, "peak_kv_usage": peak_kv,
        "kv_usage_samples": len(kv_samples), "gpu_mem_idle_mib": mem0, "gpu_mem_peak_mib": peak_mem,
        "gpu_mem_growth_mib": (peak_mem - mem0) if (peak_mem and mem0) else None,
        "all_prefilled_before_first_end": last_send < first_end, "coresident_window_s": round(first_end - last_send, 1),
        "ok": all(r and r["http"] == 200 and not r["error"] and r["completion_tokens"] == MAXTOK for r in results)}
json.dump({"summary": summ, "kv_samples": kv_samples, "mem_samples": mem_samples}, open(f"{OUT}/coresident.json", "w"))
print(json.dumps(summ, indent=1))
