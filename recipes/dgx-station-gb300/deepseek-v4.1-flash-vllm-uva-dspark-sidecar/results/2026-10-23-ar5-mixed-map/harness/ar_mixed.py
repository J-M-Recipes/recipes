#!/usr/bin/env python3
"""Mixed-domain traffic at concurrency C: JP/DE prose and agent English
alternate per request in every worker. Measures aggregate decode throughput
and per-domain per-request tok/s, overall and in the post-60 s window.

Usage: ar_mixed.py <tag> <outdir> <concurrency> <seconds>
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path

tag, outdir, C, SECS = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
B = "http://127.0.0.1:30006/v1/chat/completions"
MAXTOK = 400
POST = 60.0
JPDE = [
    "箇条書きを使わず、冬の海辺で一人で過ごす一日について、日本語の長い随筆を書いてください。波、風、光を具体的に。",
    "箇条書きなしで、子供のころ町の図書館で過ごした午後の記憶を、丁寧な日本語の長い文章で書いてください。",
    "Schreibe einen langen fließenden Essay auf Deutsch ohne Aufzählungen über das Handwerk eines Uhrmachers in Glashütte: Werkbank, Licht, Geduld.",
    "Schreibe fließende deutsche Prosa ohne Listen über eine Nachtfahrt mit dem Regionalzug durch das Ruhrgebiet im Regen.",
    "箇条書きを使わず、茶の湯における季節の移ろいについて、日本語の長い随筆を書いてください。道具と光の変化に触れること。",
]
AGENT = [
    "You are Hermes. Write a continuous English prose explanation of how you would inspect a vLLM server with curl, no lists.",
    "In flowing English with no bullets, describe a careful git bisect of a CUDA graph regression in a MoE kernel.",
    "Write an essay-style walkthrough of a Python tool-calling agent loop: parse, dispatch, observe. No lists.",
    "In narrative prose, explain how speculative decoding with a draft model interacts with routed experts.",
    "Write discursive English about pinning hot MoE experts in HBM and leaving cold ones in Grace RAM.",
]
if os.environ.get("HELDOUT") == "1":
    # Held-out prompts: never used to collect the counts the mixed map was fitted on.
    JPDE = [
        "箇条書きを使わず、雨の日の古書店で見つけた一冊の本をめぐる物語を、日本語の長い文章で書いてください。",
        "箇条書きなしで、山間の小さな駅で列車を待つ老人の一日を、静かな日本語の随筆として書いてください。",
        "Schreibe ohne Aufzählungen einen langen deutschen Text über einen Herbstmorgen auf einem Weingut an der Mosel.",
        "Schreibe fließende deutsche Prosa ohne Listen über die Arbeit einer Bibliothekarin in einer alten Klosterbibliothek.",
        "箇条書きを使わず、港町の朝市で働く人々の様子を、具体的な描写を交えた日本語の長い随筆で書いてください。",
    ]
    AGENT = [
        "In flowing English with no lists, explain how you would debug a Kubernetes pod stuck in CrashLoopBackOff using kubectl.",
        "Write continuous prose describing how an agent should plan, call a search tool, and verify a citation before answering.",
        "In narrative English without bullets, walk through profiling a slow PostgreSQL query with EXPLAIN ANALYZE and fixing an index.",
        "Explain in essay form how you would review a pull request that changes a Rust async runtime's task scheduler.",
        "In prose, describe safely rotating a leaked API key across three services with zero downtime.",
    ]


def stream(prompt):
    body = {"model": "dsv41-flash-uva", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAXTOK, "temperature": 0, "seed": 42, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"thinking": False}}
    req = urllib.request.Request(B, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    first = last = None
    usage = {}
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            chs = obj.get("choices") or []
            if chs and (chs[0].get("delta") or {}).get("content"):
                now = time.time()
                first = first or now
                last = now
    comp = int(usage.get("completion_tokens") or 0)
    gen = (last - first) if (first and last and last > first) else None
    return {"t_start": t0, "t_end": time.time(), "tokens": comp,
            "tok_s": round((comp - 1) / gen, 2) if gen and comp > 1 else None}


rows: list = []
lock = threading.Lock()
T0 = time.time()
END = T0 + SECS


def worker(w):
    i = w
    while time.time() < END:
        dom = "jpde" if i % 2 == 0 else "agent"
        p = (JPDE if dom == "jpde" else AGENT)[(i // 2) % 5]
        try:
            rec = stream(p)
        except Exception as e:  # noqa: BLE001
            rec = {"t_start": time.time(), "t_end": time.time(), "tokens": 0, "tok_s": None, "error": repr(e)}
        rec.update({"dom": dom, "worker": w})
        with lock:
            rows.append(rec)
        i += 1


ths = [threading.Thread(target=worker, args=(w,)) for w in range(C)]
for t in ths:
    t.start()
for t in ths:
    t.join()
T1 = time.time()


def agg(sel, w0, w1):
    toks = sum(r["tokens"] for r in sel if w0 <= r["t_end"] <= w1)
    return round(toks / max(w1 - w0, 1e-6), 2)


post = [r for r in rows if r["t_start"] >= T0 + POST]
out = {
    "tag": tag, "concurrency": C, "seconds": SECS, "n": len(rows),
    "errors": sum(1 for r in rows if r.get("error")),
    "agg_tok_s_all": agg(rows, T0, T1),
    "agg_tok_s_post60": agg(rows, T0 + POST, T1),
    "per_req": {},
    "wall": [T0, T1],
    "rows": rows,
}
for dom in ("jpde", "agent"):
    a = [r["tok_s"] for r in rows if r["dom"] == dom and r["tok_s"]]
    p = [r["tok_s"] for r in post if r["dom"] == dom and r["tok_s"]]
    out["per_req"][dom] = {"median_all": statistics.median(a) if a else None,
                           "median_post60": statistics.median(p) if p else None, "n_post60": len(p)}
outdir.mkdir(parents=True, exist_ok=True)
(outdir / f"mixed-{tag}.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({k: out[k] for k in ("tag", "concurrency", "n", "errors", "agg_tok_s_all",
                                      "agg_tok_s_post60", "per_req")}), flush=True)
