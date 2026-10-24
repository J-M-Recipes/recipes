#!/usr/bin/env python3
"""E4 domain-shift protocol (AR copy: + t_wall per record and per-phase wall bounds).

(a) 3 min JP/DE essays C1, sample tok/s every 30 s
(b) 3 min agent-like English C1, same sampling
(c) 3 min JP/DE again

Usage: e4_shift.py <tag> [outdir] [phase_s=180]
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

tag = sys.argv[1]
outdir = Path(sys.argv[2] if len(sys.argv) > 2 else "$HOME/pin-hot-experts/e4")
phase_s = float(sys.argv[3] if len(sys.argv) > 3 else 180)
outdir.mkdir(parents=True, exist_ok=True)
B = "http://127.0.0.1:30006/v1/chat/completions"
MAXTOK = 400

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


def stream(prompt, n=MAXTOK):
    body = {
        "model": "dsv41-flash-uva",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": n,
        "temperature": 0,
        "seed": 42,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"thinking": False},
    }
    req = urllib.request.Request(
        B, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t_open = time.monotonic()
    first = last = None
    chunks = []
    usage = {}
    finish = None
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
            if not chs:
                continue
            ch = chs[0]
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
            delta = ch.get("delta") or {}
            piece = delta.get("content") or ""
            if piece:
                now = time.monotonic()
                if first is None:
                    first = now
                last = now
                chunks.append(piece)
    t_end = time.monotonic()
    text = "".join(chunks)
    comp = int(usage.get("completion_tokens") or 0)
    wall = t_end - t_open
    gen_dt = None if first is None or last is None else max(last - first, 1e-6)
    gen_tps = (comp / gen_dt) if gen_dt and comp else None
    wall_tps = (comp / wall) if wall and comp else None
    instrument_tps = (
        gen_tps
        if (first is not None and last is not None and (last - first) >= 0.05)
        else wall_tps
    )
    return {
        "prompt_head": prompt[:80],
        "completion_tokens": comp,
        "finish": finish,
        "wall_s": round(wall, 3),
        "ttft_s": None if first is None else round(first - t_open, 3),
        "first_to_last_s": None if gen_dt is None else round(gen_dt, 3),
        "instrument_tok_s": None if instrument_tps is None else round(instrument_tps, 2),
        "n_chars": len(text),
        "text_head": text[:80].replace("\n", " "),
        "t_mono": t_open,
        "t_wall": time.time(),
    }


def phase(name, prompts, seconds):
    t0 = time.monotonic()
    w0 = time.time()
    samples = []
    last_sample = -1e9
    i = 0
    print(f"[shift] PHASE {name} {seconds}s", flush=True)
    while time.monotonic() - t0 < seconds:
        rec = stream(prompts[i % len(prompts)])
        i += 1
        now = time.monotonic()
        rec["elapsed_s"] = round(now - t0, 1)
        rec["phase"] = name
        if now - last_sample >= 30 or not samples:
            samples.append(rec)
            last_sample = now
            print(
                f"[shift] {name} t={rec['elapsed_s']:6.1f}s tok/s={rec['instrument_tok_s']} "
                f"tok={rec['completion_tokens']}",
                flush=True,
            )
        else:
            samples.append(rec)
    return {
        "name": name,
        "wall_start": w0,
        "wall_end": time.time(),
        "seconds": seconds,
        "n": len(samples),
        "samples": samples,
        "first_tok_s": samples[0]["instrument_tok_s"] if samples else None,
        "last_tok_s": samples[-1]["instrument_tok_s"] if samples else None,
        "sampled_30s": [s for s in samples if True][:],
    }


stream("hi", 8)
out = {
    "tag": tag,
    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "phase_s": phase_s,
    "phases": {},
}
out["phases"]["jpde_1"] = phase("jpde_1", JPDE, phase_s)
out["phases"]["agent"] = phase("agent", AGENT, phase_s)
out["phases"]["jpde_2"] = phase("jpde_2", JPDE, phase_s)
path = outdir / f"shift-{tag}.json"
path.write_text(json.dumps(out, indent=1) + "\n")
print(f"[shift] wrote {path}", flush=True)
print(
    json.dumps(
        {
            k: {
                "first": v["first_tok_s"],
                "last": v["last_tok_s"],
                "n": v["n"],
            }
            for k, v in out["phases"].items()
        }
    ),
    flush=True,
)
