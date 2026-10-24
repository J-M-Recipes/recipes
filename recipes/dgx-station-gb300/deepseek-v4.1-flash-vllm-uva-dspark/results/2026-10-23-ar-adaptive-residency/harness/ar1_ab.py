#!/usr/bin/env python3
"""AR1 in-lane measurements on the v26ar boot (frozen map, counting on).

  ab      same-boot A/B: counting on vs off (device flag via PIN_CTRL), C1,
          greedy, interleaved pairs. Measures the in-lane cost of the atomics.
  ttft    prefill sweep: unique random-word prompts (defeats prefix cache),
          max_tokens=1, TTFT per size. Sizes the follow-on P-stage card.
  sanity  telemetry check: selections per drain vs host steps.

Usage: ar1_ab.py <run_dir> <out.json> [pairs=6] [maxtok=512]
Stdlib only; talks to 127.0.0.1:30006.
"""
from __future__ import annotations

import json
import os
import random
import statistics
import sys
import time
import urllib.request
from pathlib import Path

RUN = Path(sys.argv[1])
OUT = Path(sys.argv[2])
PAIRS = int(sys.argv[3]) if len(sys.argv) > 3 else 6
MAXTOK = int(sys.argv[4]) if len(sys.argv) > 4 else 512
B = "http://127.0.0.1:30006/v1/chat/completions"
CTRL = RUN / "ctrl.json"

AGENT = [
    "You are Hermes. Write a continuous English prose explanation of how you would inspect a vLLM server with curl, no lists.",
    "In flowing English with no bullets, describe a careful git bisect of a CUDA graph regression in a MoE kernel.",
    "Write an essay-style walkthrough of a Python tool-calling agent loop: parse, dispatch, observe. No lists.",
]


def ctrl(**kw):
    tmp = CTRL.with_suffix(".tmp")
    tmp.write_text(json.dumps(kw))
    os.replace(tmp, CTRL)
    # bump mtime past the worker's last-seen value even within the same second
    t = time.time() + 0.01
    os.utime(CTRL, (t, t))
    time.sleep(2.5)  # worker polls ctrl once per second


def stream(prompt, n, stream_tokens=True):
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
    req = urllib.request.Request(B, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    first = last = None
    usage = {}
    with urllib.request.urlopen(req, timeout=1800) as resp:
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
            if chs and ((chs[0].get("delta") or {}).get("content")
                        or (chs[0].get("delta") or {}).get("reasoning_content")):
                now = time.monotonic()
                first = first or now
                last = now
    t1 = time.monotonic()
    comp = int(usage.get("completion_tokens") or 0)
    gen = (last - first) if (first and last and last > first) else None
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": comp,
        "ttft_s": None if first is None else round(first - t0, 4),
        "wall_s": round(t1 - t0, 4),
        "tok_s": round((comp - 1) / gen, 3) if gen and comp > 1 else None,
    }


def ab():
    rows = []
    stream("hi", 8)
    order = []
    for i in range(PAIRS):
        order += (["on", "off"] if i % 2 == 0 else ["off", "on"])
    for k, arm in enumerate(order):
        ctrl(count=(arm == "on"))
        rec = stream(AGENT[0], MAXTOK)  # one prompt for every run: same tokens
        rec["arm"] = arm
        rows.append(rec)
        print(f"[ab] {k:2d} {arm:3s} tok/s={rec['tok_s']} tok={rec['completion_tokens']}", flush=True)
    ctrl(count=True)
    on = [r["tok_s"] for r in rows if r["arm"] == "on" and r["tok_s"]]
    off = [r["tok_s"] for r in rows if r["arm"] == "off" and r["tok_s"]]
    med_on, med_off = statistics.median(on), statistics.median(off)
    # paired deltas in run order (each pair = adjacent on/off)
    pd = []
    for i in range(0, len(rows) - 1, 2):
        a, b = rows[i], rows[i + 1]
        x = a if a["arm"] == "on" else b
        y = b if a["arm"] == "on" else a
        if x["tok_s"] and y["tok_s"]:
            pd.append(100.0 * (x["tok_s"] - y["tok_s"]) / y["tok_s"])
    return {
        "rows": rows,
        "median_on": med_on,
        "median_off": med_off,
        "delta_pct_median": round(100.0 * (med_on - med_off) / med_off, 3),
        "paired_delta_pct": [round(v, 3) for v in pd],
        "paired_delta_pct_median": round(statistics.median(pd), 3) if pd else None,
        "spread_off_pct": round(100.0 * (max(off) - min(off)) / med_off, 3),
        "same_tokens": len({r["completion_tokens"] for r in rows}) == 1,
    }


WORDS = ("river stone lamp orbit cedar quartz ember vector harbor signal meadow "
         "lattice copper falcon tunnel glacier saddle prism thread anvil beacon "
         "canyon delta fjord grove hollow island jetty kernel ledger mortar nectar "
         "oasis pylon quiver ravine summit timber umber valley willow yonder zephyr").split()


def ttft():
    rows = []
    for words in (1500, 6000, 24000):
        for rep in range(2):
            rnd = random.Random(1000 * words + rep + int(time.time()))
            text = " ".join(rnd.choice(WORDS) for _ in range(words))
            prompt = (f"[{rnd.random():.12f}] Count how many times the word 'ember' appears "
                      f"in the following text and answer with one number.\n\n{text}")
            rec = stream(prompt, 1)
            rec["words"] = words
            rec["rep"] = rep
            rec["prefill_tok_s"] = (round(rec["prompt_tokens"] / rec["ttft_s"], 1)
                                    if rec["ttft_s"] else None)
            rows.append(rec)
            print(f"[ttft] words={words} rep={rep} prompt_tok={rec['prompt_tokens']} "
                  f"ttft={rec['ttft_s']}s prefill={rec['prefill_tok_s']} tok/s", flush=True)
    return {"rows": rows}


def sanity():
    p = RUN / "telemetry.jsonl"
    if not p.exists():
        return {"ok": False, "why": "no telemetry"}
    recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    drains = [r for r in recs if r.get("dsteps")]
    if not drains:
        return {"ok": False, "why": "no drains", "n": len(recs)}
    per_step = [r["sel"] / r["dsteps"] for r in drains if r["sel"]]
    return {
        "ok": True,
        "n_drains": len(drains),
        "sel_per_step_median": round(statistics.median(per_step), 1) if per_step else None,
        "cum_cold_share": drains[-1].get("cum_cold_share"),
        "last": drains[-1],
    }


if __name__ == "__main__":
    res = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    res["ab"] = ab()
    print(f"[ab] median on={res['ab']['median_on']} off={res['ab']['median_off']} "
          f"delta={res['ab']['delta_pct_median']}% paired={res['ab']['paired_delta_pct_median']}%",
          flush=True)
    res["ttft"] = ttft()
    res["sanity"] = sanity()
    print(f"[sanity] {json.dumps(res['sanity'])[:400]}", flush=True)
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print(f"[ar1] wrote {OUT}", flush=True)
