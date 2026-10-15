#!/usr/bin/env python3
"""G1: same server, same prompts, four decode regimes. Per-position acceptance each."""
import json, statistics, subprocess, sys, time, urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8888"; MODEL = "glm-5.3-flash"
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
PROMPTS = [
    ("probe_prose0", "Write a detailed 700-word explanation of how Roman aqueducts maintained gradient and water quality."),
    ("probe_prose1", "Explain speculative decoding to an inference engineer, including verification, acceptance length, and failure modes."),
    ("probe_code0", "Write a complete Python module implementing a thread-safe LRU cache with TTL expiry, type hints, docstrings, and tests. Output only code."),
    ("probe_code1", "Write a complete Rust implementation of a bounded work queue with cancellation and unit tests. Output only code."),
    ("e1_prose", "Write a 300-word essay on how tidal locking shapes planetary climates."),
]
REGIMES = {
    "think_off_t0":  {"temperature": 0, "top_p": 1, "chat_template_kwargs": {"enable_thinking": False}},
    "reason_low_t0": {"temperature": 0, "top_p": 1, "chat_template_kwargs": {"reasoning_effort": "low"}},
    "think_on_t0":   {"temperature": 0, "top_p": 1, "chat_template_kwargs": {"enable_thinking": True}},
    "think_on_t1":   {"temperature": 1.0, "top_p": 0.95, "chat_template_kwargs": {"enable_thinking": True}},
}
M = {"drafts": "vllm:spec_decode_num_drafts_total", "draft_tokens": "vllm:spec_decode_num_draft_tokens_total", "accepted": "vllm:spec_decode_num_accepted_tokens_total"}
PP = "vllm:spec_decode_num_accepted_tokens_per_pos_total"

def metrics():
    t = urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode()
    out = {k: 0.0 for k in M}; pp = {}
    for line in t.splitlines():
        for k, m in M.items():
            if line.startswith(m + "{") or line.startswith(m + " "):
                out[k] += float(line.rsplit(" ", 1)[1])
        if line.startswith(PP + "{"):
            lab = line[len(PP) + 1: line.index("}")]
            pos = [p.split("=")[1].strip('"') for p in lab.split(",") if p.split("=")[0].strip() == "position"][0]
            pp[pos] = pp.get(pos, 0.0) + float(line.rsplit(" ", 1)[1])
    out["pp"] = pp; return out

def one(name, prompt, regime, cfg, max_tokens=512):
    payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "stream": False, **cfg}
    b = metrics(); t0 = time.monotonic()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    body = json.loads(urllib.request.urlopen(req, timeout=1800).read().decode())
    wall = time.monotonic() - t0; time.sleep(1); a = metrics()
    u = body.get("usage") or {}; comp = int(u.get("completion_tokens") or 0)
    d = {k: a[k] - b[k] for k in M}
    ppd = {p: a["pp"].get(p, 0) - b["pp"].get(p, 0) for p in sorted(set(a["pp"]) | set(b["pp"]), key=int)}
    msg = (body.get("choices") or [{}])[0].get("message") or {}
    row = {"name": name, "regime": regime, "completion_tokens": comp, "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
           "wall": round(wall, 3), "tok_s": round(comp / wall, 3) if comp else None, "drafts": d["drafts"],
           "acceptance_length": (comp / d["drafts"]) if d["drafts"] else None,
           "accept_rate": (d["accepted"] / d["draft_tokens"]) if d["draft_tokens"] else None,
           "per_pos": {p: round(v / d["drafts"], 3) for p, v in ppd.items()} if d["drafts"] else {},
           "finish": (body.get("choices") or [{}])[0].get("finish_reason"),
           "has_reasoning": bool(msg.get("reasoning_content") or msg.get("reasoning")),
           "content_head": (msg.get("content") or "")[:120]}
    (OUT / f"{regime}__{name}.json").write_text(json.dumps({"row": row, "payload": payload, "body": body}, indent=2))
    print("ROW " + json.dumps(row), flush=True); return row

rows = []
for regime, cfg in REGIMES.items():
    for name, prompt in PROMPTS:
        rows.append(one(name, prompt, regime, cfg))
summary = {}
for regime in REGIMES:
    rr = [r for r in rows if r["regime"] == regime and r["acceptance_length"]]
    prose = [r["acceptance_length"] for r in rr if "prose" in r["name"]]
    summary[regime] = {"prose_accept_median": statistics.median(prose) if prose else None,
                       "all_accept_median": statistics.median([r["acceptance_length"] for r in rr]) if rr else None,
                       "prose_tok_s_median": statistics.median([r["tok_s"] for r in rr if "prose" in r["name"] and r["tok_s"]]) if prose else None}
(OUT / "summary.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
print("SUMMARY " + json.dumps(summary), flush=True)
