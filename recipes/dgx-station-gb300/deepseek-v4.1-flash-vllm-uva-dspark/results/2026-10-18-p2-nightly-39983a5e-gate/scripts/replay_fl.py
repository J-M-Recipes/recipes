#!/usr/bin/env python3
"""replay_fl.py — Card P2b sibling of $HOME/dsv41/replay_c.py (v1 is NOT edited).
Forced-length warm replay: same real Hermes turns (transcript_fixture.json), same SYS/TOOLS/temperature/N/ROUNDS as
replay_c.py, but every request is forced to emit exactly FL tokens (max_tokens=min_tokens=FL, ignore_eos=true), so every
arm does identical decode work regardless of trajectory. Streams to time TTFT and TPOT per request.
A per-arm SALT (vLLM cache_salt) isolates the prefix cache: with the same SALT, run r1 round 0 is cold, r1 round 1 and all
of r2 are warm — the same structure as replay_c r1/r2, but cold-start is guaranteed on a live lane too.
Env: TAG (required), OUT (json path, required), SALT (required), N=4, ROUNDS=2, FL=256.
"""
import json, os, time, urllib.request, threading, queue, statistics as st
base = "http://127.0.0.1:30006/v1"; model = "dsv41-flash-uva"
tag = os.environ["TAG"]; OUT = os.environ["OUT"]; SALT = os.environ["SALT"]
N = int(os.getenv("N", "4")); ROUNDS = int(os.getenv("ROUNDS", "2")); FL = int(os.getenv("FL", "256"))
H = {"Content-Type": "application/json"}
SYS = ("You are Milo, James's operations agent. Be terse. Use tools when needed. Quote CDT times. "
       "Never rephrase a blocked command. Prefer tables for numbers.")
TOOLS = [{"type": "function", "function": {"name": "terminal", "description": "Run a shell command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}}, "required": ["command"]}}},
         {"type": "function", "function": {"name": "read_file", "description": "Read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, "required": ["path"]}}},
         {"type": "function", "function": {"name": "web_search", "description": "Search the web", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}]

def metrics():
    t = urllib.request.urlopen(base.replace("/v1", "") + "/metrics", timeout=10).read().decode(); d = {"draft": 0.0, "acc": 0.0, "steps": 0.0}
    for l in t.splitlines():
        if l.startswith("vllm:spec_decode_num_draft_tokens_total"): d["draft"] = float(l.split()[-1])
        elif l.startswith("vllm:spec_decode_num_accepted_tokens_total"): d["acc"] = float(l.split()[-1])
        elif l.startswith("vllm:spec_decode_num_drafts_total"): d["steps"] = float(l.split()[-1])
    return d

def chat_stream(msgs, n):
    p = {"model": model, "messages": [{"role": "system", "content": SYS}] + msgs, "max_tokens": n, "min_tokens": n,
         "ignore_eos": True, "temperature": 0, "tools": TOOLS, "tool_choice": "auto",
         "chat_template_kwargs": {"thinking": False}, "cache_salt": SALT,
         "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.monotonic(); tfirst = None; tlast = None; usage = None
    r = urllib.request.urlopen(urllib.request.Request(base + "/chat/completions", data=json.dumps(p).encode(), headers=H), timeout=900)
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data:"): continue
        data = line[5:].strip()
        if data == "[DONE]": break
        ev = json.loads(data)
        now = time.monotonic()
        if ev.get("choices"):
            if tfirst is None: tfirst = now
            tlast = now
        if ev.get("usage"): usage = ev["usage"]
    t1 = time.monotonic()
    ct = (usage or {}).get("completion_tokens"); pt = (usage or {}).get("prompt_tokens")
    ttft = (tfirst - t0) if tfirst else None
    tpot = ((tlast - tfirst) / (ct - 1)) if (tfirst and ct and ct > 1) else None
    return ct, pt, t1 - t0, ttft, tpot

S = json.load(open("$HOME/dsv41/transcript_fixture.json"))
q = queue.Queue()
for rd in range(ROUNDS):
    for i, s in enumerate(S): q.put((rd, i, s))
lock = threading.Lock(); rows = []; errs = []
def worker():
    while True:
        try: rd, i, s = q.get_nowait()
        except queue.Empty: return
        try: ct, pt, dt, ttft, tpot = chat_stream(s["ctx"], FL)
        except Exception as e:
            with lock: errs.append(f"turn {i} round {rd}: {e}"); print(f"[{tag}] turn {i} ERROR {e}", flush=True)
            continue
        with lock: rows.append({"round": rd, "turn": i, "tool": bool(s["has_tool"]), "ct": ct, "pt": pt, "s": round(dt, 4),
                                "ttft": round(ttft, 4) if ttft is not None else None, "tpot": round(tpot, 5) if tpot is not None else None})
m0 = metrics(); t0 = time.monotonic()
th = [threading.Thread(target=worker) for _ in range(N)]
[t.start() for t in th]; [t.join() for t in th]
wall = time.monotonic() - t0; m1 = metrics()
tot = sum(r["ct"] or 0 for r in rows)
def med(xs): xs = [x for x in xs if x is not None]; return round(st.median(xs), 4) if xs else None
def mean(xs): xs = [x for x in xs if x is not None]; return round(st.mean(xs), 5) if xs else None
out = {"tag": tag, "salt": SALT, "fl": FL, "n_workers": N, "rounds": ROUNDS, "wall_s": round(wall, 2), "agg_tok_s": round(tot / wall, 1),
       "tokens": tot, "requests": len(rows), "errors": errs, "all_exact_fl": all(r["ct"] == FL for r in rows) and len(rows) == ROUNDS * len(S),
       "tpot_mean_ms": round(1000 * mean([r["tpot"] for r in rows]), 2) if rows else None,
       "tpot_median_ms": round(1000 * med([r["tpot"] for r in rows]), 2) if rows else None,
       "per_req_tok_s_mean": round(st.mean([r["ct"] / r["s"] for r in rows if r["ct"] and r["s"]]), 1) if rows else None,
       "ttft_median_s": med([r["ttft"] for r in rows]),
       "ttft_median_round0_s": med([r["ttft"] for r in rows if r["round"] == 0]),
       "ttft_median_round1_s": med([r["ttft"] for r in rows if r["round"] == 1]),
       "accept": round((m1["acc"] - m0["acc"]) / max(1, m1["draft"] - m0["draft"]), 3),
       "acc_per_step": round((m1["acc"] - m0["acc"]) / max(1, m1["steps"] - m0["steps"]), 2),
       "rows": sorted(rows, key=lambda r: (r["round"], r["turn"])), "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
print(f"[{tag}] REPLAY-FL n={N} fl={FL}: agg {out['agg_tok_s']} tok/s over {wall:.1f}s, {tot} tokens, exact={out['all_exact_fl']}; "
      f"TPOT mean {out['tpot_mean_ms']} ms med {out['tpot_median_ms']} ms; TTFT med r0 {out['ttft_median_round0_s']} r1 {out['ttft_median_round1_s']}; "
      f"accept={out['accept']:.1%} acc/step={out['acc_per_step']}", flush=True)
json.dump(out, open(OUT, "w"), indent=1)
