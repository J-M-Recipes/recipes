#!/usr/bin/env python3
"""replay_c.py — concurrent replay of real Hermes turns (transcript_fixture.json) against :30006.
N workers each pull turns from a shared queue; measures wall tok/s (sum completion tokens / wall), per-class
tok/s, DSpark acceptance via /metrics delta over the whole run, and tool_calls emitted on tool turns.
Env: TAG (required), N (workers, default 4), ROUNDS (passes over the fixture, default 2).
Writes /home/milo/dsv41/replayc-<TAG>-n<N>.json.
"""
import json, os, sys, time, urllib.request, threading, queue
base = "http://127.0.0.1:30006/v1"; model = "dsv41-flash-uva"
tag = os.environ["TAG"]; N = int(os.getenv("N", "4")); ROUNDS = int(os.getenv("ROUNDS", "2"))
H = {"Content-Type": "application/json"}
SYS = ("You are Milo, James's operations agent. Be terse. Use tools when needed. Quote CDT times. "
       "Never rephrase a blocked command. Prefer tables for numbers.")
TOOLS = [{"type": "function", "function": {"name": "terminal", "description": "Run a shell command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}}, "required": ["command"]}}},
         {"type": "function", "function": {"name": "read_file", "description": "Read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, "required": ["path"]}}},
         {"type": "function", "function": {"name": "web_search", "description": "Search the web", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}]

def metrics():
    t = urllib.request.urlopen(base.replace("/v1", "") + "/metrics", timeout=10).read().decode(); d = {}
    for l in t.splitlines():
        if l.startswith("vllm:spec_decode_num_draft_tokens_total"): d["draft"] = float(l.split()[-1])
        elif l.startswith("vllm:spec_decode_num_accepted_tokens_total"): d["acc"] = float(l.split()[-1])
        elif l.startswith("vllm:spec_decode_num_drafts_total"): d["steps"] = float(l.split()[-1])
    return d

def chat(msgs, max_tokens=400):
    p = {"model": model, "messages": [{"role": "system", "content": SYS}] + msgs, "max_tokens": max_tokens, "temperature": 0,
         "tools": TOOLS, "tool_choice": "auto", "chat_template_kwargs": {"thinking": False}}
    t0 = time.monotonic()
    r = json.load(urllib.request.urlopen(urllib.request.Request(base + "/chat/completions", data=json.dumps(p).encode(), headers=H), timeout=900))
    ch = r["choices"][0]
    return r["usage"]["completion_tokens"], time.monotonic() - t0, bool(ch["message"].get("tool_calls"))

S = json.load(open("/home/milo/dsv41/transcript_fixture.json"))
chat(S[0]["ctx"][-2:], 8)  # warm
q = queue.Queue()
for _ in range(ROUNDS):
    for i, s in enumerate(S): q.put((i, s))
lock = threading.Lock()
res = {"tool": {"tok": 0, "s": 0.0, "n": 0, "emitted": 0}, "text": {"tok": 0, "s": 0.0, "n": 0, "emitted": 0}}
def worker():
    while True:
        try: i, s = q.get_nowait()
        except queue.Empty: return
        k = "tool" if s["has_tool"] else "text"
        try: tok, dt, tc = chat(s["ctx"])
        except Exception as e:
            with lock: print(f"[{tag}] turn {i} ERROR {e}", flush=True)
            continue
        with lock:
            r = res[k]; r["tok"] += tok; r["s"] += dt; r["n"] += 1; r["emitted"] += tc
m0 = metrics(); t0 = time.monotonic()
th = [threading.Thread(target=worker) for _ in range(N)]
[t.start() for t in th]; [t.join() for t in th]
wall = time.monotonic() - t0; m1 = metrics()
tot = res["tool"]["tok"] + res["text"]["tok"]
acc = (m1["acc"] - m0["acc"]) / max(1, m1["draft"] - m0["draft"])
aps = (m1["acc"] - m0["acc"]) / max(1, m1["steps"] - m0["steps"])
out = {"tag": tag, "n_workers": N, "rounds": ROUNDS, "wall_s": round(wall, 1), "agg_tok_s": round(tot / wall, 1),
       "accept": round(acc, 3), "acc_per_step": round(aps, 2), "tokens": tot,
       "tool": {**res["tool"], "per_req_tok_s": round(res["tool"]["tok"] / max(1e-9, res["tool"]["s"]) * 1, 1)},
       "text": {**res["text"], "per_req_tok_s": round(res["text"]["tok"] / max(1e-9, res["text"]["s"]), 1)},
       "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
print(f"[{tag}] REPLAY n={N}: agg {out['agg_tok_s']} tok/s over {wall:.0f}s, {tot} tokens; accept={acc:.1%} acc/step={aps:.2f}; "
      f"tool turns {res['tool']['n']} emitted={res['tool']['emitted']} per-req {out['tool']['per_req_tok_s']} tok/s; "
      f"text turns {res['text']['n']} per-req {out['text']['per_req_tok_s']} tok/s", flush=True)
json.dump(out, open(f"/home/milo/dsv41/replayc-{tag}-n{N}.json", "w"), indent=1)
