#!/usr/bin/env python3
"""replay_sessions.py — session-ordered replay. N concurrent sessions, turns serial within each session (growing prefix),
same request shape as replay2.py. Reports prefix-cache hit rate, TTFT per turn index, agg tok/s with bootstrap CI.
Env: TAG, N (concurrent sessions, default 4), FIX (default transcript_fixture_sessions.json), MAXTOK (400).
Writes /home/milo/dsv41/replays-<TAG>-n<N>.json."""
import json, os, time, urllib.request, threading, queue, random, statistics as st
base = "http://127.0.0.1:30006/v1"; model = "dsv41-flash-uva"
tag = os.environ["TAG"]; N = int(os.getenv("N", "4")); FIX = os.getenv("FIX", "/home/milo/dsv41/transcript_fixture_sessions.json"); MAXTOK = int(os.getenv("MAXTOK", "400"))
H = {"Content-Type": "application/json"}
SYS = ("You are Milo, James's operations agent. Be terse. Use tools when needed. Quote CDT times. "
       "Never rephrase a blocked command. Prefer tables for numbers.")
TOOLS = [{"type": "function", "function": {"name": "terminal", "description": "Run a shell command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}}, "required": ["command"]}}},
         {"type": "function", "function": {"name": "read_file", "description": "Read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, "required": ["path"]}}},
         {"type": "function", "function": {"name": "web_search", "description": "Search the web", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}]
WANT = {"vllm:spec_decode_num_draft_tokens_total": "draft", "vllm:spec_decode_num_accepted_tokens_total": "acc", "vllm:spec_decode_num_drafts_total": "steps",
        "vllm:prefix_cache_queries_total": "pc_q", "vllm:prefix_cache_hits_total": "pc_h", "vllm:prompt_tokens_total": "prompt_tok", "vllm:generation_tokens_total": "gen_tok"}
def metrics():
    t = urllib.request.urlopen(base.replace("/v1", "") + "/metrics", timeout=10).read().decode(); d = {}
    for l in t.splitlines():
        for k, v in WANT.items():
            if l.startswith(k): d[v] = d.get(v, 0.0) + float(l.split()[-1])
    return d
def chat(msgs):
    p = {"model": model, "messages": [{"role": "system", "content": SYS}] + msgs, "max_tokens": MAXTOK, "temperature": 0, "tools": TOOLS, "tool_choice": "auto",
         "chat_template_kwargs": {"thinking": False}, "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.monotonic(); ttft = None; usage = None; tc = False
    with urllib.request.urlopen(urllib.request.Request(base + "/chat/completions", data=json.dumps(p).encode(), headers=H), timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:"): continue
            body = line[5:].strip()
            if body == "[DONE]": break
            ev = json.loads(body)
            if ev.get("usage"): usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                if (d.get("content") or d.get("tool_calls")) and ttft is None: ttft = time.monotonic() - t0
                if d.get("tool_calls"): tc = True
    dt = time.monotonic() - t0
    return (usage or {}).get("completion_tokens", 0), (usage or {}).get("prompt_tokens", 0), dt, ttft or dt, tc
S = json.load(open(FIX)); chat(S[0]["turns"][0]["ctx"][-2:])
q = queue.Queue(); [q.put(s) for s in S]
lock = threading.Lock(); recs = []; errs = 0
def worker():
    global errs
    while True:
        try: s = q.get_nowait()
        except queue.Empty: return
        for j, t in enumerate(s["turns"]):
            try: tok, ptok, dt, ttft, tc = chat(t["ctx"])
            except Exception as e:
                with lock: errs += 1; print(f"[{tag}] {s['session']} turn {j} ERROR {str(e)[:80]}", flush=True)
                continue
            with lock: recs.append({"sess": s["session"], "j": j, "cls": "tool" if t["has_tool"] else "text", "tok": tok, "ptok": ptok, "s": round(dt, 3), "ttft": round(ttft, 3), "emitted": tc})
m0 = metrics(); T0 = time.monotonic()
th = [threading.Thread(target=worker) for _ in range(N)]; [t.start() for t in th]; [t.join() for t in th]
wall = time.monotonic() - T0; m1 = metrics(); dm = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
tot = sum(r["tok"] for r in recs); random.seed(1)
def agg_of(rs): return sum(r["tok"] for r in rs) / max(1e-9, sum(r["s"] for r in rs)) * N
boots = sorted(agg_of(random.choices(recs, k=len(recs))) for _ in range(2000)); ci = (round(boots[50], 1), round(boots[1949], 1))
def ttft_by_j():
    out = {}
    for j in sorted(set(r["j"] for r in recs)):
        v = sorted(r["ttft"] for r in recs if r["j"] == j); out[j] = {"n": len(v), "med": round(v[len(v) // 2], 2), "p90": round(v[int(0.9 * (len(v) - 1))], 2), "ptok_mean": round(st.mean(r["ptok"] for r in recs if r["j"] == j))}
    return out
out = {"tag": tag, "fixture": os.path.basename(FIX), "sessions": len(S), "turns": len(recs), "n_workers": N, "errors": errs, "wall_s": round(wall, 1),
       "agg_tok_s": round(tot / wall, 1), "agg_ci95_boot": ci, "tokens": tot, "accept": round(dm["acc"] / max(1, dm["draft"]), 3), "acc_per_step": round(dm["acc"] / max(1, dm["steps"]), 2),
       "prefix_cache_hit_rate": round(dm["pc_h"] / max(1, dm["pc_q"]), 3), "prompt_tokens": int(dm["prompt_tok"]), "gen_tokens": int(dm["gen_tok"]),
       "ttft_all_med": round(sorted(r["ttft"] for r in recs)[len(recs) // 2], 2), "ttft_all_p90": round(sorted(r["ttft"] for r in recs)[int(0.9 * (len(recs) - 1))], 2),
       "ttft_by_turn_index": ttft_by_j(), "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
print(f"[{tag}] SESSIONS n={N}: {len(S)} sessions / {len(recs)} turns: agg {out['agg_tok_s']} tok/s (CI95 {ci[0]}–{ci[1]}) over {wall:.0f}s; accept={out['accept']:.1%} acc/step={out['acc_per_step']}; "
      f"prefix-cache hit {out['prefix_cache_hit_rate']:.1%} on {out['prompt_tokens']} prompt tok; ttft med/p90 {out['ttft_all_med']}/{out['ttft_all_p90']}s; "
      f"ttft by turn: " + " ".join(f"t{j}={v['med']}" for j, v in out['ttft_by_turn_index'].items()) + f"; errors {errs}", flush=True)
out["records"] = recs; json.dump(out, open(f"/home/milo/dsv41/replays-{tag}-n{N}.json", "w"), indent=1)
