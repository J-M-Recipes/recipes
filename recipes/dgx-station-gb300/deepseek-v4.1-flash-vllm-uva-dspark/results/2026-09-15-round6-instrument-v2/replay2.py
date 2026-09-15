#!/usr/bin/env python3
"""replay2.py — v2 of the real-transcript replay instrument. Same protocol as replay_c.py, plus:
  * per-request records (tokens, wall, TTFT via streaming, prompt tokens, class) → bootstrap 95% CI on agg tok/s
  * prefix-cache hit rate and prompt-token totals from /metrics deltas (what agents actually wait on)
  * TTFT median/p90 per class
Env: TAG (required), N (workers, default 4), ROUNDS (default 1), FIX (fixture path, default transcript_fixture_v2.json),
     LIMIT (use first LIMIT turns, default all), MAXTOK (default 400).
Writes /home/milo/dsv41/replay2-<TAG>-n<N>.json.
"""
import json, os, sys, time, urllib.request, threading, queue, random, statistics as st
base = "http://127.0.0.1:30006/v1"; model = "dsv41-flash-uva"
tag = os.environ["TAG"]; N = int(os.getenv("N", "4")); ROUNDS = int(os.getenv("ROUNDS", "1"))
FIX = os.getenv("FIX", "/home/milo/dsv41/transcript_fixture_v2.json"); LIMIT = int(os.getenv("LIMIT", "0")); MAXTOK = int(os.getenv("MAXTOK", "400"))
H = {"Content-Type": "application/json"}
SYS = ("You are Milo, James's operations agent. Be terse. Use tools when needed. Quote CDT times. "
       "Never rephrase a blocked command. Prefer tables for numbers.")
TOOLS = [{"type": "function", "function": {"name": "terminal", "description": "Run a shell command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}}, "required": ["command"]}}},
         {"type": "function", "function": {"name": "read_file", "description": "Read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, "required": ["path"]}}},
         {"type": "function", "function": {"name": "web_search", "description": "Search the web", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}]
WANT = {"vllm:spec_decode_num_draft_tokens_total": "draft", "vllm:spec_decode_num_accepted_tokens_total": "acc",
        "vllm:spec_decode_num_drafts_total": "steps", "vllm:prefix_cache_queries_total": "pc_q", "vllm:prefix_cache_hits_total": "pc_h",
        "vllm:prompt_tokens_total": "prompt_tok", "vllm:generation_tokens_total": "gen_tok"}
def metrics():
    t = urllib.request.urlopen(base.replace("/v1", "") + "/metrics", timeout=10).read().decode(); d = {}
    for l in t.splitlines():
        for k, v in WANT.items():
            if l.startswith(k): d[v] = d.get(v, 0.0) + float(l.split()[-1])
    return d

def chat(msgs):
    p = {"model": model, "messages": [{"role": "system", "content": SYS}] + msgs, "max_tokens": MAXTOK, "temperature": 0,
         "tools": TOOLS, "tool_choice": "auto", "chat_template_kwargs": {"thinking": False},
         "stream": True, "stream_options": {"include_usage": True}}
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

S = json.load(open(FIX)); S = S[:LIMIT] if LIMIT else S
chat(S[0]["ctx"][-2:])  # warm
q = queue.Queue()
for _ in range(ROUNDS):
    for i, s in enumerate(S): q.put((i, s))
lock = threading.Lock(); recs = []; errs = 0
def worker():
    global errs
    while True:
        try: i, s = q.get_nowait()
        except queue.Empty: return
        try: tok, ptok, dt, ttft, tc = chat(s["ctx"])
        except Exception as e:
            with lock: errs += 1; print(f"[{tag}] turn {i} ERROR {str(e)[:80]}", flush=True)
            continue
        with lock: recs.append({"i": i, "cls": "tool" if s["has_tool"] else "text", "tok": tok, "ptok": ptok, "s": round(dt, 3), "ttft": round(ttft, 3), "emitted": tc, "t_end": round(time.monotonic() - T0, 2)})
m0 = metrics(); T0 = time.monotonic()
th = [threading.Thread(target=worker) for _ in range(N)]
[t.start() for t in th]; [t.join() for t in th]
wall = time.monotonic() - T0; m1 = metrics(); dm = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
tot = sum(r["tok"] for r in recs)
# bootstrap CI on aggregate tok/s: resample requests, agg = sum(tok)/sum(s)*N (N workers busy ≈ wall) — use wall-proportional estimator
random.seed(1)
def agg_of(rs): return sum(r["tok"] for r in rs) / max(1e-9, sum(r["s"] for r in rs)) * N
boots = sorted(agg_of(random.choices(recs, k=len(recs))) for _ in range(2000))
ci = (round(boots[int(0.025 * len(boots))], 1), round(boots[int(0.975 * len(boots))], 1))
def cls(k):
    rs = [r for r in recs if r["cls"] == k]
    if not rs: return {}
    tt = sorted(r["ttft"] for r in rs)
    return {"n": len(rs), "tok": sum(r["tok"] for r in rs), "per_req_tok_s": round(sum(r["tok"] for r in rs) / max(1e-9, sum(r["s"] for r in rs)), 1),
            "ttft_med": round(tt[len(tt) // 2], 2), "ttft_p90": round(tt[int(0.9 * (len(tt) - 1))], 2), "emitted": sum(r["emitted"] for r in rs),
            "ptok_mean": round(st.mean(r["ptok"] for r in rs))}
out = {"tag": tag, "fixture": os.path.basename(FIX), "turns": len(S), "n_workers": N, "rounds": ROUNDS, "errors": errs,
       "wall_s": round(wall, 1), "agg_tok_s": round(tot / wall, 1), "agg_ci95_boot": ci, "agg_est_boot": round(agg_of(recs), 1), "tokens": tot,
       "accept": round(dm["acc"] / max(1, dm["draft"]), 3), "acc_per_step": round(dm["acc"] / max(1, dm["steps"]), 2),
       "prefix_cache_hit_rate": round(dm["pc_h"] / max(1, dm["pc_q"]), 3), "prompt_tokens": int(dm["prompt_tok"]), "gen_tokens": int(dm["gen_tok"]),
       "prefill_share_est": round(dm["prompt_tok"] / max(1, dm["prompt_tok"] + dm["gen_tok"]), 3),
       "tool": cls("tool"), "text": cls("text"), "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
print(f"[{tag}] REPLAY2 n={N} turns={len(S)}: agg {out['agg_tok_s']} tok/s (boot est {out['agg_est_boot']} CI95 {ci[0]}–{ci[1]}) over {wall:.0f}s, {tot} tok; "
      f"accept={out['accept']:.1%} acc/step={out['acc_per_step']}; prefix-cache hit {out['prefix_cache_hit_rate']:.1%} on {out['prompt_tokens']} prompt tok; "
      f"tool ttft med/p90 {out['tool'].get('ttft_med')}/{out['tool'].get('ttft_p90')}s per-req {out['tool'].get('per_req_tok_s')}; "
      f"text ttft {out['text'].get('ttft_med')}/{out['text'].get('ttft_p90')}s per-req {out['text'].get('per_req_tok_s')}; errors {errs}", flush=True)
out["records"] = recs
json.dump(out, open(f"/home/milo/dsv41/replay2-{tag}-n{N}.json", "w"), indent=1)
