#!/usr/bin/env python3
"""ed_replay.py — replay the Hermes transcript fixture against the GLM-Flash lane (Card ED, 2026-11-01).
Adapted from dsv41 replay_turns.py: same SYS/TOOLS/contexts, sequential C1, GLM model name, reasoning_effort low.
usage: TAG=<tag> PASSES=1 OUT=<path.json> python3 ed_replay.py
"""
import json, os, time, urllib.request
base = os.getenv("BASE_URL", "http://127.0.0.1:30001/v1"); model = "glm-5.3-flash"
tag = os.environ.get("TAG", "ed"); PASSES = int(os.getenv("PASSES", "1")); OUT = os.environ["OUT"]
H = {"Content-Type": "application/json"}
SYS = ("You are Milo, James's operations agent. Be terse. Use tools when needed. Quote CDT times. "
       "Never rephrase a blocked command. Prefer tables for numbers.")
TOOLS = [{"type": "function", "function": {"name": "terminal", "description": "Run a shell command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}}, "required": ["command"]}}},
         {"type": "function", "function": {"name": "read_file", "description": "Read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, "required": ["path"]}}},
         {"type": "function", "function": {"name": "web_search", "description": "Search the web", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}]
S = json.load(open("~/dsv41/transcript_fixture.json"))
out = {"tag": tag, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_turns": len(S), "passes": []}
for p in range(PASSES):
    rows = []
    for i, s in enumerate(S):
        body = {"model": model, "messages": [{"role": "system", "content": SYS}] + s["ctx"], "max_tokens": 600, "temperature": 0,
                "tools": TOOLS, "tool_choice": "auto", "chat_template_kwargs": {"reasoning_effort": "low"}}
        t0 = time.monotonic()
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(), H), timeout=900))
            m = r["choices"][0]["message"]; tc = m.get("tool_calls") or []
            rows.append({"i": i, "has_tool": s.get("has_tool"), "emitted": bool(tc), "ptoks": r["usage"]["prompt_tokens"],
                         "ctoks": r["usage"]["completion_tokens"], "s": round(time.monotonic() - t0, 2)})
        except Exception as e:  # noqa: BLE001
            rows.append({"i": i, "error": str(e)[:200]})
    out["passes"].append(rows)
    ok = [r for r in rows if "error" not in r]
    print(f"[{tag}] pass {p}: {len(ok)}/{len(rows)} ok, ctoks {sum(r['ctoks'] for r in ok)}, ptoks {sum(r['ptoks'] for r in ok)}", flush=True)
json.dump(out, open(OUT, "w"), indent=1)
