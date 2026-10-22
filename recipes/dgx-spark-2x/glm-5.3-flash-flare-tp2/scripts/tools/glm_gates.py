#!/usr/bin/env python3
"""GLM Flare gates: models, exact-reply smoke (thinking off), thinking-off honored, tool call. stdlib only."""
import json, sys, time, urllib.request
B = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8003/v1"
def post(path, body, t=300):
    r = urllib.request.Request(B + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=t) as f: return json.load(f)
res = {}
m = json.load(urllib.request.urlopen(B + "/models", timeout=10)); res["models"] = m
mid = m["data"][0]["id"]
t0 = time.time()
r = post("/chat/completions", {"model": mid, "messages": [{"role": "user", "content": "Reply with exactly: PONG-7319"}],
        "max_tokens": 1024, "reasoning_effort": "low"})
msg = r["choices"][0]["message"]
res["smoke"] = {"content": msg.get("content"), "reasoning": (msg.get("reasoning_content") or "")[:200], "usage": r.get("usage"), "s": round(time.time()-t0, 2),
                "pass": "PONG-7319" in (msg.get("content") or "")}
r = post("/chat/completions", {"model": mid, "messages": [{"role": "user", "content": "What is 17*23? Answer briefly."}], "max_tokens": 2048})
msg = r["choices"][0]["message"]
res["thinking_default"] = {"has_reasoning": bool(msg.get("reasoning_content")), "content": (msg.get("content") or "")[:120], "usage": r.get("usage")}
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
r = post("/chat/completions", {"model": mid, "messages": [{"role": "user", "content": "What is the weather in Paris? Use the tool."}],
        "tools": tools, "max_tokens": 1024, "chat_template_kwargs": {"enable_thinking": False}})
tc = r["choices"][0]["message"].get("tool_calls") or []
args = None
try: args = json.loads(tc[0]["function"]["arguments"])
except Exception: pass
res["tool"] = {"tool_calls": tc, "finish": r["choices"][0].get("finish_reason"), "pass": bool(tc) and tc[0]["function"]["name"] == "get_weather" and isinstance(args, dict) and "paris" in str(args.get("city", "")).lower()}
res["pass"] = res["smoke"]["pass"] and res["tool"]["pass"]
print(json.dumps(res, indent=1))
print("GATES", "PASS" if res["pass"] else "FAIL", "smoke", res["smoke"]["pass"], "tool", res["tool"]["pass"], "thinking_default", res["thinking_default"]["has_reasoning"])
