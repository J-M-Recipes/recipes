#!/usr/bin/env python3
"""T0 gate: /v1/models id, exact-word chat smoke, parsed tool_calls smoke. stdlib only."""
import json, time, urllib.request
U = "http://<gb300>:30006/v1"; M = "dsv41-flash-uva"
def req(path, body=None, t=300):
    r = urllib.request.Request(U + path, data=json.dumps(body).encode() if body else None,
                               headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(r, timeout=t) as f:
        return json.load(f), round(time.time() - t0, 2)
out = {}
m, _ = req("/models"); ids = [d["id"] for d in m.get("data", [])]
out["models"] = {"ids": ids, "pass": M in ids}
c, dt = req("/chat/completions", {"model": M, "temperature": 0, "max_tokens": 512,
          "messages": [{"role": "user", "content": "Reply with exactly the single word: PINEAPPLE"}]})
txt = (c["choices"][0]["message"].get("content") or "").strip()
out["chat"] = {"content": txt, "pass": txt.strip(".!\"' \n") == "PINEAPPLE", "wall_s": dt, "usage": c.get("usage")}
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
                         "required": ["city"]}}}]
c, dt = req("/chat/completions", {"model": M, "temperature": 0, "max_tokens": 1024, "tools": tools, "tool_choice": "auto",
          "messages": [{"role": "user", "content": "What's the weather in Austin in fahrenheit? Use the tool."}]})
msg = c["choices"][0]["message"]; tc = msg.get("tool_calls") or []
ok = False; parsed = None
if tc:
    try:
        parsed = {"name": tc[0]["function"]["name"], "args": json.loads(tc[0]["function"]["arguments"])}
        ok = parsed["name"] == "get_weather" and "austin" in str(parsed["args"].get("city", "")).lower()
    except Exception as e:
        parsed = {"error": str(e), "raw": tc}
out["tool_call"] = {"n_calls": len(tc), "parsed": parsed, "finish_reason": c["choices"][0].get("finish_reason"), "pass": ok, "wall_s": dt}
out["T0_pass"] = all(out[k]["pass"] for k in ("models", "chat", "tool_call"))
print(json.dumps(out, indent=1))
