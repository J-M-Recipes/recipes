import json, os, sys, time, urllib.request, urllib.error
B = os.environ.get("BASE_URL","http://127.0.0.1:8891/v1"); K = os.environ["GLM53_API_KEY"]; M = "glm-5.3-nvfp4-sglang"
def req(path, body=None, key=True, timeout=900):
    h = {"Content-Type": "application/json"}
    if key: h["Authorization"] = "Bearer " + K
    r = urllib.request.Request(B + path, data=json.dumps(body).encode() if body else None, headers=h)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as f: return f.status, json.load(f)
    except urllib.error.HTTPError as e: return e.code, None
res = {}
s, _ = req("/models", key=False); res["keyless_models"] = s
s, d = req("/models"); res["models"] = (s, [(m["id"], m.get("max_model_len")) for m in d["data"]])
t = time.time()
s, d = req("/chat/completions", {"model": M, "messages": [{"role": "user", "content": "Reply exactly ROUTE_OK."}], "max_tokens": 512, "temperature": 0})
m = d["choices"][0]["message"]
res["smoke"] = {"status": s, "content": m.get("content"), "reasoning_len": len(m.get("reasoning") or m.get("reasoning_content") or ""), "finish": d["choices"][0]["finish_reason"], "usage": d["usage"], "wall_s": round(time.time() - t, 2)}
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
s, d = req("/chat/completions", {"model": M, "messages": [{"role": "user", "content": "What's the weather in Telluride? Use the tool."}], "tools": tools, "max_tokens": 1024, "temperature": 0})
m = d["choices"][0]["message"]
res["tool"] = {"status": s, "tool_calls": [(c["function"]["name"], c["function"]["arguments"]) for c in (m.get("tool_calls") or [])], "finish": d["choices"][0]["finish_reason"]}
if m.get("tool_calls"):
    c = m["tool_calls"][0]
    msgs = [{"role": "user", "content": "What's the weather in Telluride? Use the tool."}, {"role": "assistant", "content": m.get("content") or "", "tool_calls": m["tool_calls"]},
            {"role": "tool", "tool_call_id": c["id"], "content": json.dumps({"city": "Telluride", "temp_f": 41, "conditions": "light snow"})}]
    s, d = req("/chat/completions", {"model": M, "messages": msgs, "tools": tools, "max_tokens": 1024, "temperature": 0})
    res["tool_round2"] = {"status": s, "content": d["choices"][0]["message"].get("content"), "finish": d["choices"][0]["finish_reason"]}
print(json.dumps(res, indent=1))
