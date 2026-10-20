#!/usr/bin/env python3
"""Lane C functional gates: /v1/models, exact-reply smoke, tool-call round trip, metrics snapshot."""
import json, os, sys, time, urllib.request

BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8002/v1").rstrip("/").removesuffix("/v1")
MODEL = os.environ.get("MODEL", "deepseek-v4.1-flash-spark-vllm")
OUT = sys.argv[1] if len(sys.argv) > 1 else "."

def req(path, body=None, timeout=600):
    r = urllib.request.Request(BASE + path, data=json.dumps(body).encode() if body is not None else None,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        raw = f.read()
    return json.loads(raw) if path != "/metrics" else raw.decode()

res = {}
m = req("/v1/models")
d = m["data"][0]
res["models"] = {"ids": [x["id"] for x in m["data"]], "max_model_len": d.get("max_model_len"),
                 "pass": d["id"] == MODEL and d.get("max_model_len") == 1048576}

t0 = time.time()
r = req("/v1/chat/completions", {"model": MODEL, "temperature": 0, "max_tokens": 16,
        "messages": [{"role": "user", "content": "Reply with exactly LANE_C_OK and nothing else."}]})
msg = r["choices"][0]["message"]
res["smoke"] = {"content": msg.get("content"), "reasoning": (msg.get("reasoning_content") or msg.get("reasoning") or "")[:80],
                "usage": r["usage"], "seconds": round(time.time() - t0, 2),
                "pass": (msg.get("content") or "").strip() == "LANE_C_OK"}

tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
msgs = [{"role": "user", "content": "What's the weather in Paris right now? Use the tool."}]
r1 = req("/v1/chat/completions", {"model": MODEL, "temperature": 0, "max_tokens": 200, "tools": tools, "messages": msgs})
m1 = r1["choices"][0]["message"]
tc = m1.get("tool_calls") or []
ok1 = bool(tc) and tc[0]["function"]["name"] == "get_weather" and "paris" in tc[0]["function"]["arguments"].lower()
final = None
if ok1:
    msgs += [{"role": "assistant", "content": m1.get("content") or "", "tool_calls": tc},
             {"role": "tool", "tool_call_id": tc[0]["id"], "content": json.dumps({"city": "Paris", "temp_c": 17, "sky": "light rain"})}]
    r2 = req("/v1/chat/completions", {"model": MODEL, "temperature": 0, "max_tokens": 120, "tools": tools, "messages": msgs})
    final = r2["choices"][0]["message"].get("content") or ""
res["tool"] = {"tool_calls": tc, "finish_reason": r1["choices"][0]["finish_reason"], "final": final,
               "pass": ok1 and final is not None and "17" in final}

met = req("/metrics")
open(f"{OUT}/metrics-after-gates.txt", "w").write(met)
pre = [l for l in met.splitlines() if l.startswith("vllm:num_preemptions_total")]
res["preemptions"] = pre
json.dump(res, open(f"{OUT}/gates-functional.json", "w"), indent=1)
print(json.dumps(res, indent=1))
print("FUNCTIONAL_GATES", "PASS" if all(res[k]["pass"] for k in ("models", "smoke", "tool")) else "FAIL")
