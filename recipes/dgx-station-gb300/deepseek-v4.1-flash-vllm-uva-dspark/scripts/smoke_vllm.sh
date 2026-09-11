#!/usr/bin/env bash
set -euo pipefail
B=http://127.0.0.1:30006
curl -fsS $B/v1/models | python3 -c 'import sys,json; d=json.load(sys.stdin); print("models:",[m["id"] for m in d["data"]])'
python3 - <<'PY'
import json,time,urllib.request
B="http://127.0.0.1:30006/v1/chat/completions"
def chat(msgs,mx=64,think=False,**kw):
    body={"model":"dsv41-flash-uva","messages":msgs,"max_tokens":mx,"temperature":0,
          "chat_template_kwargs":{"thinking":think}}; body.update(kw)
    t=time.time(); r=json.load(urllib.request.urlopen(urllib.request.Request(B,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"}),timeout=600)); dt=time.time()-t
    m=r["choices"][0]["message"]; u=r.get("usage",{})
    return m,u,dt,r["choices"][0].get("finish_reason")
m,u,dt,f=chat([{"role":"user","content":"What is 17*19? Return only the integer."}])
print("ARITH:",repr(m.get("content")),u,f"{dt:.2f}s",f)
m,u,dt,f=chat([{"role":"user","content":"Count from 1 to 60 separated by spaces. Only the numbers."}],mx=300)
c=m.get("content") or ""; nums=[int(x) for x in c.split() if x.isdigit()]
print("COUNT ok:",nums[:60]==list(range(1,61)),"tokens",u.get("completion_tokens"),f"{dt:.2f}s -> {u.get('completion_tokens',0)/dt:.1f} tok/s")
m,u,dt,f=chat([{"role":"user","content":"Write a 300-word paragraph on why local inference hardware matters for a small studio. No lists."}],mx=450)
print("PROSE tokens",u.get("completion_tokens"),f"{dt:.2f}s -> {u.get('completion_tokens',0)/dt:.1f} tok/s; tail:",repr((m.get("content") or "")[-120:]))
tools=[{"type":"function","function":{"name":"get_weather","description":"Get current weather","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}]
m,u,dt,f=chat([{"role":"user","content":"What's the weather in Telluride right now? Use the tool."}],mx=200,tools=tools)
print("TOOL finish:",f,"tool_calls:",json.dumps(m.get("tool_calls"))[:300],"content:",repr((m.get("content") or "")[:80]))
m,u,dt,f=chat([{"role":"user","content":"What is 15% of 240?"}],mx=400,think=True,reasoning_effort="low")
print("THINK reasoning_len:",len(m.get("reasoning_content") or m.get("reasoning") or ""),"content:",repr((m.get("content") or "")[:100]),u)
PY
