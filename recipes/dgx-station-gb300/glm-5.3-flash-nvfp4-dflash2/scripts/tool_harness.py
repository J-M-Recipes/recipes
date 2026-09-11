#!/usr/bin/env python3
"""10-prompt tool-call gate against an OpenAI-compatible SGLang endpoint. No Hermes wiring."""
import json, os, urllib.request, time
BASE=os.getenv("BASE_URL","http://127.0.0.1:30001/v1"); MODEL=os.getenv("MODEL","glm-5.3-flash")
H={"Content-Type":"application/json"}
TOOLS=[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}},
        {"type":"function","function":{"name":"run_shell","parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}}},
        {"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]
PROMPTS=[
    ("weather-telluride","What's the weather in Telluride right now? Use the tool.", "get_weather"),
    ("weather-nyc","Look up the current weather for New York City.", "get_weather"),
    ("ls-tmp","List files in /tmp using a shell command.", "run_shell"),
    ("disk","Check disk free space with df -h.", "run_shell"),
    ("read-hosts","Read /etc/hosts for me.", "read_file"),
    ("read-passwd","Open /etc/os-release and show it.", "read_file"),
    ("weather-gj","What's it like outside in Grand Junction, CO?", "get_weather"),
    ("uname","Run uname -a on the box.", "run_shell"),
    ("read-release","Read the file /etc/lsb-release.", "read_file"),
    ("weather-pen","Current weather in Pensacola, Florida please.", "get_weather"),
]
def chat(prompt):
    p={"model":MODEL,"messages":[{"role":"user","content":prompt}],"max_tokens":256,"temperature":0,
       "tools":TOOLS,"chat_template_kwargs":{"enable_thinking":False}}
    req=urllib.request.Request(BASE+"/chat/completions",data=json.dumps(p).encode(),headers=H)
    return json.load(urllib.request.urlopen(req,timeout=120))
ok=0
for name,prompt,want in PROMPTS:
    t0=time.time();
    try:
        r=chat(prompt); ch=r["choices"][0]; calls=ch["message"].get("tool_calls") or []
        names=[c["function"]["name"] for c in calls]; hit=want in names
        print(f"{'OK' if hit else 'FAIL'} {name:16s} finish={ch.get('finish_reason')} calls={names} {time.time()-t0:.1f}s")
        ok += int(hit)
    except Exception as e:
        print(f"ERR  {name:16s} {e}")
print(f"SUMMARY tool_ok={ok}/{len(PROMPTS)}")
raise SystemExit(0 if ok==len(PROMPTS) else 1)
