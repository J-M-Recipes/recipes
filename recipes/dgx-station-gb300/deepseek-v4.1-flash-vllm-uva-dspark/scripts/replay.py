#!/usr/bin/env python3
"""Replay real Hermes turns (transcript_fixture.json) against the lane with the terminal/read_file/web_search tool schemas.
Measures DSpark acceptance via /metrics deltas, split tool-call turns vs text turns. Env: BASE_URL MODEL TAG."""
import json, os, sys, time, urllib.request
base=os.getenv("BASE_URL","http://127.0.0.1:30006/v1"); model=os.getenv("MODEL","dsv41-flash-uva"); tag=os.getenv("TAG","x")
H={"Authorization":"Bearer x","Content-Type":"application/json"}
SYS=("You are Milo, James's operations agent. Be terse. Use tools when needed. Quote CDT times. "
     "Never rephrase a blocked command. Prefer tables for numbers.")
TOOLS=[{"type":"function","function":{"name":"terminal","description":"Run a shell command","parameters":{"type":"object","properties":{"command":{"type":"string"},"timeout":{"type":"integer"}},"required":["command"]}}},
       {"type":"function","function":{"name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"},"offset":{"type":"integer"},"limit":{"type":"integer"}},"required":["path"]}}},
       {"type":"function","function":{"name":"web_search","description":"Search the web","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}]
def metrics():
    t=urllib.request.urlopen(base.replace("/v1","")+"/metrics",timeout=10).read().decode(); d={}
    for l in t.splitlines():
        if l.startswith("vllm:spec_decode_num_draft_tokens_total"): d["draft"]=float(l.split()[-1])
        if l.startswith("vllm:spec_decode_num_accepted_tokens_total"): d["acc"]=float(l.split()[-1])
        if l.startswith("vllm:spec_decode_num_drafts_total"): d["steps"]=float(l.split()[-1])
    return d
def chat(msgs, max_tokens=400):
    p={"model":model,"messages":[{"role":"system","content":SYS}]+msgs,"max_tokens":max_tokens,"temperature":0,
       "tools":TOOLS,"tool_choice":"auto","chat_template_kwargs":{"thinking":False}}
    t0=time.monotonic(); r=json.load(urllib.request.urlopen(urllib.request.Request(base+"/chat/completions",data=json.dumps(p).encode(),headers=H),timeout=600))
    dt=time.monotonic()-t0; ch=r["choices"][0]
    return r["usage"]["completion_tokens"], dt, ch["finish_reason"], bool(ch["message"].get("tool_calls"))
S=json.load(open(sys.argv[1]))
res={"tool":{"tok":0,"s":0,"n":0,"draft":0,"acc":0,"steps":0,"emitted_tool":0},"text":{"tok":0,"s":0,"n":0,"draft":0,"acc":0,"steps":0,"emitted_tool":0}}
chat(S[0]["ctx"][-2:],8)
for s in S:
    k="tool" if s["has_tool"] else "text"
    m0=metrics(); tok,dt,fin,tc=chat(s["ctx"]); m1=metrics()
    r=res[k]; r["tok"]+=tok; r["s"]+=dt; r["n"]+=1; r["emitted_tool"]+=tc
    for f in ("draft","acc","steps"): r[f]+=m1[f]-m0[f]
    print(f"[{tag}] {k:4} tok={tok:4} {tok/dt:6.1f} tok/s acc={(m1['acc']-m0['acc'])/max(1,m1['draft']-m0['draft']):.2f} fin={fin} tool_call={tc}", flush=True)
for k,r in res.items():
    if r["n"]: print(f"[{tag}] SUMMARY {k:4}: n={r['n']} {r['tok']/r['s']:.1f} tok/s  accept={r['acc']/max(1,r['draft']):.1%}  acc/step={r['acc']/max(1,r['steps']):.2f}  tool_calls_emitted={r['emitted_tool']}/{r['n']}")
a=res["tool"]["acc"]+res["text"]["acc"]; d=res["tool"]["draft"]+res["text"]["draft"]; t=res["tool"]["tok"]+res["text"]["tok"]; ss=res["tool"]["s"]+res["text"]["s"]
print(f"[{tag}] OVERALL {t/ss:.1f} tok/s accept={a/max(1,d):.1%} tokens={t}")
