#!/usr/bin/env python3
"""Flash bench: smoke + knee (2 reps per concurrency) against an SGLang OpenAI endpoint. Streams; counts content tokens
via usage when available else chunks. Prints one line per result. TAG env labels rows."""
import json, os, sys, time, threading, urllib.request, uuid
BASE=os.getenv("BASE_URL","http://127.0.0.1:30001/v1"); MODEL=os.getenv("MODEL","glm-5.3-flash"); TAG=os.getenv("TAG","x")
H={"Authorization":"Bearer x","Content-Type":"application/json"}
PROSE="Write a 350-word essay on why engineers should keep failure ledgers. No headings, no lists."
def chat(msgs,max_tokens=256,stream=False,tools=None,thinking=None,temperature=0.0):
    p={"model":MODEL,"messages":msgs,"max_tokens":max_tokens,"temperature":temperature,"stream":stream}
    if tools: p["tools"]=tools
    if thinking is not None: p["chat_template_kwargs"]={"enable_thinking":thinking}
    req=urllib.request.Request(BASE+"/chat/completions",data=json.dumps(p).encode(),headers=H)
    return urllib.request.urlopen(req,timeout=900)
def gen_tps(prompt,max_tokens=256,nonce=True):
    msgs=[{"role":"user","content":prompt+(f"\n[{uuid.uuid4().hex[:8]}]" if nonce else "")}]
    t0=time.monotonic(); r=json.load(chat(msgs,max_tokens,thinking=False)); dt=time.monotonic()-t0
    n=r["usage"]["completion_tokens"]; return n,dt,r["choices"][0]["message"]["content"] or ""
def smoke():
    n,dt,c=gen_tps("What is 17*19? Reply with just the number.",8); print(f"[{TAG}] ARITH {c.strip()!r} {dt:.2f}s")
    n,dt,c=gen_tps("Count from 1 to 60, comma separated, nothing else.",200); ok=all(str(i) in c for i in (1,30,60)); print(f"[{TAG}] COUNT ok={ok} {n} tok {dt:.2f}s -> {n/dt:.1f} tok/s")
    n,dt,c=gen_tps(PROSE,300); print(f"[{TAG}] PROSE {n} tok {dt:.2f}s -> {n/dt:.1f} tok/s; tail: {c.strip()[-80:]!r}")
    tools=[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}]
    r=json.load(chat([{"role":"user","content":"What's the weather in Telluride right now? Use the tool."}],64,tools=tools,thinking=False))
    ch=r["choices"][0]; print(f"[{TAG}] TOOL finish={ch['finish_reason']} calls={json.dumps([t['function'] for t in (ch['message'].get('tool_calls') or [])])[:120]}")
    r=json.load(chat([{"role":"user","content":"What is 15% of 240? Think briefly, then answer with just the number."}],200,thinking=True))
    m=r["choices"][0]["message"]; print(f"[{TAG}] THINK reasoning_len={len(m.get('reasoning_content') or '')} content={ (m.get('content') or '').strip()!r} usage={r['usage'].get('completion_tokens')}")
def knee(concs=(1,2,4,8,16,32),reps=3,max_tokens=256):
    """rep 0 at each shape is a warm/JIT pass and is DISCARDED; report mean of the remaining reps.
    reps=0 → warm-only (no ZeroDivision). Never publish the warm-pass."""
    for c in concs:
        aggs=[]
        warm=0.0
        for rep in range(max(1, reps+1)):
            res=[None]*c
            def w(i):
                try: res[i]=gen_tps(PROSE,max_tokens)
                except Exception as e: res[i]=(0,1,str(e))
            th=[threading.Thread(target=w,args=(i,)) for i in range(c)]
            t0=time.monotonic(); [t.start() for t in th]; [t.join() for t in th]; wall=time.monotonic()-t0
            toks=sum(r[0] for r in res)
            if reps==0 or rep==0:
                warm=toks/wall
                if reps==0:
                    print(f"[{TAG}] warm conc={c:2d}: {warm:.0f} agg",flush=True)
                continue
            aggs.append(toks/wall)
        if not aggs: continue
        a=sum(aggs)/len(aggs); spread=(max(aggs)-min(aggs))/a*100
        print(f"[{TAG}] conc={c:2d}: {a:7.1f} agg tok/s  {a/c:6.2f} per-stream  (reps {'/'.join(f'{x:.0f}' for x in aggs)} spread {spread:.0f}%; warm-pass {warm:.0f})",flush=True)
if __name__=="__main__":
    what=sys.argv[1] if len(sys.argv)>1 else "all"
    if what in("smoke","all"): smoke()
    if what in("knee","all"): knee()
