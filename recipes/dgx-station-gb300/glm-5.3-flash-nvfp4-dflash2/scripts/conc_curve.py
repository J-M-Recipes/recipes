#!/usr/bin/env python3
"""High-concurrency curve for 'prose/knowledge work'. Per shape: 1 discarded warm pass + REPS measured.
Reports aggregate tok/s, per-user tok/s (median across users), TTFT p50/p95, wall. Streams so TTFT is real.
Prompt ~1.5K tokens (knowledge-work style), OSL fixed via max_tokens (ignore_eos-like: we ask for long output).
Env: BASE_URL MODEL TAG CONCS (csv) REPS OSL"""
import json, os, sys, time, threading, urllib.request, uuid, statistics
BASE=os.getenv("BASE_URL","http://127.0.0.1:30001/v1"); MODEL=os.getenv("MODEL","glm-5.3-flash"); TAG=os.getenv("TAG","x")
CONCS=[int(x) for x in os.getenv("CONCS","1,4,8,16,32,48,64,96,128").split(",")]; REPS=int(os.getenv("REPS","3")); OSL=int(os.getenv("OSL","384"))
H={"Content-Type":"application/json"}
BG=("Background briefing (read fully before answering). "+" ".join(
 f"Section {i}: In fiscal review {i}, the regional office reported that operational throughput rose modestly while unit costs held flat; the analysts attributed the change to process consolidation, a revised vendor contract, and staff cross-training, and they flagged data-quality risks in the reporting pipeline as the main caveat."
 for i in range(1,40)))
Q="\n\nTask: Write a 400-word executive memo synthesizing the briefing above: key trends, the two most important risks, and three concrete recommendations. Prose only, no bullet lists."
def one(i,res):
    p={"model":MODEL,"messages":[{"role":"user","content":BG+Q+f"\n[{uuid.uuid4().hex[:6]}]"}],"max_tokens":OSL,"temperature":0.7,"stream":True,
       "stream_options":{"include_usage":True},"chat_template_kwargs":{"enable_thinking":False}}
    t0=time.monotonic(); first=None; n=0; usage=None
    try:
        r=urllib.request.urlopen(urllib.request.Request(BASE+"/chat/completions",data=json.dumps(p).encode(),headers=H),timeout=900)
        for line in r:
            if not line.startswith(b"data: ") or b"[DONE]" in line: continue
            d=json.loads(line[6:])
            if d.get("usage"): usage=d["usage"]
            ch=d.get("choices") or []
            if ch and (ch[0].get("delta") or {}).get("content"):
                if first is None: first=time.monotonic()-t0
                n+=1
        end=time.monotonic()-t0
        toks=(usage or {}).get("completion_tokens") or n
        res[i]=(toks,first if first is not None else end,end,(usage or {}).get("prompt_tokens"))
    except Exception as e: res[i]=(0,0,time.monotonic()-t0,None); sys.stderr.write(f"err {e}\n")
def shape(c):
    res=[None]*c; th=[threading.Thread(target=one,args=(i,res)) for i in range(c)]
    t0=time.monotonic(); [t.start() for t in th]; [t.join() for t in th]; wall=time.monotonic()-t0
    toks=sum(r[0] for r in res); ttft=sorted(r[1] for r in res); per=[r[0]/max(r[2]-r[1],1e-6) for r in res if r[0]]
    return dict(agg=toks/wall, per_user=statistics.median(per) if per else 0, ttft50=ttft[len(ttft)//2], ttft95=ttft[int(0.95*(len(ttft)-1))], wall=wall, isl=res[0][3], fails=sum(1 for r in res if r[0]==0))
for c in CONCS:
    w=shape(c)  # discarded warm
    ms=[shape(c) for _ in range(REPS)]
    agg=[m["agg"] for m in ms]; spread=(max(agg)-min(agg))/statistics.mean(agg)*100
    m=ms[len(ms)//2] if REPS%2 else ms[0]
    print(f"[{TAG}] C{c:3d}: agg {statistics.mean(agg):7.1f} tok/s (spread {spread:2.0f}%) | per-user {statistics.median([m['per_user'] for m in ms]):6.1f} tok/s | TTFT p50 {statistics.median([m['ttft50'] for m in ms]):5.2f}s p95 {max(m['ttft95'] for m in ms):5.2f}s | wall {statistics.mean([m['wall'] for m in ms]):5.1f}s | isl {m['isl']} fails {sum(m['fails'] for m in ms)}",flush=True)
