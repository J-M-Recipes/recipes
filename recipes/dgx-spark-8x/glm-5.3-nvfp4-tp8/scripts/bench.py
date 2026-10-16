import os
# bench.py <tag> — C1: 1 warm + 3 reps x 3 realistic prompts (temp 0, effort low, max_tokens 800, streamed);
# C4: 4 concurrent (one of each prompt + 1). Writes bench-<tag>.jsonl. Decode tps = (completion-1)/(t_last - t_first).
import json, os, sys, time, statistics, urllib.request, concurrent.futures as cf
B=os.environ.get("BASE_URL","http://127.0.0.1:8890/v1") + "/chat/completions"; K=os.environ["GLM53_API_KEY"]; TAG=sys.argv[1]
P={"code":"Write a Python function that parses an nginx access log line into a dict (ip, time, method, path, status, bytes, referrer, user agent). Include a regex, type hints, and three pytest tests.",
   "explain":"Explain, for a smart engineer new to inference, why tensor-parallel decode across 8 nodes over 200G Ethernet is latency-bound rather than bandwidth-bound. Use concrete numbers.",
   "prose":"Draft a careful email to a contractor explaining that the 20A circuit they installed trips under sustained 1.5 kW load, asking them to inspect the breaker and wiring, and proposing two times to visit."}
def run(name):
    body={"model":"glm-5.3-nvfp4","messages":[{"role":"user","content":P[name]}],"max_tokens":800,"temperature":0,"stream":True,"stream_options":{"include_usage":True},"chat_template_kwargs":{"reasoning_effort":"low"}}
    r=urllib.request.Request(B,data=json.dumps(body).encode(),headers={"Content-Type":"application/json","Authorization":"Bearer "+K})
    t0=time.time(); tf=None; tl=None; u=None
    with urllib.request.urlopen(r,timeout=900) as f:
        for line in f:
            line=line.decode().strip()
            if not line.startswith("data: ") or line=="data: [DONE]": continue
            d=json.loads(line[6:])
            if d.get("usage"): u=d["usage"]
            for c in d.get("choices",[]):
                dl=c.get("delta",{})
                if dl.get("content") or dl.get("reasoning") or dl.get("reasoning_content"):
                    tl=time.time(); tf=tf or tl
    n=u["completion_tokens"]
    return dict(tag=TAG,prompt=name,completion=n,ttft=round(tf-t0,3),tps=round((n-1)/(tl-tf),2))
out=open(f"bench-{TAG}.jsonl","a")
run("code")  # warm
res={k:[] for k in P}
for rep in range(3):
    for k in P:
        x=run(k); x["rep"]=rep; res[k].append(x["tps"]); out.write(json.dumps(x)+"\n"); out.flush()
summ={k:dict(median=statistics.median(v),min=min(v),max=max(v),completion=None) for k,v in res.items()}
t=time.time()
with cf.ThreadPoolExecutor(4) as ex: cc=list(ex.map(run,["code","explain","prose","code"]))
w=time.time()-t; agg=round(sum(x["completion"] for x in cc)/w,1)
for x in cc: x["c4"]=True; out.write(json.dumps(x)+"\n")
print(json.dumps(dict(tag=TAG,c1=summ,c4_agg_tps=agg,c4_per_user=[x["tps"] for x in cc])))
out.write(json.dumps(dict(summary=True,tag=TAG,c1=summ,c4_agg_tps=agg))+"\n")
