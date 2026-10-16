import json, os, time, urllib.request
B=os.environ.get("BASE_URL","http://127.0.0.1:8890/v1") + "/chat/completions"; K=os.environ["GLM53_API_KEY"]
P=["Write a Python function that parses an nginx access log line into a dict (ip, time, method, path, status, bytes, referrer, user agent). Include a regex, type hints, and three pytest tests.",
   "Explain, for a smart engineer new to inference, why tensor-parallel decode across 8 nodes over 200G Ethernet is latency-bound rather than bandwidth-bound. Use concrete numbers.",
   "Draft a careful email to a contractor explaining that the 20A circuit they installed trips under sustained 1.5 kW load, asking them to inspect the breaker and wiring, and proposing two times to visit."]
def run(p, effort):
    body={"model":"glm-5.3-fp8","messages":[{"role":"user","content":p}],"max_tokens":1500,"temperature":0.6,"top_p":0.95,"stream":True,"stream_options":{"include_usage":True},"chat_template_kwargs":{"reasoning_effort":effort}}
    r=urllib.request.Request(B,data=json.dumps(body).encode(),headers={"Content-Type":"application/json","Authorization":"Bearer "+K})
    t0=time.time(); tf=None; u=None; rl=cl=0; fin=None
    with urllib.request.urlopen(r,timeout=900) as f:
        for line in f:
            line=line.decode().strip()
            if not line.startswith("data: ") or line=="data: [DONE]": continue
            d=json.loads(line[6:])
            if d.get("usage"): u=d["usage"]
            for c in d.get("choices",[]):
                dl=c.get("delta",{})
                if (dl.get("content") or dl.get("reasoning") or dl.get("reasoning_content")) and tf is None: tf=time.time()
                rl+=len(dl.get("reasoning") or dl.get("reasoning_content") or ""); cl+=len(dl.get("content") or "")
                fin=c.get("finish_reason") or fin
    t1=time.time(); n=u["completion_tokens"]
    return dict(effort=effort,prompt=u["prompt_tokens"],completion=n,ttft=round(tf-t0,2),decode_tps=round((n-1)/(t1-tf),1),reasoning_chars=rl,content_chars=cl,finish=fin)
for e in ["low"]:
    for p in P: print(json.dumps(run(p,e)),flush=True)
print(json.dumps(run(P[1],"high")),flush=True)
