import json, os, sys, time, random, urllib.request
B=os.environ.get("BASE_URL","http://127.0.0.1:8891/v1")+"/chat/completions"; K=os.environ["GLM53_API_KEY"]
target=int(sys.argv[1]); depth=float(sys.argv[2]) if len(sys.argv)>2 else 0.5
random.seed(target)
words="the of and to in is was for on that with as by at from this be are or an it his which not have had but they were her all has one their there been would more can its who will".split()
sents=[]
while len(" ".join(sents))/4.3 < target:
    sents.append(" ".join(random.choice(words) for _ in range(random.randint(8,20))).capitalize()+".")
code="MAGENTA-%06d"%random.randint(0,999999)
sents.insert(int(len(sents)*depth), f"Note for later: the vault access code is {code}.")
hay=" ".join(sents)
body={"model":"glm-5.3-nvfp4-sglang","messages":[{"role":"user","content":hay+"\n\nWhat is the vault access code mentioned in the text above? Reply with the code only."}],"max_tokens":int(os.environ.get("NEEDLE_MAXTOK","400")),"temperature":0,"chat_template_kwargs":{"reasoning_effort":"low"}}
r=urllib.request.Request(B,data=json.dumps(body).encode(),headers={"Content-Type":"application/json","Authorization":"Bearer "+K})
t=time.time(); d=json.load(urllib.request.urlopen(r,timeout=1800)); w=time.time()-t
m=d["choices"][0]["message"]; c=(m.get("content") or "").strip(); rs=(m.get("reasoning_content") or m.get("reasoning") or ""); fin=d["choices"][0].get("finish_reason")
print(json.dumps(dict(target=target,depth=depth,prompt_tokens=d["usage"]["prompt_tokens"],expected=code,got=c[:80],PASS=code in c,finish=fin,completion_tokens=d["usage"]["completion_tokens"],reasoning_has_code=code in rs,reasoning_tail=rs[-200:],cached=(d["usage"].get("prompt_tokens_details") or {}).get("cached_tokens"),wall_s=round(w,1),prefill_tps_est=round(d["usage"]["prompt_tokens"]/w))))
