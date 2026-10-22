#!/usr/bin/env python3
"""Matched OpenAI-compatible operational probe for GLM quant faceoff."""
import argparse, concurrent.futures, json, statistics, time, urllib.request, uuid


def get_json(url):
    with urllib.request.urlopen(url, timeout=20) as r:
        return json.load(r)


def stream_chat(base, model, messages, max_tokens, temperature=0.0, seed=None):
    body = {
        "model": model, "messages": messages, "max_tokens": max_tokens,
        "temperature": temperature, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    t0 = time.perf_counter(); first = None; usage = {}; text = []
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            if not raw.startswith(b"data: "): continue
            payload = raw[6:].strip()
            if payload == b"[DONE]": break
            d = json.loads(payload)
            if d.get("usage"): usage = d["usage"]
            for ch in d.get("choices", []):
                c = ch.get("delta", {}).get("content")
                if c:
                    if first is None: first = time.perf_counter()
                    text.append(c)
    t1 = time.perf_counter()
    ttft = (first or t1) - t0
    return {"ttft_s":ttft,"total_s":t1-t0,"usage":usage,"text":"".join(text)}


def cold_prompt(target):
    nonce = uuid.uuid4().hex
    # Unique first cache block defeats APC reuse; random-ish numbered records keep tokenizer stable.
    parts = [f"COLD-NONCE {nonce}\n"]
    i = 0
    line = "Record {i:07d}: copper juniper atlas river quartz meadow lantern.\n"
    # Approx 4 chars/token; overbuild then truncate by chars. Actual API usage is authoritative.
    chars = target * 4
    while sum(map(len, parts)) < chars:
        parts.append(line.format(i=i)); i += 1
    parts.append("\nReply with exactly: OK")
    return "".join(parts)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--base", default="http://127.0.0.1:8888/v1"); ap.add_argument("--label", required=True); ap.add_argument("--out", required=True); a=ap.parse_args()
    m = get_json(a.base+"/models")["data"][0]
    model=m["id"]
    out={"label":a.label,"timestamp":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"model_contract":m,"method":{"prefill":"cold nonce in first cache block, streaming max_tokens=1, effective prompt_tokens/TTFT","decode":"streamed completion_tokens/(total-TTFT)","concurrency":"3 simultaneous cold 18k prompts, max_tokens=1"}}

    pre=[]
    for target in (8000,32000,100000):
        for run in range(2):
            r=stream_chat(a.base,model,[{"role":"user","content":cold_prompt(target)}],1,seed=7319+run)
            pt=r["usage"].get("prompt_tokens",0); rate=pt/r["ttft_s"] if r["ttft_s"] else 0
            row={"target":target,"run":run+1,"prompt_tokens":pt,"ttft_s":round(r["ttft_s"],4),"effective_prefill_tok_s":round(rate,1),"answer":r["text"][:40]}; pre.append(row); print("PREFILL",json.dumps(row),flush=True)
    out["prefill"]=pre

    fixtures={
      "structured_count":"Print integers 1 through 300 separated only by commas. Do not omit any integer.",
      "prose":"Write 450 words explaining why measurement methodology matters in systems engineering. Plain prose, no headings or lists."
    }
    dec=[]
    for name,prompt in fixtures.items():
        for run in range(2):
            r=stream_chat(a.base,model,[{"role":"user","content":prompt}],650,seed=9000+run)
            ct=r["usage"].get("completion_tokens",0); span=max(0.0001,r["total_s"]-r["ttft_s"]); rate=ct/span
            row={"fixture":name,"run":run+1,"prompt_tokens":r["usage"].get("prompt_tokens",0),"completion_tokens":ct,"ttft_s":round(r["ttft_s"],4),"decode_s":round(span,4),"decode_tok_s":round(rate,1)}; dec.append(row); print("DECODE",json.dumps(row),flush=True)
    out["decode"]=dec

    def one(i):
        r=stream_chat(a.base,model,[{"role":"user","content":cold_prompt(18000)+f"\nSession {i}"}],1,seed=10000+i)
        pt=r["usage"].get("prompt_tokens",0)
        return {"session":i,"prompt_tokens":pt,"ttft_s":round(r["ttft_s"],4),"effective_prefill_tok_s":round(pt/r["ttft_s"],1)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex: conc=list(ex.map(one,range(3)))
    out["concurrent_prefill_3x18k"]=conc
    print("CONCURRENT",json.dumps(conc),flush=True)

    # Compact medians for later HTML generation.
    out["summary"]={}
    for target in (8000,32000,100000):
        vals=[x["effective_prefill_tok_s"] for x in pre if x["target"]==target]
        out["summary"][f"prefill_{target}"]=round(statistics.median(vals),1)
    for name in fixtures:
        vals=[x["decode_tok_s"] for x in dec if x["fixture"]==name]
        out["summary"][f"decode_{name}"]=round(statistics.median(vals),1)
    out["summary"]["concurrent_aggregate_effective_tok_s"]=round(sum(x["prompt_tokens"] for x in conc)/max(x["ttft_s"] for x in conc),1)
    with open(a.out,"w") as f: json.dump(out,f,indent=2)
    print("RESULT",a.out,json.dumps(out["summary"]),flush=True)

if __name__=="__main__": main()
