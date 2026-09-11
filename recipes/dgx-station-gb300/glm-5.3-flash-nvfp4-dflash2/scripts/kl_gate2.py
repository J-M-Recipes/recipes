#!/usr/bin/env python3
"""Teacher-forced logprob divergence gate, two-phase so servers can run one at a time on a single GPU.

  phase gen   : BASE=<ref server> python3 kl_gate2.py gen  ref.json        (greedy completions from REF, then score them on REF)
  phase score : BASE=<cand server> python3 kl_gate2.py score ref.json cand.json   (score the SAME texts on CAND)
  phase cmp   : python3 kl_gate2.py cmp ref.json cand.json                  (report)

Scoring uses /v1/completions echo=True logprobs=1 max_tokens=1 (SGLang rejects 0; last token dropped) on prompt+generation; only generated-region tokens count.
Reports mean|Δlogp|, p95, and top-1 agreement (fraction of generated tokens that are also CAND's argmax)."""
import json, os, sys, time, urllib.request, statistics
MODEL=os.getenv("MODEL","glm-5.3-flash"); G=int(os.getenv("GEN_TOKENS","128")); N=int(os.getenv("N_PROMPTS","40"))
H={"Authorization":"Bearer x","Content-Type":"application/json"}
PROMPTS=[
 "Explain why a failure ledger is more valuable than a scoreboard for an engineering team.",
 "Write a Python function that parses an nginx access log line into a dict, with a docstring.",
 "A train leaves at 3:40 PM and arrives at 7:15 PM the same day. How long is the trip? Show your work.",
 "Summarize the tradeoffs between pinned and pageable host memory for GPU offload in three paragraphs.",
 "Write a bash one-liner that finds the ten largest files under /var/log and prints size and path.",
 "Describe the difference between speculative decoding with a draft model and with an MTP head.",
 "What does 'quality-neutral speedup' mean, and how would you test for it?",
 "Draft a terse status update: server migrated, 8-minute boot, one regression retracted.",
 "Given a 5-node RAID6 array of 20 TB disks, what is the usable capacity and why?",
 "Write a haiku about a GPU waiting for weights to load.",
 "Explain NVFP4 block scaling to a systems engineer who knows FP8 but not FP4.",
 "List five things that make a benchmark number publishable and one that disqualifies it.",
 "Convert this to JSON: name Milo, role handler, hosts 3, ports 30001 and 30003.",
 "Why does temperature-0 sampling on a batched MoE model still produce different outputs run to run?",
 "Write a SQL query returning the top 3 customers by total order value in 2025.",
 "Explain what a 'knee' is in a concurrency sweep and how you find it.",
 "Compose a two-sentence apology for publishing a metric you later retracted.",
 "What is the time complexity of building a suffix array naively, and of the SA-IS algorithm?",
 "Describe how transparent huge pages interact with shmem-backed allocations on Linux.",
 "Write a Dockerfile snippet that pins a base image by digest and installs curl.",
 "Refactor this: for i in range(len(xs)): print(xs[i]) — and say why.",
 "Explain the difference between RAID0 and RAID1 to a photographer.",
 "Write a Go function that reverses a UTF-8 string correctly.",
 "What are three failure modes of speculative decoding acceptance-rate measurements?",
 "Give a one-paragraph plain-English explanation of KV cache quantization risks.",
 "Write a regex that matches ISO-8601 dates with optional time and timezone.",
 "Explain why a 1M-token context costs more KV memory than a 131K one, with the arithmetic.",
 "Draft a commit message for: fix launcher flag rename, add env knob, no behavior change.",
 "Write a Python generator that yields Fibonacci numbers below a limit.",
 "Describe what 'chunked prefill' does and when you would lower the chunk size.",
 "List the steps to safely swap a served model with zero client-visible downtime.",
 "Write a limerick about a benchmark that turned out to be noise.",
 "What is the difference between MXFP4 and NVFP4 scale granularity?",
 "Write a shell function that retries a command 3 times with backoff.",
 "Explain to a new engineer why pinning image digests matters.",
 "Given TTFT 0.37 s for 6538 tokens, what is the prefill rate in tokens per second?",
 "Write a Python dataclass for a benchmark result with validation in __post_init__.",
 "Describe two ways a tool-call parser can silently fail and how to detect each.",
 "Summarize why a frozen docker-commit image is a reproducibility risk.",
 "Write a single sentence that a model would generate identically at temperature zero, and explain why it might not.",
]
def post(base,path,p):
    return json.load(urllib.request.urlopen(urllib.request.Request(base+path,data=json.dumps(p).encode(),headers=H),timeout=900))
def gen(base,prompt):
    r=post(base,"/chat/completions",{"model":MODEL,"messages":[{"role":"user","content":prompt}],"max_tokens":G,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}})
    return r["choices"][0]["message"]["content"] or ""
def score(base,text):
    r=post(base,"/completions",{"model":MODEL,"prompt":text,"max_tokens":1,"echo":True,"logprobs":1,"temperature":0})
    lp=r["choices"][0]["logprobs"]; n=len(lp["tokens"])-1  # drop the one generated token
    return lp["tokens"][:n],lp["token_logprobs"][:n],(lp.get("top_logprobs") or [None]*(n+1))[:n]
def phase_gen(base,out):
    rows=[]
    for i,pr in enumerate(PROMPTS[:N]):
        text=gen(base,pr)
        if not text.strip(): continue
        full=pr+"\n\n"+text; toks,lps,top=score(base,full); plen=len(score(base,pr+"\n\n")[0])
        rows.append({"i":i,"prompt":pr,"text":text,"tokens":toks,"logprobs":lps,"top":top,"start":plen})
        print(f"gen[{i}] {len(toks)-plen} gen tokens",flush=True)
    json.dump(rows,open(out,"w")); print("wrote",out,len(rows))
def phase_score(base,ref,out):
    rows=json.load(open(ref)); res=[]
    for r in rows:
        toks,lps,top=score(base,r["prompt"]+"\n\n"+r["text"])
        res.append({"i":r["i"],"tokens":toks,"logprobs":lps,"top":top}); print(f"score[{r['i']}] {len(toks)} tokens",flush=True)
    json.dump(res,open(out,"w")); print("wrote",out)
def phase_cmp(ref,cand):
    R={r["i"]:r for r in json.load(open(ref))}; C={r["i"]:r for r in json.load(open(cand))}
    d_all=[]; ag=[]; skipped=0
    for i,r in R.items():
        c=C.get(i)
        if not c or c["tokens"]!=r["tokens"]: skipped+=1; continue
        s=r["start"]
        for t,a,b,tp in zip(r["tokens"][s:],r["logprobs"][s:],c["logprobs"][s:],(c["top"] or [None]*len(c["tokens"]))[s:]):
            if a is None or b is None: continue
            d_all.append(abs(a-b))
            if tp: ag.append(max(tp,key=tp.get)==t)
    d_all.sort()
    print(f"SUMMARY prompts={len(R)} skipped_tokmismatch={skipped} gen_tokens={len(d_all)} mean|dlogp|={statistics.mean(d_all):.4f} median={d_all[len(d_all)//2]:.4f} p95={d_all[int(.95*len(d_all))]:.3f} max={d_all[-1]:.2f} cand_top1_agrees_with_ref_greedy={statistics.mean(ag) if ag else float('nan'):.4f}")
if __name__=="__main__":
    ph=sys.argv[1]
    if ph=="gen": phase_gen(os.environ["BASE"],sys.argv[2])
    elif ph=="score": phase_score(os.environ["BASE"],sys.argv[2],sys.argv[3])
    elif ph=="cmp": phase_cmp(sys.argv[2],sys.argv[3])
