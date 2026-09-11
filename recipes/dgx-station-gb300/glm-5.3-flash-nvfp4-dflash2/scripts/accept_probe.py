#!/usr/bin/env python3
"""Acceptance-rate probe: run one prompt class at a time on an idle server, scrape the scheduler's
'Decode batch ... accept len / accept rate' lines emitted DURING that request, report per-class means.
Uses docker logs --since to window each run. Run on the box."""
import json, os, subprocess, time, urllib.request, statistics, sys
BASE=os.getenv("BASE_URL","http://127.0.0.1:30001/v1"); MODEL=os.getenv("MODEL","glm-5.3-flash")
CONT=os.getenv("CONT","glmf-g2-new-new-dflash"); TAG=os.getenv("TAG","x")
H={"Content-Type":"application/json"}
CLASSES={
 "prose":   ("Write a 350-word essay on why engineers should keep failure ledgers. No headings, no lists.",320),
 "history": ("Write a 500-word history of the Roman aqueduct system: engineering, politics, and decline.",512),
 "code":    ("Write a Python module with a dataclass BenchResult, a parse_nginx_line(line:str)->dict function with type hints and docstring, and a small CLI using argparse.",450),
 "shell":   ("Give me a bash script that rotates logs in /var/log/app, keeps 7 days, compresses with zstd, and is safe to run from cron. Include comments.",300),
 "json":    ("Return ONLY a JSON array of 25 objects with fields id (int), city (string), lat, lon (floats) for real US cities.",400),
 "math":    ("Solve step by step: a train leaves at 3:40 PM at 72 mph and another at 4:10 PM at 90 mph on the same track. When does the second catch the first? Show all arithmetic.",350),
 "agentic": ("You are an SRE. A pod is CrashLoopBackOff. Walk through, as a numbered runbook with exact kubectl commands, how you would diagnose it. Be concrete.",400),
}
def run(prompt,mt):
    p={"model":MODEL,"messages":[{"role":"user","content":prompt}],"max_tokens":mt,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}
    t0=time.time(); r=json.load(urllib.request.urlopen(urllib.request.Request(BASE+"/chat/completions",data=json.dumps(p).encode(),headers=H),timeout=300))
    return r["usage"]["completion_tokens"], time.time()-t0
def scrape(since_iso):
    out=subprocess.run(["docker","logs","--since",since_iso,CONT],capture_output=True,text=True).stdout+subprocess.run(["docker","logs","--since",since_iso,CONT],capture_output=True,text=True).stderr
    lens=[];rates=[]
    for line in out.replace("\r","\n").splitlines():
        if "Decode batch" in line and "accept len:" in line:
            try:
                lens.append(float(line.split("accept len:")[1].split(",")[0])); rates.append(float(line.split("accept rate:")[1].split(",")[0]))
            except: pass
    return lens,rates
rows=[]
for name,(prompt,mt) in CLASSES.items():
    run(prompt,64)  # warm the prefix/kernels lightly, ignore
    time.sleep(1.5)
    t_start=time.time()
    n,dt=run(prompt,mt); time.sleep(1.0)
    lens,rates=scrape(f"{int(time.time()-t_start)+2}s")
    al=statistics.mean(lens) if lens else float("nan"); ar=statistics.mean(rates) if rates else float("nan")
    rows.append((name,n,dt,n/dt,al,ar,len(lens)))
    print(f"[{TAG}] {name:8s} {n:4d} tok {dt:5.2f}s -> {n/dt:6.1f} tok/s | accept len {al:.2f} rate {ar:.2f} (n_lines={len(lens)})",flush=True)
good=[r for r in rows if r[4]==r[4]]
if good: print(f"[{TAG}] MEAN accept len {statistics.mean(r[4] for r in good):.2f} rate {statistics.mean(r[5] for r in good):.2f}  mean C1 {statistics.mean(r[3] for r in rows):.1f}")
else: print(f"[{TAG}] no accept lines scraped; mean C1 {statistics.mean(r[3] for r in rows):.1f}")
