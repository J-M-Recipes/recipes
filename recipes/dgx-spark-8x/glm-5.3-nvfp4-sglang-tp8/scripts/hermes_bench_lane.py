#!/usr/bin/env python3
"""Hermes-bench: same `hermes chat -Q` harness against two serving engines. One lane per invocation.
Usage: HERMES_HOME=... python3 run_lane.py <lane> <provider> <model>   e.g.  sglang 'custom[sglang]' glm-5.3-nvfp4-sglang
Fixtures are frozen here; each run is a fresh session; wall = subprocess wall-clock (includes the ~15 s CLI floor)."""
import json, os, random, re, string, subprocess, sys, time, pathlib
lane, prov, model = sys.argv[1:4]
OUT = pathlib.Path(f"results/{lane}"); OUT.mkdir(parents=True, exist_ok=True)
TMP = pathlib.Path(os.environ["TMPDIR"]) / f"gauntlet-{lane}"; 
def hermes(prompt, budget=900):
    cmd = ["hermes","chat","-Q","--ignore-rules","--provider",prov,"-m",model,"-t","terminal","--run-budget",str(budget),"--format","stream-json","-q",prompt]
    t0=time.time(); p=subprocess.run(cmd,capture_output=True,text=True,timeout=budget+120); w=time.time()-t0
    final=""; tools=0; agent_ms=None; toks={}
    for line in p.stdout.splitlines():
        try: ev=json.loads(line)
        except Exception: continue
        t=ev.get("type","")
        if t=="tool_use": tools+=1
        if t=="result": final=ev.get("text") or ""; agent_ms=ev.get("duration_ms"); toks=ev.get("tokens") or {}
    global LAST; LAST={"agent_ms":agent_ms,"tokens":toks}
    return w, final, tools, p.returncode, p.stdout[-4000:], p.stderr[-1500:]
def gauntlet():
    import shutil; shutil.rmtree(TMP, ignore_errors=True); TMP.mkdir(parents=True)
    nonce="".join(random.choice(string.ascii_uppercase+string.digits) for _ in range(8))
    for i in range(1,21):
        body = f"TOKEN: {nonce}\n" if i==20 else f"NEXT: {TMP}/f{i+1:02d}.txt\n"
        (TMP/f"f{i:02d}.txt").write_text("log line filler\n"*3+body)
    return nonce, f"Start by reading {TMP}/f01.txt with one terminal command. Each file names the next file to read on a NEXT: line; read exactly one file per command and follow the chain until a file contains a TOKEN: line. Then reply with just the token."
FIX = [
 ("oneword-1","Reply with exactly one word: hello", lambda s: re.sub(r"[^a-z]","",s.lower())=="hello"),
 ("oneword-2","Reply with exactly one word: hello", lambda s: re.sub(r"[^a-z]","",s.lower())=="hello"),
 ("count300-1","Count from 1 to 300 as one comma-separated line. Do not use tools or code; just write the numbers.", lambda s: len(re.findall(r"\b\d+\b",s))>=300 and "300" in s),
 ("count300-2","Count from 1 to 300 as one comma-separated line. Do not use tools or code; just write the numbers.", lambda s: len(re.findall(r"\b\d+\b",s))>=300 and "300" in s),
 ("code","Write a Python function is_prime(n) with a one-line docstring and a short __main__ demo. Code only, no explanation, no tools.", lambda s: "def is_prime" in s),
 ("artifact","Write a 400-word blog post titled 'Why unified memory changes inference serving'. Plain markdown, no tools, no preamble.", lambda s: 250<=len(s.split())<=800),
]
rows=[]; LAST={}
hermes("Reply with exactly one word: ready", 300)  # warm-up (deps, caches) — not scored
for fid,prompt,ok in FIX:
    w,final,tools,rc,so,se=hermes(prompt); r={"id":fid,"wall_s":round(w,1),"agent_s":round((LAST["agent_ms"] or 0)/1000,1),"out_tok":LAST["tokens"].get("output"),"ok":bool(ok(final)),"tools":tools,"rc":rc,"final_head":final[:200]}
    rows.append(r); (OUT/f"{fid}.raw.txt").write_text(so+"\n---STDERR---\n"+se); print(json.dumps(r),flush=True)
nonce,prompt=gauntlet(); w,final,tools,rc,so,se=hermes(prompt,1200)
r={"id":"gauntlet20","wall_s":round(w,1),"agent_s":round((LAST["agent_ms"] or 0)/1000,1),"out_tok":LAST["tokens"].get("output"),"ok":nonce in (final or ""),"tools":tools,"rc":rc,"final_head":(final or "")[:200],"nonce":nonce}
rows.append(r); (OUT/"gauntlet20.raw.txt").write_text(so+"\n---STDERR---\n"+se); print(json.dumps(r),flush=True)
(OUT/"summary.json").write_text(json.dumps({"lane":lane,"provider":prov,"model":model,"when":time.strftime("%Y-%m-%d %H:%M:%S"),"rows":rows},indent=1))
print("LANE-DONE",lane)
