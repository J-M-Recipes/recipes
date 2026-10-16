#!/usr/bin/env python3
"""gsm_eval.py OUTJSONL [N=200] — GSM8K test, first N, T=0, reasoning_effort=low, C4, exact numeric match on the last number in content.
Env: BASE (http://127.0.0.1:18100/v1) MODEL (muse) MAXTOK (4096) CONC (4)
"""
import json, os, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
OUT = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
BASE = os.getenv("BASE", "http://127.0.0.1:18100/v1"); MODEL = os.getenv("MODEL", "muse")
MAXTOK = int(os.getenv("MAXTOK", "4096")); CONC = int(os.getenv("CONC", "4"))
rows = [json.loads(l) for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gsm8k-test.jsonl"))][:N]


def num(s):
    m = re.findall(r"-?\d[\d,]*\.?\d*", (s or "").replace("$", ""))
    if not m: return None
    v = m[-1].replace(",", "").rstrip(".")
    try: return float(v)
    except ValueError: return None


def one(r):
    b = {"model": MODEL, "temperature": 0, "max_tokens": MAXTOK, "chat_template_kwargs": {"reasoning_effort": "low"},
         "messages": [{"role": "user", "content": r["q"] + "\n\nSolve it, then give the final answer as a number on the last line."}]}
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request(BASE + "/chat/completions", data=json.dumps(b).encode(),
                                                                    headers={"Content-Type": "application/json"}), timeout=1800))
        c = d["choices"][0]["message"].get("content") or ""; fr = d["choices"][0]["finish_reason"]; u = d["usage"]["completion_tokens"]
    except Exception as e:  # noqa
        c, fr, u = "", "error:" + repr(e)[:100], 0
    p = num(c)
    return {"i": r["i"], "gold": r["a"], "pred": p, "ok": p is not None and abs(p - float(r["a"])) < 1e-6, "finish": fr, "ctoks": u, "content_tail": c[-200:]}


with ThreadPoolExecutor(CONC) as ex:
    res = list(ex.map(one, rows))
with open(OUT, "w") as f:
    for x in res: f.write(json.dumps(x) + "\n")
ok = sum(x["ok"] for x in res); print(f"GSM8K {ok}/{len(res)} = {ok/len(res):.3f}  truncated={sum(x['finish']=='length' for x in res)} errors={sum(str(x['finish']).startswith('error') for x in res)}")
