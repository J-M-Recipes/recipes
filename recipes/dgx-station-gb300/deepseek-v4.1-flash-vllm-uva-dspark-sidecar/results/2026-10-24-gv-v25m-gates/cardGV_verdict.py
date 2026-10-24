#!/usr/bin/env python3
"""cardGV_verdict.py <results-cardGV> <results-cardSC2> — pre-registered gates for v25m (see cardGV_runner.sh)."""
import json
import re
import sys
from pathlib import Path

R, S = Path(sys.argv[1]), Path(sys.argv[2])
log = (R / "runner.log").read_text(errors="replace") if (R / "runner.log").exists() else ""
G = {}


def jl(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def last(pat):
    m = re.findall(pat, log)
    return m[-1] if m else None


# G1 TF flips
f1, f2 = last(r"FLIPGATE V-vs-CTL1: (.*)"), last(r"FLIPGATE V-vs-v25: (.*)")
G["G1 TF flips vs v23 CTL1 and vs v25 static"] = (
    "PASS" if f1 and f2 and "GATE PASS" in f1 and "GATE PASS" in f2 else "FAIL", f"vs CTL1: {f1} | vs v25: {f2}")


# G2 GSM8K
def acc(p):
    j = jl(p)
    return None if not j else j["accuracy"]["accuracy"]


v, c = acc(R / "gsm8k1319-V.json"), acc(S / "gsm8k1319-C.json")
G["G2 GSM8K-1319 >= v25 - 1 pt"] = ("PASS" if v is not None and c is not None and v >= c - 0.01 else "FAIL",
                                    f"v25m {v} vs v25 {c}")


# G3 BFCL
def bf(p):
    j = jl(p)
    if not j:
        return None
    for k in ("all", "accuracy", "acc"):
        if isinstance(j.get(k), (int, float)):
            return j[k]
        if isinstance(j.get(k), dict) and "acc" in j[k]:
            return j[k]["acc"]
    m = re.search(r"all (\d+)/(\d+)", json.dumps(j))
    return int(m.group(1)) / int(m.group(2)) if m else None


def bfl(tag, suite, d):
    t = (d / f"bfcl-{tag}-{suite}.log").read_text(errors="replace") if (d / f"bfcl-{tag}-{suite}.log").exists() else ""
    m = re.findall(r"all (\d+)/(\d+)", t)
    return int(m[-1][0]) / int(m[-1][1]) if m else None


bd = [(bfl("V", s, R), bfl("C", s, S)) for s in ("dev", "heldout")]
ok = all(a is not None and b is not None and a >= b - 0.01 for a, b in bd)
G["G3 BFCL dev + heldout >= v25 - 1 pt"] = ("PASS" if ok else "FAIL",
                                            f"dev {bd[0][0]} vs {bd[0][1]} · heldout {bd[1][0]} vs {bd[1][1]}")
# G4 gauntlet
gs = jl(R / "gauntlet-V/summary.json") or {}
G["G4 gauntlet 12/12"] = ("PASS" if gs.get("success_count") == 12 and gs.get("task_count") == 12 else "FAIL",
                          f"{gs.get('success_count')}/{gs.get('task_count')}")
# G5 needle
nd = (jl(R / "needle-V.json") or {}).get("selected_summary") or {}
G["G5 needle 500/500 exact"] = ("PASS" if nd.get("exact") == 500 and nd.get("attempted") == 500 else "FAIL",
                                f"exact {nd.get('exact')}/{nd.get('attempted')} errors {nd.get('errors')}")
# G6 effort
ev, ec = last(r"EFFORT V: (\{.*?\})"), None
m = re.search(r"EFFORT C: (\{.*?\})", (S / "SUMMARY.txt").read_text(errors="replace")) if (S / "SUMMARY.txt").exists() else None
ec = m.group(1) if m else None
G["G6 reasoning_effort codes == v25"] = ("PASS" if ev and ec and json.loads(ev) == json.loads(ec) else "FAIL", f"v25m {ev} · v25 {ec}")
# G7 image
im = last(r"IMAGE V: (.*)")
G["G7 image request"] = ("PASS" if im and "http=200" in im and "names_content=True" in im else "FAIL", im)


# G8 per-turn tool emission
def tn(p):
    j = jl(p)
    if not j or not j.get("passes"):
        return None
    return [{"tool": sum(1 for r in rows if r.get("has_tool") and r.get("emitted")),
             "text": sum(1 for r in rows if not r.get("has_tool") and r.get("emitted")),
             "err": sum(1 for r in rows if "error" in r)} for rows in j["passes"]]


TV, TA, TD = tn(R / "turns-SC2-V.json"), tn(S / "turns-SC2-A.json"), tn(S / "turns-SC2-D.json")
if TV and TA and TD:
    ct = min(min(p["tool"] for p in TA), min(p["tool"] for p in TD))
    cx = max(max(p["text"] for p in TA), max(p["text"] for p in TD))
    ok = all(p["tool"] >= ct - 1 and p["text"] <= cx + 1 for p in TV) and sum(p["err"] for p in TV) == 0
    G["G8 per-turn tool emission"] = ("PASS" if ok else "FAIL",
                                      f"v25m tool {[p['tool'] for p in TV]} (bar >= {ct - 1}), text {[p['text'] for p in TV]} (bar <= {cx + 1}), errors {sum(p['err'] for p in TV)}")
else:
    G["G8 per-turn tool emission"] = ("FAIL", "turns file missing")
# G9 cosine
gk = last(r"GATE K: (PASS|FAIL) (.*)")
G["G9 cosine gate (check boot)"] = (gk[0] if gk else "FAIL", gk[1] if gk else "K gate missing")
# G10 timeouts
tv = jl(R / "timeouts-V.json") or {}
kj = None
if (R / "gate-K.json").exists():
    t = (R / "gate-K.json").read_text()
    try:
        kj = json.loads(t[t.index("{"):])
    except Exception:
        kj = None
ok = tv.get("shm_delta_timeouts") == 0 and kj is not None and (kj.get("peer_timeouts") in (0, None)) and (kj.get("hook_timeout_lines") in (0, None))
G["G10 0 sidecar timeouts (V, K)"] = ("PASS" if ok else "FAIL",
                                      f"V shm delta {tv.get('shm_delta_timeouts')} · K peer {kj and kj.get('peer_timeouts')} hook {kj and kj.get('hook_timeout_lines')}")
allp = all(v[0] == "PASS" for v in G.values())
print(f"**Verdict: {'PASS' if allp else 'FAIL'}** — v25m fidelity gates ({sum(v[0] == 'PASS' for v in G.values())}/{len(G)} PASS). "
      "Reference: SC2 C arm (v25 static map). Status stays experimental until the 48 h soak completes.\n")
print("| gate | result | detail |\n|---|---|---|")
for k, (r, d) in G.items():
    print(f"| {k} | {r} | {str(d)[:300]} |")
print("\nReported (not gated):")
for pat in (r" RT V: (.*)", r"KNEE V r1: (.*)", r"KNEE V r2: (.*)", r"REPLAY V r2: (.*)", r"TIMEOUTS V: (.*)", r"POWER V: (.*)", r"FACTS P: (.*)", r"SOAK clock started (.*)"):
    x = last(pat)
    print(f"- {pat.split(':')[0]}: {x}")
