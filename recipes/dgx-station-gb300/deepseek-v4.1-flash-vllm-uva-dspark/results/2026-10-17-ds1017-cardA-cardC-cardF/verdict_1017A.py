#!/usr/bin/env python3
"""Card A verdict: DeepSeek-V4.1-Flash-1017 (CAND) vs df42c109 (CTL1/CTL2), identical v21 launch, same window."""
import json, os, re, sys
R = sys.argv[1] if len(sys.argv) > 1 else "/home/milo/dsv41/1017/results"

def knee(tag, r):
    p = f"{R}/knee-DS1017A-{tag}-{r}.json"
    if not os.path.exists(p):
        return {}
    d = json.load(open(p))
    rows = d["rows"] if isinstance(d, dict) and "rows" in d else d
    out = {}
    for row in rows:
        if isinstance(row, dict):
            c = row.get("conc") or row.get("concurrency")
            v = row.get("agg") or row.get("agg_tok_s") or row.get("tok_s")
        else:
            c, v = row[0], row[1]
        try:
            out[int(c)] = float(v)
        except Exception:
            pass
    return out

def gaunt(tag):
    p = f"{R}/gauntlet-{tag}/summary.json"
    if not os.path.exists(p):
        return "missing"
    d = json.load(open(p))
    return f"{d.get('success_count')}/{d.get('task_count')}"

def rep(tag):
    out = []
    for r in ("r1", "r2"):
        p = f"{R}/replayc-DS1017A-{tag}-{r}-n4.json"
        if os.path.exists(p):
            d = json.load(open(p))
            out.append(round(d.get("agg_tok_s") or d.get("agg") or 0, 1))
    return out

def fixture(tag):
    """Parse the window log lines: [TAG] cat  tok/s  accept=..%  acc/step=.."""
    p = f"{R}/window-DS1017A-{tag}.log"
    rows = {}
    if os.path.exists(p):
        for line in open(p):
            m = re.match(rf"\[DS1017A-{tag}\]\s+(\w+)\s+([\d.]+) tok/s\s+accept=([\d.]+)%\s+acc/step=([\d.]+)", line)
            if m:
                rows[m.group(1)] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    return rows

print("Card A — DeepSeek-V4.1-Flash-1017 (CAND) vs df42c109 (CTL1/CTL2), identical v21 launch, same window.")
print("knee r2 agg tok/s")
print("arm    C1     C2     C4     C8     C12    C16   | gauntlet | replay n=4 r1/r2")
ks = {}
for t in ("CTL1", "CAND", "CTL2"):
    k = knee(t, "r2"); ks[t] = k
    print(f"{t:5} " + " ".join(f"{k.get(c, 0):6.1f}" for c in (1, 2, 4, 8, 12, 16)) + f"  | {gaunt(t):>7} | {rep(t)}")
if ks["CTL1"] and ks["CTL2"] and ks["CAND"]:
    print("CAND vs mean(CTL): " + "  ".join(
        f"C{c} {100*(ks['CAND'].get(c,0)/((ks['CTL1'].get(c,1)+ks['CTL2'].get(c,1))/2)-1):+.1f}%" for c in (1, 2, 4, 8, 12, 16)))
print()
print("agent fixture: accept% (tok/s, acc/step)")
for t in ("CTL1", "CAND", "CTL2"):
    f = fixture(t)
    print(f"{t:5} " + "  ".join(f"{c} {f[c][1]:.1f}% ({f[c][0]:.0f}, {f[c][2]:.2f})" for c in ("prose", "structured", "code", "shell_ops", "tool_json") if c in f))
print()
print("Parity: CAND vs CTL = different weights (expected to differ). CTL1 vs CTL2 = instrument baseline.")
import glob
def load(t):
    fs = glob.glob(f"{R}/parity/parity-DS1017A-{t}.json")
    return json.load(open(fs[0]))["rows"] if fs else None
a, b, c = load("CTL1"), load("CAND"), load("CTL2")
def cmp(x, y, label):
    if not x or not y:
        print(label, "missing"); return
    xs = {p["id"]: p for p in x}; ys = {p["id"]: p for p in y}
    key = lambda p: (p.get("content"), p.get("tool_norm"), p.get("finish"))
    nontool = [k for k in xs if xs[k].get("finish") != "tool_calls"]
    same_nt = sum(1 for k in nontool if k in ys and key(xs[k]) == key(ys[k]))
    same_all = sum(1 for k in xs if k in ys and key(xs[k]) == key(ys[k]))
    print(f"parity {label}: non-tool identical {same_nt}/{len(nontool)} · all {same_all}/{len(xs)}")
cmp(a, c, "CTL1 vs CTL2"); cmp(a, b, "CAND vs CTL1")
