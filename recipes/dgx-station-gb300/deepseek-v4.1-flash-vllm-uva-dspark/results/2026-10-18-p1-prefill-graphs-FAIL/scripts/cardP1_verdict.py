#!/usr/bin/env python3
"""Card P1 verdict: CAND (CG8K, else CG2K) vs MEAN of CTL1/CTL2. Bars pre-registered in cardP1_runner.sh header.
Usage: cardP1_verdict.py <results-dir>   -> markdown on stdout; first 'SUFFIX <x>' line = container suffix."""
import json, os, re, sys, glob
R = sys.argv[1] if len(sys.argv) > 1 else "$HOME/dsv41/1017/results-cardP1"
REF_HASH = "62426808ae2b0ddbdfbf2278702ac4e367078b9884817c169ddee40495c4e6a6"

def j(p):
    try:
        return json.load(open(p))
    except Exception:
        return None

def rt(tag):
    o = {}
    for k in ("c1-r1", "c1-r2", "c1-r3", "c4", "c16", "c32"):
        d = j(f"{R}/rt-{tag}/{k}.json")
        if d and "summary" in d:
            o[k] = d["summary"]
    c1 = [o[k]["decode_agg"] for k in ("c1-r1", "c1-r2", "c1-r3") if k in o]
    tt = [o[k]["median_ttft_s"] for k in ("c1-r1", "c1-r2", "c1-r3") if k in o]
    ac = [o[k].get("acceptance_len") or 0 for k in ("c1-r1", "c1-r2", "c1-r3") if k in o]
    return {
        "C1": sum(c1) / len(c1) if c1 else None,
        "TTFT8K": sum(tt) / len(tt) if tt else None,
        "acc1": sum(ac) / len(ac) if ac else None,
        "C4": o.get("c4", {}).get("decode_agg"),
        "C16": o.get("c16", {}).get("decode_agg"),
        "C32out": o.get("c32", {}).get("output_tok_s"),
        "C16ttft": o.get("c16", {}).get("median_ttft_s"),
    }

def knee(tag, r="r2"):
    d = j(f"{R}/knee-P1-{tag}-{r}.json")
    if not d:
        return {}
    rows = d["rows"] if isinstance(d, dict) and "rows" in d else d
    out = {}
    for row in rows:
        if isinstance(row, dict):
            c = row.get("conc") or row.get("concurrency"); v = row.get("agg") or row.get("agg_tok_s") or row.get("tok_s")
        else:
            c, v = row[0], row[1]
        try:
            out[int(c)] = float(v)
        except Exception:
            pass
    return out

def replay(tag, r):
    d = j(f"{R}/replayc-P1-{tag}-{r}-n4.json")
    return d.get("agg_tok_s") if d else None

def gaunt(tag):
    d = j(f"{R}/gauntlet-{tag}/summary.json")
    return f"{d.get('success_count')}/{d.get('task_count')}" if d else "–"

def cold(tag):
    try:
        t = open(f"{R}/cold64k-{tag}.txt").read()
        m = re.search(r"ttft_s=\[([0-9.]+)", t)
        return float(m.group(1)) if m else None
    except Exception:
        return None

def facts(tag):
    f = {}
    try:
        t = open(f"{R}/facts-{tag}.txt").read().splitlines()
    except Exception:
        return f
    kv = [l for l in t if "GPU KV cache size" in l]
    if kv:
        f["kv"] = int(re.search(r"GPU KV cache size: ([0-9,]+)", kv[-1]).group(1).replace(",", ""))
    for key, pat in (("kvmem", r"Available KV cache memory: ([0-9.]+) GiB"), ("graph", r"Graph capturing finished in (\d+) secs, took ([0-9.]+) GiB"),
                     ("pool", r"CUDA graph pool memory: ([0-9.]+) GiB \(actual\), ([0-9.]+) GiB \(estimated\)")):
        ms = [re.search(pat, l) for l in t]; ms = [m for m in ms if m]
        if ms:
            f[key] = ms[-1].groups()
    loaded = [l for l in t if re.search(r"Loaded \d+ configs", l)]
    if loaded:
        f["hash"] = (re.search(r"/([0-9a-f]{64})/", loaded[-1]) or [None, None])[1] if re.search(r"/([0-9a-f]{64})/", loaded[-1]) else None
        f["nconf"] = re.search(r"Loaded (\d+) configs", loaded[-1]).group(1)
    f["tune_starts"] = sum("Autotuning process starts" in l for l in t)
    return f

def boot_s(tag):
    try:
        for l in open(f"{R}/runner.log"):
            m = re.search(rf"boot->ready {tag} (\S+)", l)
            if m:
                return m.group(1)
    except Exception:
        pass
    return "–"

tags = [t for t in ("CTL1", "CG8K", "CG2K", "CTL2") if os.path.exists(f"{R}/facts-{t}.txt")]
A = {}
for t in tags:
    k = knee(t); f = facts(t)
    A[t] = dict(rt(t), knee1=k.get(1), knee8=k.get(8), knee16=k.get(16), rep1=replay(t, "r1"), rep2=replay(t, "r2"),
                g=gaunt(t), cold=cold(t), boot=boot_s(t), **{"f_" + a: b for a, b in f.items()})

def fm(v, p=1):
    return "–" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))

print("SUFFIX PENDING" if False else "", end="")
lines = ["| arm | RT C1 (acc) | C4 | C16 | C32 out | TTFT 8K | cold 64K TTFT | knee r2 C1/C8/C16 | replay r1/r2 | gauntlet | KV tokens | KV mem GiB | graph capture | boot | autotune |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for t in tags:
    a = A[t]
    gr = a.get("f_graph"); po = a.get("f_pool")
    gtxt = (f"{gr[1]} GiB/{gr[0]} s" if gr else "–") + (f"; pool est {po[1]}" if po else "")
    h = a.get("f_hash"); hh = ("hit " + h[:8]) if h == REF_HASH else (("NEW " + h[:8]) if h else "–")
    lines.append(f"| {t} | {fm(a['C1'])} ({fm(a['acc1'],2)}) | {fm(a['C4'])} | {fm(a['C16'])} | {fm(a['C32out'])} | {fm(a['TTFT8K'],2)} | {fm(a['cold'],2)} | "
                 f"{fm(a['knee1'],0)}/{fm(a['knee8'],0)}/{fm(a['knee16'],0)} | {fm(a['rep1'],0)}/{fm(a['rep2'],0)} | {a['g']} | {a.get('f_kv','–')} | "
                 f"{(a.get('f_kvmem') or ['–'])[0]} | {gtxt} | {a['boot']} | {hh} (Loaded {a.get('f_nconf','–')}) |")

ctl = [A[t] for t in ("CTL1", "CTL2") if t in A]
def cm(k):
    v = [c[k] for c in ctl if c.get(k) is not None]
    return sum(v) / len(v) if v else None
cand = "CG8K" if "CG8K" in A and A["CG8K"].get("C1") is not None else ("CG2K" if "CG2K" in A else None)
out = []
suffix = "DONE"
if cand and ctl:
    c = A[cand]
    kvc = cm("f_kv"); kvx = c.get("f_kv")
    loss = (kvc - kvx) if (kvc and kvx) else None
    def d(k):
        b = cm(k); x = c.get(k)
        return (x / b - 1) * 100 if (b and x) else None
    bars = {
        "TTFT 8K <= 0.55 s": (c["TTFT8K"] is not None and c["TTFT8K"] <= 0.55, f"{fm(c['TTFT8K'],2)} s (CTL mean {fm(cm('TTFT8K'),2)})"),
        "KV loss <= 0.3M": (loss is not None and loss <= 300000, f"loss {loss:,.0f}" if loss is not None else "–"),
        "RT C1 >= -2%": (d("C1") is not None and d("C1") >= -2, f"{fm(d('C1'))}%"),
        "replay r2 >= -2%": (d("rep2") is not None and d("rep2") >= -2, f"{fm(d('rep2'))}%"),
    }
    fail_kv = loss is not None and loss > 500000
    dec = [d(k) for k in ("C1", "C4", "C16")]
    fail_dec = any(x is not None and x < -3 for x in dec)
    win = all(v[0] for v in bars.values())
    verdict = "FAIL" if (fail_kv or fail_dec) else ("WIN" if win else "NEUTRAL")
    h = c.get("f_hash"); prov = (h != REF_HASH)
    suffix = verdict + ("-PROV" if prov else "")
    out.append(f"**Candidate arm:** {cand}  ·  **Verdict: {verdict}**{' (PROVISIONAL — new autotune hash)' if prov else ' (autotune hash hit)'}")
    for k, v in bars.items():
        out.append(f"- WIN bar {k}: {'PASS' if v[0] else 'MISS'} — {v[1]}")
    out.append(f"- FAIL bar KV loss > 0.5M: {'HIT' if fail_kv else 'clear'}; decode −3% (C1/C4/C16 Δ = {', '.join(fm(x) for x in dec)} %): {'HIT' if fail_dec else 'clear'}")
    out.append(f"- other Δ vs CTL mean: C32 out {fm(d('C32out'))}%, TTFT 8K {fm(d('TTFT8K'))}%, cold 64K {fm(d('cold'))}%, C16 TTFT {fm(d('C16ttft'))}%, knee C8 {fm(d('knee8'))}%")
    if "CG8K" in A and cand == "CG2K":
        out.append(f"- CG8K: KV {A['CG8K'].get('f_kv','–')} (benches skipped or boot failed)")
else:
    out.append("**No measured candidate arm** — see runner.log")
print(f"SUFFIX {suffix}")
print("\n".join(lines))
print()
print("\n".join(out))
