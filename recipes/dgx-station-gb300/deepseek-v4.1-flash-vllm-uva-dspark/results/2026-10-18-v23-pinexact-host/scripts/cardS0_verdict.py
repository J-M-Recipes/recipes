#!/usr/bin/env python3
"""Card S0 verdict: S0 (PYTORCH_CUDA_ALLOC_CONF=pinned_max_cached_size_mb:1024) vs MEAN of CTL1/CTL2.
Bars pre-registered in cardS0_runner.sh header. Usage: cardS0_verdict.py <results-dir>  -> markdown; first line 'SUFFIX <x>'."""
import json, os, re, sys, glob
R = sys.argv[1] if len(sys.argv) > 1 else "$HOME/dsv41/1017/results-cardS0"
REF_HASH = "62426808ae2b0ddbdfbf2278702ac4e367078b9884817c169ddee40495c4e6a6"
KV_REF = 2504891
TAGS = ("CTL1", "S0", "CTL2")

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
    return {"C1": sum(c1) / len(c1) if c1 else None, "TTFT8K": sum(tt) / len(tt) if tt else None,
            "acc1": sum(ac) / len(ac) if ac else None, "C4": o.get("c4", {}).get("decode_agg"),
            "C16": o.get("c16", {}).get("decode_agg"), "C32out": o.get("c32", {}).get("output_tok_s")}

def knee(tag, r="r2"):
    d = j(f"{R}/knee-S0-{tag}-{r}.json")
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
    d = j(f"{R}/replayc-S0-{tag}-{r}-n4.json")
    return d.get("agg_tok_s") if d else None

def gaunt(tag):
    d = j(f"{R}/gauntlet-{tag}/summary.json")
    return (d.get("success_count"), d.get("task_count")) if d else (None, None)

def cold(tag):
    try:
        m = re.search(r"ttft_s=\[([0-9.]+)", open(f"{R}/cold64k-{tag}.txt").read())
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
    loaded = [l for l in t if re.search(r"Loaded \d+ configs", l)]
    if loaded:
        m = re.search(r"/([0-9a-f]{64})/", loaded[-1]); f["hash"] = m.group(1) if m else None
        f["nconf"] = re.search(r"Loaded (\d+) configs", loaded[-1]).group(1)
    f["saved"] = sum(bool(re.search(r"Saved \d+ configs", l)) for l in t)
    # hook guard: last boot's rehome layer lines (40 per boot)
    rl = [l for l in t if "rehome layer=" in l][-40:]
    ha = {}
    for l in rl:
        m = re.search(r"rehome layer=(\d+) .*host_avail=([0-9.]+)GiB", l)
        if m:
            ha[int(m.group(1))] = float(m.group(2))
    if ha:
        f["L9"] = ha.get(9); f["minha"] = min(ha.values())
    dn = [l for l in t if "rehome done" in l]
    if dn:
        m = re.search(r"host_avail=([0-9.]+)GiB", dn[-1]); f["done_ha"] = float(m.group(1)) if m else None
        m = re.search(r"pinned_expert=([0-9.]+)GiB", dn[-1]); f["pinned_expert"] = float(m.group(1)) if m else None
    f["abort"] = [l for l in t if "PIN_HOT abort" in l][-1:] or None
    return f

def hm(tag, ph):
    o = {}
    try:
        for l in open(f"{R}/hostmem-{tag}-{ph}.txt"):
            m = re.match(r"meminfo (\w+): ([0-9.]+) GiB", l)
            if m: o[m.group(1)] = float(m.group(2))
            m = re.match(r"cgroup (\S+) ([0-9.]+) GiB", l)
            if m: o["cg_" + m.group(1)] = float(m.group(2))
            m = re.match(r"pinned_devzero maps=(\d+) size_gib=([0-9.]+) rss_gib=([0-9.]+)", l)
            if m: o["pin_maps"], o["pin_size"], o["pin_rss"] = int(m.group(1)), float(m.group(2)), float(m.group(3))
    except Exception:
        pass
    return o

def boot_s(tag):
    try:
        for l in open(f"{R}/runner.log"):
            m = re.search(rf"boot->ready {tag} (\S+)", l)
            if m: return m.group(1)
    except Exception:
        pass
    return "–"

def parity(tag):
    fs = glob.glob(f"{R}/parity/parity-S0-{tag}.json")
    return json.load(open(fs[0]))["rows"] if fs else None

def pcmp(x, y):
    if not x or not y:
        return None
    xs = {p["id"]: p for p in x}; ys = {p["id"]: p for p in y}
    key = lambda p: (p.get("content"), p.get("tool_norm"), p.get("finish"))
    nontool = [k for k in xs if xs[k].get("finish") != "tool_calls"]
    same_nt = sum(1 for k in nontool if k in ys and key(xs[k]) == key(ys[k]))
    same_all = sum(1 for k in xs if k in ys and key(xs[k]) == key(ys[k]))
    return same_nt, len(nontool), same_all, len(xs)

tags = [t for t in TAGS if os.path.exists(f"{R}/facts-{t}.txt")]
A = {}
for t in tags:
    k = knee(t); f = facts(t); hs = hm(t, "steady"); hd = hm(t, "steady-dc"); hb = hm(t, "bound"); g = gaunt(t)
    A[t] = dict(rt(t), knee1=k.get(1), knee8=k.get(8), knee16=k.get(16), rep1=replay(t, "r1"), rep2=replay(t, "r2"),
                gs=g[0], gt=g[1], cold=cold(t), boot=boot_s(t), **{"f_" + a: b for a, b in f.items()},
                pin=hs.get("pin_size"), pin_rss=hs.get("pin_rss"), pin_maps=hs.get("pin_maps"), cg_shmem=hs.get("cg_shmem"),
                cg_cur=hs.get("cg_memory.current"), shmem=hs.get("Shmem"), mavail=hs.get("MemAvailable"),
                mavail_dc=hd.get("MemAvailable"), mavail_bound=hb.get("MemAvailable"), mlocked=hs.get("Mlocked"), unev=hs.get("Unevictable"))

def fm(v, p=1):
    return "–" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))

PAR = {}
for a, b in (("CTL1", "CTL2"), ("S0", "CTL1"), ("S0", "CTL2")):
    PAR[(a, b)] = pcmp(parity(a), parity(b))

perf = ["| arm | RT C1 (acc) | C4 | C16 | C32 out | TTFT 8K | cold 64K TTFT | knee r2 C1/C8/C16 | replay r1/r2 | gauntlet | KV tokens | boot | autotune |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
mem = ["| arm | pinned /dev/zero maps (n · Size GiB) | cgroup shmem GiB | cgroup memory.current GiB | host Shmem GiB | MemAvailable steady / after drop_caches GiB | Mlocked / Unevictable GiB | hook host_avail L9 / min / rehome-done GiB | pinned_expert GiB |",
       "|---|---|---|---|---|---|---|---|---|"]
for t in tags:
    a = A[t]
    h = a.get("f_hash"); hh = ("hit " + h[:8]) if h == REF_HASH else (("NEW " + h[:8]) if h else "–")
    perf.append(f"| {t} | {fm(a['C1'])} ({fm(a['acc1'],2)}) | {fm(a['C4'])} | {fm(a['C16'])} | {fm(a['C32out'])} | {fm(a['TTFT8K'],2)} | {fm(a['cold'],2)} | "
                f"{fm(a['knee1'],0)}/{fm(a['knee8'],0)}/{fm(a['knee16'],0)} | {fm(a['rep1'],0)}/{fm(a['rep2'],0)} | {a['gs']}/{a['gt']} | {a.get('f_kv','–')} | {a['boot']} | "
                f"{hh} (Loaded {a.get('f_nconf','–')}, saved-lines {a.get('f_saved','–')}) |")
    mem.append(f"| {t} | {a.get('pin_maps','–')} · {fm(a.get('pin'),2)} | {fm(a.get('cg_shmem'),2)} | {fm(a.get('cg_cur'),2)} | {fm(a.get('shmem'),2)} | "
               f"{fm(a.get('mavail'),2)} / {fm(a.get('mavail_dc'),2)} | {fm(a.get('mlocked'),2)} / {fm(a.get('unev'),2)} | "
               f"{fm(a.get('f_L9'),2)} / {fm(a.get('f_minha'),2)} / {fm(a.get('f_done_ha'),2)} | {fm(a.get('f_pinned_expert'),2)} |")

ctl = [A[t] for t in ("CTL1", "CTL2") if t in A]
def cm(k):
    v = [c[k] for c in ctl if c.get(k) is not None]
    return sum(v) / len(v) if v else None
out = []; suffix = "DONE"
cand = "S0" if "S0" in A and A["S0"].get("C1") is not None else None
bootfail = False
try:
    bootfail = any("BOOTFAIL S0" in l for l in open(f"{R}/runner.log"))
except Exception:
    pass
if bootfail:
    suffix = "FAIL"; out.append("**Verdict: FAIL — S0 boot death** (see runner.log BOOTFAIL line, facts-S0.txt)")
elif cand and ctl:
    c = A[cand]
    def d(k):
        b = cm(k); x = c.get(k)
        return (x / b - 1) * 100 if (b and x) else None
    pin_ctl = cm("pin"); pin_c = c.get("pin"); freed = (pin_ctl - pin_c) if (pin_ctl is not None and pin_c is not None) else None
    pb = PAR[("CTL1", "CTL2")]; p1 = PAR[("S0", "CTL1")]; p2 = PAR[("S0", "CTL2")]
    par_ok = bool(p1 and p2 and p1[0] == p1[1] and p2[0] == p2[1])
    ctl_noise = bool(pb and pb[0] != pb[1])
    bars = {
        "host pinned down >= 40 GiB": (freed is not None and freed >= 40, f"freed {fm(freed,2)} GiB (CTL mean {fm(pin_ctl,2)} -> S0 {fm(pin_c,2)})"),
        "RT C1 >= -2%": (d("C1") is not None and d("C1") >= -2, f"{fm(d('C1'))}%"),
        "replay r2 >= -2%": (d("rep2") is not None and d("rep2") >= -2, f"{fm(d('rep2'))}%"),
        "gauntlet 12/12": (c.get("gs") == 12 and c.get("gt") == 12, f"{c.get('gs')}/{c.get('gt')}"),
        "non-tool greedy parity vs both CTL": (par_ok, f"S0~CTL1 {p1[0] if p1 else '–'}/{p1[1] if p1 else '–'}, S0~CTL2 {p2[0] if p2 else '–'}/{p2[1] if p2 else '–'} (CTL1~CTL2 {pb[0] if pb else '–'}/{pb[1] if pb else '–'}; all-prompt S0~CTL1 {p1[2] if p1 else '–'}/{p1[3] if p1 else '–'}, S0~CTL2 {p2[2] if p2 else '–'}/{p2[3] if p2 else '–'}, CTL1~CTL2 {pb[2] if pb else '–'}/{pb[3] if pb else '–'})"),
        "KV tokens unchanged": (c.get("f_kv") == KV_REF, f"{c.get('f_kv')} vs {KV_REF}"),
    }
    dec = {k: d(k) for k in ("C1", "C4", "C16", "rep2")}
    fail_dec = any(x is not None and x < -3 for x in dec.values())
    fail_par = (not par_ok) and not ctl_noise and p1 is not None and p2 is not None
    win = all(v[0] for v in bars.values())
    verdict = "FAIL" if (fail_dec or fail_par) else ("WIN" if win else "NEUTRAL")
    h = c.get("f_hash"); prov = (h != REF_HASH)
    suffix = verdict + ("-PROV" if prov else "")
    out.append(f"**Candidate arm:** S0  ·  **Verdict: {verdict}**{' (PROVISIONAL — autotune hash not the REF hash)' if prov else ' (autotune hash hit ' + REF_HASH[:8] + ')'}")
    for k, v in bars.items():
        out.append(f"- WIN bar {k}: {'PASS' if v[0] else 'MISS'} — {v[1]}")
    out.append(f"- FAIL bar decode −3% (Δ C1/C4/C16/replay r2 = {', '.join(fm(x) for x in dec.values())} %): {'HIT' if fail_dec else 'clear'}; "
               f"non-tool parity mismatch: {'HIT' if fail_par else ('instrument noise (CTL1≠CTL2)' if (not par_ok and ctl_noise) else 'clear')}; boot death: clear")
    out.append(f"- other Δ vs CTL mean: C32 out {fm(d('C32out'))}%, TTFT 8K {fm(d('TTFT8K'))}%, cold 64K {fm(d('cold'))}%, knee C1 {fm(d('knee1'))}%, knee C8 {fm(d('knee8'))}%")
    out.append(f"- host: MemAvailable steady Δ {fm((c.get('mavail') or 0) - (cm('mavail') or 0),2)} GiB; after drop_caches Δ {fm((c.get('mavail_dc') or 0) - (cm('mavail_dc') or 0),2)} GiB; "
               f"hook guard L9 host_avail Δ {fm((c.get('f_L9') or 0) - (cm('f_L9') or 0),2)} GiB; cgroup shmem Δ {fm((c.get('cg_shmem') or 0) - (cm('cg_shmem') or 0),2)} GiB")
    out.append(f"HOST_FREED_GIB {fm(freed,2)}")
else:
    out.append("**No measured candidate arm** — see runner.log")
print(f"SUFFIX {suffix}")
print("### Performance\n" + "\n".join(perf) + "\n\n### Host memory (steady state, after all benches)\n" + "\n".join(mem) + "\n")
print("\n".join(out))
