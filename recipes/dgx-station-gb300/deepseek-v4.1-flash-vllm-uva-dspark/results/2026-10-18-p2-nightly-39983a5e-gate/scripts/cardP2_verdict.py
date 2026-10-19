#!/usr/bin/env python3
"""cardP2_verdict.py — Card P2 verdict: CAND (hook on nightly 39983a5e) vs MEAN of CTL1/CTL2 (v23).
Bars pre-registered in cardP2_runner.sh header (plan §P2). Usage: cardP2_verdict.py <results-dir>
Markdown on stdout; first line 'SUFFIX <x>' (PASS-PROV / FAIL / INVALID / NOCAND). Armed 2026-10-18 from the .DRAFT."""
import json, os, re, sys

R = sys.argv[1] if len(sys.argv) > 1 else "$HOME/dsv41/1017/results-cardP2"

def j(p):
    try: return json.load(open(p))
    except Exception: return None

def txt(p):
    try: return open(p).read()
    except Exception: return ""

def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None

def rt(tag):
    o = {}
    for k in ("c1-r1", "c1-r2", "c1-r3", "c4", "c16", "c32"):
        d = j(f"{R}/rt-{tag}/{k}.json")
        if d and "summary" in d: o[k] = d["summary"]
    c1k = [k for k in ("c1-r1", "c1-r2", "c1-r3") if k in o]
    return {"C1": mean([o[k]["decode_agg"] for k in c1k]), "TTFT8K": mean([o[k]["median_ttft_s"] for k in c1k]),
            "acc1": mean([o[k].get("acceptance_len") for k in c1k]),
            "C4": o.get("c4", {}).get("decode_agg"), "C16": o.get("c16", {}).get("decode_agg"), "C32out": o.get("c32", {}).get("output_tok_s")}

def cold(tag):
    out = {}
    for l in txt(f"{R}/cold-{tag}.txt").splitlines():
        m = re.search(r"PREFILL target=(\d+) prompt_tokens=(\d+) ttft_s=\[([^\]]*)\].*mean_tok_s=(\d+)", l)
        if m:
            tt = [float(x) for x in m.group(3).split(",") if x.strip()]
            out[int(m.group(1))] = {"pt": int(m.group(2)), "ttft": mean(tt), "tps": float(m.group(4))}
    return out

def knee(path):
    d = j(path)
    if not d: return {}
    rows = d["rows"] if isinstance(d, dict) and "rows" in d else d
    o = {}
    for r in rows:
        try: o[int(r[0])] = float(r[1])
        except Exception: pass
    return o

def replay(tag, r):
    d = j(f"{R}/replayc-P2-{tag}-{r}-n4.json")
    return d.get("agg_tok_s") if d else None

def fixture(tag):
    d = j(f"{R}/agentfix-P2-{tag}.json") or {}
    return {k: v["accept"] * 100 for k, v in d.get("res", {}).items() if isinstance(v, dict) and v.get("accept") is not None}

def gaunt(tag):
    d = j(f"{R}/gauntlet-{tag}/summary.json")
    return (d.get("success_count"), d.get("task_count")) if d else (None, None)

def bfcl(tag):
    m = re.findall(r"all (\d+)/(\d+) = ([0-9.]+)", txt(f"{R}/bfcl-{tag}.log"))
    return float(m[-1][2]) * 100 if m else None

def gate(fn):
    t = txt(f"{R}/{fn}")
    g = [l for l in t.splitlines() if l.startswith("GATE ")]
    return g[-1] if g else None   # None = UNKNOWN, never a number

def gW(line):
    m = re.search(r" W=([0-9.]+)", line or "")
    return float(m.group(1)) if m else None

def facts(tag):
    t = txt(f"{R}/facts-{tag}.txt")
    f = {"replay": "Decoder SWA bounded replay:" in t, "replay_off": ("bounded replay is off" in t) or ("needs model runner V2" in t),
         "rehome": "rehome done layers=40" in t, "starts": t.count("Autotuning process starts"),
         "loaded": (re.findall(r"Loaded (\d+) configs", t) or [None])[-1], "saved": len(re.findall(r"Saved \d+ configs", t)),
         "preloaded": "/".join((re.findall(r"Preloaded (\d+)/(\d+)", t) or [("", "")])[-1]) or None,
         "managed": "managed autotune cache" in t.lower() or "Managed autotune cache" in t,
         "breakable": "Auto-enabling VLLM_USE_BREAKABLE" in t}
    m = re.findall(r"GPU KV cache size: ([0-9,]+)", t); f["kv"] = int(m[-1].replace(",", "")) if m else None
    ha = {int(a): float(b) for a, b in re.findall(r"rehome layer=(\d+) .*?host_avail=([0-9.]+)GiB", t)[-40:]}
    f["L9"] = ha.get(9); f["minha"] = min(ha.values()) if ha else None
    m = re.findall(r"rehome done (.*?) \(expect", t); f["rehome_line"] = m[-1] if m else None
    m = re.findall(r"TUNEWINDOW (\S+) (\S+)", t); f["tune"] = f"{m[-1][0]}->{m[-1][1]}" if m else None
    m = re.findall(r"STORE_ENTRIES (\d+)", t); f["store"] = m[-1] if m else None
    return f

def hm(tag, ph):
    o = {}
    for l in txt(f"{R}/hostmem-{tag}-{ph}.txt").splitlines():
        m = re.match(r"meminfo (\w+): ([0-9.]+) GiB", l)
        if m: o[m.group(1)] = float(m.group(2))
        m = re.match(r"pinned_devzero maps=(\d+) size_gib=([0-9.]+)", l)
        if m: o["pin"] = float(m.group(2))
    return o

def boot_s(tag):
    m = re.search(rf"boot->ready {tag} (\S+)", txt(f"{R}/runner.log"))
    return m.group(1) if m else "–"

tags = [t for t in ("CTL1", "CAND", "CTL2") if os.path.exists(f"{R}/facts-{t}.txt")]
A = {}
for t in tags:
    c = cold(t); k2 = knee(f"{R}/knee-P2-{t}-r2.json"); k6 = knee(f"{R}/knee6-P2-{t}.json"); g = gaunt(t)
    hs = hm(t, "steady")
    A[t] = dict(rt(t), cold8_ttft=c.get(8192, {}).get("ttft"), cold8_tps=c.get(8192, {}).get("tps"),
                cold64_ttft=c.get(65536, {}).get("ttft"), cold64_tps=c.get(65536, {}).get("tps"), cold64_pt=c.get(65536, {}).get("pt"),
                knee8=k2.get(8), knee6_8=k6.get(8), rep1=replay(t, "r1"), rep2=replay(t, "r2"), fx=fixture(t), g=g, bfcl=bfcl(t),
                boot=boot_s(t), pin=hs.get("pin"), mavail=hs.get("MemAvailable"), **{"f_" + a: b for a, b in facts(t).items()})

G = {"CAND_vs_CTL1": gate("flips-CAND-vsCTL1.txt"), "CAND_vs_Rref": gate("flips-CAND-vsRref.txt"),
     "CTL1_vs_Rref": gate("flips-CTL1-vsRref.txt"), "CTL2_vs_CTL1": gate("flips-CTL2-vsCTL1.txt"), "CAND_vs_CTL2": gate("flips-CAND-vsCTL2.txt")}

def fm(v, p=1):
    return "–" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))

rows = ["| arm | RT C1 (acc) | C4 | C16 | C32 out | TTFT 8K (RT C1) | cold 8K TTFT / tok/s | cold 64K TTFT / tok/s | knee r2 C8 / knee6 C8 | replay r1/r2 | BFCL dev | gauntlet | KV tok | pinned host GiB | MemAvail GiB | hook L9 host_avail | boot |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for t in tags:
    a = A[t]
    rows.append(f"| {t} | {fm(a['C1'])} ({fm(a['acc1'],2)}) | {fm(a['C4'])} | {fm(a['C16'])} | {fm(a['C32out'])} | {fm(a['TTFT8K'],3)} | "
                f"{fm(a['cold8_ttft'],2)} / {fm(a['cold8_tps'],0)} | {fm(a['cold64_ttft'],2)} / {fm(a['cold64_tps'],0)} | {fm(a['knee8'])} / {fm(a['knee6_8'])} | "
                f"{fm(a['rep1'],0)}/{fm(a['rep2'],0)} | {fm(a['bfcl'])} | {a['g'][0]}/{a['g'][1]} | {fm(a['f_kv'])} | {fm(a['pin'])} | {fm(a['mavail'])} | {fm(a['f_L9'])} | {a['boot']} |")
frows = ["| arm | fixture accept % (prose / structured / code / shell_ops / tool_json) | replay boot line | rehome | autotune: Loaded / Saved / Preloaded / starts / tune window / store entries |", "|---|---|---|---|---|"]
CLS = ("prose", "structured", "code", "shell_ops", "tool_json")
for t in tags:
    a = A[t]
    frows.append(f"| {t} | {' / '.join(fm(a['fx'].get(k)) for k in CLS)} | {'yes' if a['f_replay'] else ('OFF' if a['f_replay_off'] else 'absent')} | "
                 f"{'yes' if a['f_rehome'] else 'NO'} ({a['f_rehome_line'] or '–'}) | {fm(a['f_loaded'])} / {a['f_saved']} / {fm(a['f_preloaded'])} / {a['f_starts']} / {fm(a['f_tune'])} / {fm(a['f_store'])} |")

ctl = [A[t] for t in ("CTL1", "CTL2") if t in A]
cm = lambda k: mean([c.get(k) for c in ctl])
out = []; suffix = "NOCAND"
if "CAND" in A and A["CAND"].get("C1") is not None and ctl:
    c = A["CAND"]
    d = lambda k: (c[k] / cm(k) - 1) * 100 if (cm(k) and c.get(k) is not None) else None
    g64 = (c["cold64_tps"] / cm("cold64_tps")) if (cm("cold64_tps") and c.get("cold64_tps")) else None
    ctl_fx = {k: mean([x["fx"].get(k) for x in ctl]) for k in CLS}
    spread = {k: (abs(A["CTL1"]["fx"].get(k, 0) - A["CTL2"]["fx"].get(k, 0)) if ("CTL1" in A and "CTL2" in A) else None) for k in CLS}
    fx_d = {k: (c["fx"][k] - ctl_fx[k]) if (c["fx"].get(k) is not None and ctl_fx[k] is not None) else None for k in CLS}
    fx_ok = all(v is not None and abs(v) <= 2.0 for v in fx_d.values())
    acc_ok = c["acc1"] is not None and cm("acc1") is not None and abs(c["acc1"] - cm("acc1")) <= 0.10
    bfcl_ok = c["bfcl"] is not None and cm("bfcl") is not None and c["bfcl"] >= cm("bfcl") - 1.0
    gl = G["CAND_vs_CTL1"]
    bars = {
        "TTFT 8K <= 0.45 s": (c["TTFT8K"] is not None and c["TTFT8K"] <= 0.45, f"{fm(c['TTFT8K'],3)} s (CTL mean {fm(cm('TTFT8K'),3)})"),
        "cold 64K prefill tok/s >= +30%": (g64 is not None and g64 >= 1.30, f"x{fm(g64,3)} ({fm(c['cold64_tps'],0)} vs CTL {fm(cm('cold64_tps'),0)} tok/s; TTFT {fm(c['cold64_ttft'],2)} vs {fm(cm('cold64_ttft'),2)} s)"),
        "RT C1 >= -2%": (d("C1") is not None and d("C1") >= -2, f"{fm(d('C1'))}% ({fm(c['C1'])} vs {fm(cm('C1'))})"),
        "replay r2 >= -2%": (d("rep2") is not None and d("rep2") >= -2, f"{fm(d('rep2'))}% ({fm(c['rep2'],0)} vs {fm(cm('rep2'),0)})"),
        "flip gate (margin-weighted, ref CTL1)": (bool(gl) and gl.startswith("GATE PASS"), gl or "UNKNOWN (missing file)"),
        "gauntlet 12/12": (c["g"] == (12, 12), f"{c['g'][0]}/{c['g'][1]}"),
        "fixture accept per class within 2 pts": (fx_ok, ", ".join(f"{k} {fm(fx_d[k])} pt (CTL1-CTL2 spread {fm(spread[k])})" for k in CLS)),
        "RT C1 acceptance_len within 0.10": (acc_ok, f"{fm(c['acc1'],3)} vs {fm(cm('acc1'),3)}"),
        "BFCL dev >= CTL mean - 1 pt": (bfcl_ok, f"{fm(c['bfcl'])} vs {fm(cm('bfcl'))}"),
    }
    invalid = not (c["f_replay"] and c["f_rehome"])
    passed = all(v[0] for v in bars.values())
    verdict = "INVALID" if invalid else ("PASS" if passed else "FAIL")
    suffix = verdict + ("-PROV" if verdict == "PASS" else "")
    out.append(f"**Verdict: {verdict}**" + (" (PROVISIONAL — first boot on a new FlashInfer managed autotune store; confirm boot + BFCL held-out/GSM8K before any promotion)" if verdict == "PASS" else "")
               + f" — misses: {', '.join(k for k, v in bars.items() if not v[0]) or 'none'}")
    if invalid:
        out.append(f"- mechanism: replay line {'present' if c['f_replay'] else 'ABSENT'}, hook rehome {'present' if c['f_rehome'] else 'ABSENT'}")
    for k, v in bars.items():
        out.append(f"- {k}: {'PASS' if v[0] else 'MISS'} — {v[1]}")
    out.append(f"- record only: C4 {fm(d('C4'))}%, C16 {fm(d('C16'))}%, C32 out {fm(d('C32out'))}%, knee r2 C8 {fm(d('knee8'))}%, knee6 C8 {fm(d('knee6_8'))}% (1 rep — not gated), "
               f"cold 8K TTFT {fm(c['cold8_ttft'],2)} vs {fm(cm('cold8_ttft'),2)} s, KV {fm(c['f_kv'])} vs CTL {fm(cm('f_kv'))}")
else:
    out.append("**No measured candidate arm** — see runner.log (BOOTFAIL/FASTFAIL/TIMEOUT/DIED lines)")
out.append("- flip-gate lines: " + " | ".join(f"{k}: {v or 'UNKNOWN'}" for k, v in G.items()))
print(f"SUFFIX {suffix}")
print("\n".join(rows)); print(); print("\n".join(frows)); print(); print("\n".join(out))
