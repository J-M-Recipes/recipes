#!/usr/bin/env python3
"""Compare arms. usage: table.py LABEL [LABEL...]   (labels = receipts/<label> dirs; post suite under receipts/post/)"""
import json, glob, os, statistics as st, sys
B = "~/hermes/research/dsv41-gb300-owen-transfer-20261121/cards/receipts"
def J(p):
    try: return json.load(open(p))
    except Exception: return None
def med(x): x = [v for v in x if v is not None]; return round(st.median(x), 3) if x else None
def row(label):
    r = {}; d = f"{B}/{label}"
    post = f"{B}/post/{label}" if os.path.isdir(f"{B}/post/{label}") else f"{B}/post/{label.replace('gb300-pg62-','')}"
    c1 = J(f"{d}/c1.json"); c16 = J(f"{d}/c16.json"); pf = J(f"{d}/prefill.json")
    r["C1 per-stream"] = c1 and c1["summary"].get("per_stream_tok_s")
    r["C16 agg"] = c16 and c16["summary"].get("agg_tok_s")
    if pf:
        for L, v in pf["summary"]["lengths"].items(): r[f"prefill {int(L)//1000}K tok/s"] = v.get("median_tok_s")
    nd = J(f"{d}/needle.json"); r["needle"] = nd and f'{nd["summary"]["found"]}/{nd["summary"]["planted"]} {nd["summary"]["median_s"]}s'
    ms = J(f"{d}/mixed_short.json"); r["mixed_short (to answer) s"] = ms and ms["summary"]["median_ttft_s"]
    mw = J(f"{d}/mixed_warm.json"); r["mixed_warm s"] = mw and mw["summary"]["warm_median_ttft_s"]
    at = J(f"{d}/agent_turn.json"); r["agent turn s"] = at and med([t["ttft_s"] for t in at["summary"]["turns"]])
    q = J(f"{d}/qeval.json"); r["qeval"] = q and q["summary"]["passed"]
    ts = [J(p)["ev"] for p in sorted(glob.glob(f"{post}/trace-s-*.json")) if J(p)]
    r["true TTFT behind 413K s"] = med([e.get("short_ttft") for e in ts])
    r["short decode tok/s (32t)"] = med([e.get("short_decode_tok_s") for e in ts])
    td = [J(p)["ev"] for p in sorted(glob.glob(f"{post}/trace-d-*.json")) if J(p)]
    r["decode beside 413K tok/s"] = med([e.get("short_decode_tok_s") for e in td])
    r["decode DURING prefill tok/s (400t)"] = med([e.get("decode_during_prefill_tok_s") for e in td])
    tm = [J(p)["ev"] for p in sorted(glob.glob(f"{post}/trace-m1m-*.json")) if J(p)]
    r["decode DURING 1M prefill tok/s"] = med([e.get("decode_during_prefill_tok_s") for e in tm])
    r["1M time while shared s"] = med([e.get("long_done") for e in tm])
    r["413K time while shared s"] = med([e.get("long_done") for e in ts + td])
    c32 = glob.glob(f"{post}/bench-*-c32.json") or glob.glob(os.path.expanduser(f"~/hermes/work/dsv41-5way-2026-10-26/bench-{label}-c32.json"))
    if c32:
        b = J(c32[0]); bs = [x for x in b.get("batches", []) if x.get("c") == 32]
        r["C32 agg (mean of cats)"] = round(st.mean(x["agg_tok_s"] for x in bs), 1) if bs else None
    log = open(f"{post}/../../../logs/post-{label}.log").read() if os.path.exists(f"{post}/../../../logs/post-{label}.log") else ""
    for k, pat in [("greedy same-boot", "greedy same-boot identical"), ("JSON burst", '"ok_200"'), ("DSML live", "ALL_PASS"), ("alive", "after suite")]:
        ln = [l for l in log.splitlines() if pat in l]; r[k] = ln[-1][-60:] if ln else None
    return r
labels = sys.argv[1:]; rows = {l: row(l) for l in labels}
keys = list(dict.fromkeys(k for r in rows.values() for k in r))
print("| row | " + " | ".join(labels) + " |"); print("|---|" + "---:|" * len(labels))
for k in keys: print(f"| {k} | " + " | ".join(str(rows[l].get(k)) for l in labels) + " |")
