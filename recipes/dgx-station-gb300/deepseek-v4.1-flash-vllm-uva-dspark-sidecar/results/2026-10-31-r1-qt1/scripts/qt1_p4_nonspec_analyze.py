#!/usr/bin/env python3
"""qt1_p4_nonspec_analyze.py — P4 statistic on the no-draft replay samples, side by side with the speculative
samples at the SAME (prompt, position) set. (Milo, 2026-10-31)"""
import json, math, sys, collections
ns_file = sys.argv[1] if len(sys.argv) > 1 else "run1/p4ns-T1.0a.jsonl"
gen_file = sys.argv[2] if len(sys.argv) > 2 else "run1/gen-T1.0a.jsonl"

def stat(rows):
    n = len(rows); o = sum(h for h, p in rows); e = sum(p for h, p in rows); v = sum(p * (1 - p) for h, p in rows)
    return {"n": n, "obs": o / n, "exp": e / n, "diff_pp": (o - e) / n * 100, "z": (o - e) / math.sqrt(v)}

ns, by_cls, pos = [], collections.defaultdict(list), set()
for l in open(ns_file):
    r = json.loads(l)
    t = r["tops"]; top = max(t, key=t.get); p = math.exp(t[top])
    h = 1.0 if str(r["emitted"]) == top else 0.0
    ns.append((h, p)); by_cls[r["cls"]].append((h, p)); pos.add((r["pid"], r["pos"]))
# speculative samples at the same positions (QT1 gen; string-interned identity, collision-checked clean above)
spec = []
for l in open(gen_file):
    r = json.loads(l)
    for j, tops in enumerate(r["top_logprobs"]):
        if (r["prompt_id"], j) in pos and tops:
            items = sorted(((int(k), v["logprob"]) for k, v in tops.items()), key=lambda x: -x[1])
            spec.append((1.0 if r["token_ids"][j] == items[0][0] else 0.0, math.exp(items[0][1])))
out = {"no_draft_replay": stat(ns), "speculative_same_positions": stat(spec),
       "no_draft_by_class": {c: stat(v) for c, v in sorted(by_cls.items())}}
print(json.dumps(out, indent=1))
