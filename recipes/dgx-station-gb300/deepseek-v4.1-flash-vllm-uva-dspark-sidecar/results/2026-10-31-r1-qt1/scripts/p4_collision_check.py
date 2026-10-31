#!/usr/bin/env python3
"""p4_collision_check.py — is QT1's P4 (argmax excess) a string-collision artefact? (Milo, 2026-10-31)

QT1 keyed tokens by decoded string. A position is 'collision-suspect' if its returned top-k dict has fewer
entries than requested (k+1 incl. the sampled token, or k when sampled is inside top-k), i.e. two ids
decoded to the same string and the JSON dict merged them. Report the argmax excess split by
suspect / clean, and by whether the emitted string is the replacement char or pure whitespace.
"""
import json, math, sys, collections
files = sys.argv[1:] or ["run1/gen-T1.0a.jsonl", "run1/gen-T1.0b.jsonl"]
acc = collections.defaultdict(lambda: [0, 0.0, 0.0, 0])   # n, obs, exp, var
entries_hist = collections.Counter()
for f in files:
    for line in open(f):
        r = json.loads(line)
        for j, tops in enumerate(r["top_logprobs"]):
            if not tops:
                continue
            items = sorted(((int(k), v["logprob"]) for k, v in tops.items()), key=lambda x: -x[1])
            ne = r["n_entries"][j]; entries_hist[ne] += 1
            em = r["token_ids"][j]
            top_id, top_lp = items[0]
            p = math.exp(top_lp)
            hit = 1.0 if em == top_id else 0.0
            # was the emitted token inside the returned set? if so the server returns k entries (sampled is one of
            # them); if not, k+1. Fewer than that = a merged duplicate string.
            inside = any(i == em for i, _ in items)
            expected_n = 20 if inside else 21
            suspect = ne < expected_n
            q = "Q1" if p < 0.75 else "Q2+"
            for key in (("all",), ("suspect" if suspect else "clean",), ("suspect" if suspect else "clean", q)):
                a = acc[key]; a[0] += 1; a[1] += hit; a[2] += p; a[3] += p * (1 - p)
print("n_entries histogram:", dict(sorted(entries_hist.items())))
for k in sorted(acc, key=str):
    n, o, e, v = acc[k]
    print(f"{'/'.join(k):16s} n={n:6d} obs={o/n:.4f} exp={e/n:.4f} diff={(o-e)/n*100:+.2f}pp z={(o-e)/math.sqrt(v) if v else 0:+.2f}")
