#!/usr/bin/env python3
"""build_tf_corpus.py — teacher-forcing corpus for tf_logprob.py: 18 parity prompts + their v18 greedy outputs
(from the latest parity capture) plus 3 x 14k-char slices from each fund-harness 10-K. ~90k tokens. LAN IPs scrubbed."""
import json, glob, os, sys
cap = sys.argv[1] if len(sys.argv) > 1 else "e2c/t3b-parity/parity-v18ctl10.json"
docs = []
for r in json.load(open(cap))["rows"]:
    txt = (r["prompt"] + "\n\n" + (r.get("content") or "")).replace("<lan-host>", "<lan-host>")
    docs.append({"id": "parity-" + r["id"], "text": txt})
for f in sorted(glob.glob("/home/milo/iwyzer/corpus/raw/*.txt")):
    s = open(f, errors="ignore").read()
    for k in range(3):
        sl = s[k * 60000:k * 60000 + 14000]
        if len(sl) > 8000:
            docs.append({"id": os.path.basename(f)[:-4] + "-" + str(k), "text": sl})
with open("tf_corpus.jsonl", "w") as o:
    for d in docs:
        o.write(json.dumps(d) + "\n")
ch = sum(len(d["text"]) for d in docs)
print(len(docs), "docs; chars", ch, "~tok", int(ch / 3.9))
