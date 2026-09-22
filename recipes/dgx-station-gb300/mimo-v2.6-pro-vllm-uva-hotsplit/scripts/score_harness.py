#!/usr/bin/env python3
"""Re-score harness_test.sh logs. Extracts the final Hermes answer box (the block between the
'☤ Hermes' / 'Hermes' header rule and the next closing rule) from each runN.log, then applies the same
ground-truth checks as harness_test.sh. Written because the shell awk extractor in harness_test.sh
matched '⚕ Hermes' (older CLI) and silently read an empty answer on the current CLI ('☤ Hermes').
usage: python3 score_harness.py /tmp/harness-mimo26"""
import os, re, sys, glob
D = sys.argv[1]
T = {0: "5", 1: "MAGIC-7731", 2: "30007", 3: "Darwin", 4: "charlie", 5: ("file", "/tmp/hx/out1.txt", "HELLO"),
     6: ("count_txt",), 7: "7006652", 8: ("all", ["echo", "gb300"]), 9: ("file_lines", "/tmp/hx/words.txt", 6)}
def answer(log):
    s = open(log, errors="replace").read().replace("\r", "")
    lines = s.split("\n")
    idx = [i for i, l in enumerate(lines) if re.search(r"(☤|⚕)\s*Hermes", l)]
    if not idx:
        return ""
    out = []
    for l in lines[idx[-1] + 1:]:
        if re.match(r"^\s*─{10,}\s*$", l):
            break
        out.append(l.strip())
    return "\n".join(x for x in out if x)
res = []
for i in range(10):
    log = os.path.join(D, f"run{i}.log")
    a = answer(log)
    calls = re.findall(r"tool_turns=(\d+)", open(log, errors="replace").read())
    c = int(calls[-1]) if calls else 0
    t = T[i]
    if isinstance(t, str):
        ok = t in a
    elif t[0] == "file":
        # harness_test.sh deletes this file at the start of every loop, so it cannot be checked after the run.
        # Use harness_test.sh's own live side-effect verdict (LIVE_OK env, e.g. LIVE_OK=5) plus the answer text.
        ok = str(i) in os.environ.get("LIVE_OK", "").split(",") and t[2] in a
    elif t[0] == "count_txt":
        k = len(glob.glob("/tmp/hx/*.txt")); ok = re.search(rf"\b{k}\b", a) is not None
    elif t[0] == "all":
        ok = all(w in a for w in t[1])
    elif t[0] == "file_lines":
        n = sum(1 for _ in open(t[1])); ok = n == t[2] and re.search(rf"\b{t[2]}\b", a) is not None
    else:
        ok = False
    res.append((i, c, ok, a.replace("\n", " ")[:70]))
    print(f"[{i}] tools={c} ok={int(ok)}  answer={a.replace(chr(10), ' ')[:70]!r}")
print(f"SUMMARY tool_call_turns={sum(1 for r in res if r[1] > 0)}/10 correct={sum(r[2] for r in res)}/10")
