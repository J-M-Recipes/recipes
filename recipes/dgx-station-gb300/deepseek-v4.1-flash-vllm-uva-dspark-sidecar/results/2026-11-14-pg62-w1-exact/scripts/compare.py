#!/usr/bin/env python3
"""Greedy oracle: compare an arm's w0 parity captures against a reference capture dir. Stdlib only.

usage: compare.py <ref_dir> <arm_dir> [reps]
For each rep rN.jsonl present in both: per tag, exact text match, first differing token index, and the largest
selected-token |dlogprob| over the identical prefix. Writes <arm_dir>/verdict.json and prints one VERDICT line.
"""
import json, pathlib, sys

ref_dir, arm_dir = map(pathlib.Path, sys.argv[1:3])
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 2


def load(p):
    return {r["tag"]: r for r in map(json.loads, p.read_text().splitlines())}


def cmp_rows(a, b):
    ta = [(t["token"], t["logprob"]) for t in a["logprobs"]]
    tb = [(t["token"], t["logprob"]) for t in b["logprobs"]]
    first = None
    maxd = 0.0
    for i, (x, y) in enumerate(zip(ta, tb)):
        if x[0] != y[0]:
            first = i
            break
        maxd = max(maxd, abs(x[1] - y[1]))
    if first is None and len(ta) != len(tb):
        first = min(len(ta), len(tb))
    return {"text_equal": a["text"] == b["text"], "first_diff_token": first, "max_prefix_dlogprob": maxd,
            "n_ref": len(ta), "n_arm": len(tb)}


out = {"ref": str(ref_dir), "arm": str(arm_dir), "reps": {}}
line = []
for rep in range(reps):
    rp, ap = ref_dir / f"r{rep}.jsonl", arm_dir / f"r{rep}.jsonl"
    if not (rp.exists() and ap.exists()):
        continue
    ref, arm = load(rp), load(ap)
    per = {tag: cmp_rows(ref[tag], arm[tag]) for tag in ref if tag in arm}
    match = sum(v["text_equal"] for v in per.values())
    exact_lp = sum(v["text_equal"] and v["max_prefix_dlogprob"] == 0.0 for v in per.values())
    bad = sorted(t for t, v in per.items() if not v["text_equal"])
    toks = sum(r["n_tokens"] for r in arm.values())
    wall = sum(r["wall_s"] for r in arm.values())
    out["reps"][f"r{rep}"] = {"text_match": match, "of": len(per), "logprob_exact": exact_lp, "mismatch_tags": bad,
                              "rough_tok_s_incl_ttft": round(toks / wall, 1), "per_tag": per}
    line.append(f"r{rep} text {match}/{len(per)} lp-exact {exact_lp}/{len(per)} bad={','.join(bad) or '-'} "
                f"~{toks / wall:.0f} tok/s")
# within-arm r0 vs r1
if (arm_dir / "r0.jsonl").exists() and (arm_dir / "r1.jsonl").exists():
    a0, a1 = load(arm_dir / "r0.jsonl"), load(arm_dir / "r1.jsonl")
    per = {tag: cmp_rows(a0[tag], a1[tag]) for tag in a0 if tag in a1}
    out["self_r0_r1"] = {"text_match": sum(v["text_equal"] for v in per.values()), "of": len(per),
                         "mismatch_tags": sorted(t for t, v in per.items() if not v["text_equal"])}
    line.append(f"self r0/r1 {out['self_r0_r1']['text_match']}/{len(per)}")
(arm_dir / "verdict.json").write_text(json.dumps(out, indent=2) + "\n")
print("VERDICT " + " | ".join(line))
