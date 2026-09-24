#!/usr/bin/env python3
"""merge_gpqa.py BASE.jsonl RERUN.jsonl OUT.json — replace BASE records by index with RERUN records; summarize.
BASE = Card E e0 GPQA-Diamond at 32K cap; RERUN = the finish=length subset at a higher cap. Fail-closed:
refuses if the rerun is missing any truncated index or golds disagree."""
import json, sys

base = {r["i"]: r for r in map(json.loads, open(sys.argv[1]))}
rer = {r["i"]: r for r in map(json.loads, open(sys.argv[2]))}
trunc = sorted(i for i, r in base.items() if r.get("finish") == "length")
missing = [i for i in trunc if i not in rer]
bad_gold = [i for i in rer if rer[i]["gold"] != base[i]["gold"]]
if len(base) != 198 or missing or bad_gold:
    print(f"MERGE_FAIL n_base={len(base)} missing={missing} gold_mismatch={bad_gold}"); sys.exit(2)
merged = dict(base); merged.update(rer)
n = len(merged); ok = sum(r["ok"] for r in merged.values())
still = sorted(i for i, r in merged.items() if r.get("finish") == "length")
errs = sorted(i for i, r in merged.items() if r.get("error"))
sub_ok = sum(rer[i]["ok"] for i in trunc)
sub_fin = [i for i in trunc if rer[i].get("finish") == "stop"]
sub_fin_ok = sum(rer[i]["ok"] for i in sub_fin)
fin = [r for r in merged.values() if r.get("finish") == "stop"]
fin_ok = sum(r["ok"] for r in fin)
p = ok / n
out = {
    "n": n, "correct": ok, "acc": round(p * 100, 2), "ci95_pp": round(196 * (p * (1 - p) / n) ** 0.5, 1),
    "cap_base": base[trunc[0]]["maxtok"] if trunc else None, "cap_rerun": next(iter(rer.values()))["maxtok"],
    "rerun_n": len(trunc), "rerun_correct": sub_ok, "rerun_finished": len(sub_fin), "rerun_finished_correct": sub_fin_ok,
    "still_truncated": still, "errors": errs,
    "finished_total": len(fin), "finished_correct": fin_ok,
    "base_acc_32k": round(sum(r["ok"] for r in base.values()) / 198 * 100, 2),
}
json.dump(out, open(sys.argv[3], "w"), indent=1)
print("MERGED " + json.dumps(out))
