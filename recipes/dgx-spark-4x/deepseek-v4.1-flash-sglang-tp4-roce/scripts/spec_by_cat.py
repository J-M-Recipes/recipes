#!/usr/bin/env python3
# Published copy (J-M-Recipes dgx-spark-4x/deepseek-v4.1-flash-sglang-tp4-roce): box paths -> env/BASE_URL; logic unchanged.
# NOTE: reads vllm:spec_decode_* counters; the SGLang lane has no /metrics, so this tool does not run against it
# (per-class DSpark acceptance on this recipe is pending). Shipped because both lanes share one instrument.
"""Per-category DSpark acceptance at C1 from /metrics deltas (vllm:spec_decode_*).
Uses v41bench_tf.py's prompt set v1 (same encoding path: /v1/completions, thinking off, T=0).
Each category: REPS sequential requests with a unique leading tag; /metrics read before and after.
usage: spec_by_cat.py BASE MODEL OUTDIR [reps]"""
import json, os, sys, time, re, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v41bench_tf as vb

base, model, out = sys.argv[1], sys.argv[2], sys.argv[3]
reps = int(sys.argv[4]) if len(sys.argv) > 4 else 3
root = base.rsplit("/v1", 1)[0]

def metrics():
    raw = urllib.request.urlopen(root + "/metrics", timeout=30).read().decode()
    vals = {}
    for l in raw.splitlines():
        if l.startswith("vllm:spec_decode") and "_created" not in l:
            m = re.match(r'(vllm:[a-z_]+)\{([^}]*)\} ([0-9.e+]+)', l)
            if m:
                pos = re.search(r'position="(\d+)"', m.group(2))
                key = m.group(1) + (f"[{pos.group(1)}]" if pos else "")
                vals[key] = float(m.group(3))
        if l.startswith("vllm:num_preemptions_total"):
            vals["preemptions"] = float(l.split()[-1])
    return raw, vals

rows = []
for name, prompt, mt in vb.CATEGORIES + [vb.CEILING]:
    raw0, a = metrics()
    toks = 0; dt = 0.0; decs = []
    for r in range(reps):
        p = f"[bench {vb.PROMPT_SET_VERSION} acc{r} {name} c1 s0] " + prompt
        res = vb.stream_chat(base, model, p, mt)
        toks += res["completion_tokens"]; decs.append(res["decode_tok_s"])
    raw1, b = metrics()
    open(f"{out}/metrics-acc-{name}.txt", "w").write(raw1)
    d = {k: b.get(k, 0) - a.get(k, 0) for k in b}
    drafts = d.get("vllm:spec_decode_num_drafts_total", 0)
    dtok = d.get("vllm:spec_decode_num_draft_tokens_total", 0)
    acc = d.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    row = {"category": name, "reps": reps, "completion_tokens": toks,
           "per_stream_tok_s_mean": round(sum(x for x in decs if x) / len([x for x in decs if x]), 2),
           "drafts": drafts, "draft_tokens": dtok, "accepted_tokens": acc,
           "acceptance_rate": round(acc / dtok, 4) if dtok else None,
           "mean_accepted_len": round(1 + acc / drafts, 3) if drafts else None,
           "per_pos_accept": [round(d.get(f"vllm:spec_decode_num_accepted_tokens_per_pos_total[{i}]", 0) /
                                    max(1, d.get(f"vllm:spec_decode_num_draft_tokens_per_pos_total[{i}]", 0)), 3) for i in range(5)],
           "preemptions_delta": d.get("preemptions", 0)}
    rows.append(row); print(json.dumps(row), flush=True)
json.dump(rows, open(f"{out}/spec-acceptance-C1.json", "w"), indent=1)
md = ["| category | C1 per-stream tok/s | acceptance (acc/draft tok) | mean accepted len (1+acc/step) | per-position |", "|---|---|---|---|---|"]
for r in rows:
    md.append(f"| {r['category']} | {r['per_stream_tok_s_mean']} | {r['acceptance_rate']} | {r['mean_accepted_len']} | {r['per_pos_accept']} |")
open(f"{out}/spec-acceptance-C1.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
