#!/usr/bin/env python3
"""seed_structure.py — does vLLM v0.31.0's Gumbel sampler stay calibrated for small, sequential request seeds?
Uses sampler_replay's bit-exact function (verified 2000/2000 against the live server). (Milo, 2026-10-31)"""
import json, math, sys
import numpy as np
sys.argv = [sys.argv[0], "run1/p4ns-T1.0a.jsonl", "run1/promptids.jsonl", "1"]
src = open("sampler_replay.py").read().split("# Step 2")[0]
exec(src)   # defines gumbel(), rows, prompts, best (pos convention)
posf = best[1]
def calib(seed_fn, S, label):
    obs = exp = var = 0.0; n = 0
    for i, r in enumerate(rows):
        ids = np.array([int(k) for k in r["tops"]], dtype=np.uint64)
        lp = np.array(list(r["tops"].values()), dtype=np.float32)
        p = np.exp(lp.astype(np.float64)); psub = p.max() / p.sum(); top = np.argmax(lp)
        seeds = seed_fn(i, S); pos = posf(r)
        g = gumbel(np.repeat(seeds, len(ids)), np.full(len(seeds) * len(ids), pos), np.tile(ids, len(seeds))).reshape(len(seeds), len(ids))
        win = np.argmax(lp[None, :] + g, axis=1)
        obs += (win == top).sum(); exp += psub * len(seeds); var += psub * (1 - psub) * len(seeds); n += len(seeds)
    out = {"label": label, "draws": n, "obs": round(obs / n, 5), "exp": round(exp / n, 5),
           "diff_pp": round((obs - exp) / n * 100, 3), "z": round((obs - exp) / math.sqrt(var), 2)}
    print(json.dumps(out)); return out
rng = np.random.default_rng(2)
res = [
    calib(lambda i, S: rng.integers(0, 2**62, S), 200, "random 62-bit seeds"),
    calib(lambda i, S: np.arange(S, dtype=np.int64) + 50000, 200, "seeds 50000..50199 (same set every position)"),
    calib(lambda i, S: np.arange(S, dtype=np.int64) + 100, 200, "seeds 100..299"),
    calib(lambda i, S: rng.integers(0, 2**31, S), 200, "random 31-bit seeds"),
    calib(lambda i, S: (np.arange(S, dtype=np.int64) + 50000) << 32, 200, "sequential in the HIGH word only"),
]
json.dump(res, open("run1/seed_structure.json", "w"), indent=1)
