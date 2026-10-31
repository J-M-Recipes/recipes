#!/usr/bin/env python3
"""sampler_replay.py — bit-level replay of vLLM v0.31.0's fp32 Gumbel-max sampler (vllm/v1/worker/gpu/sample/gumbel.py
@ db9527a4: murmur3_hash32 -> murmur3_uniform32 -> -log(-_log1p_neg_stable(u)) -> argmax) on the QT1 no-draft samples.
(Milo, 2026-10-31)

Step 1: reproduce the server's emitted token from the reported top-20 logprobs, the request seed and the position, for
        several (seed, pos) conventions; the one that matches ~99% (tail tokens can win ~0.4%) is the kernel's.
Step 2: with the function pinned, draw many fresh seeds per recorded distribution and compare the argmax frequency to
        the reported p_max. That is the sampling FUNCTION's calibration, with no GPU, no lane and no statistics budget.
"""
import json, math, sys
import numpy as np

M32 = np.uint64(0xFFFFFFFF)
def u32(x): return np.asarray(x, dtype=np.uint64) & M32
def rotl(v, s): v = u32(v); return u32((v << np.uint64(s)) | (v >> np.uint64(32 - s)))
def mix(h, key):
    key = u32(u32(key) * np.uint64(0xCC9E2D51)); key = rotl(key, 15); key = u32(key * np.uint64(0x1B873593))
    h = u32(h) ^ key; h = rotl(h, 13); return u32(h * np.uint64(5) + np.uint64(0xE6546B64))
def fmix(h):
    h = u32(h); h ^= h >> np.uint64(16); h = u32(h * np.uint64(0x85EBCA6B)); h ^= h >> np.uint64(13)
    h = u32(h * np.uint64(0xC2B2AE35)); return h ^ (h >> np.uint64(16))
def hash32(seed, pos, key, domain=0):
    seed = np.asarray(seed, dtype=np.int64).astype(np.uint64); pos = np.asarray(pos, dtype=np.int64).astype(np.uint64)
    h = u32(np.zeros_like(seed) ^ np.uint64(domain))
    h = mix(h, seed & M32); h = mix(h, (seed >> np.uint64(32)) & M32); h = mix(h, pos & M32); h = mix(h, key)
    return fmix(h ^ np.uint64(16))
def uniform32(seed, pos, key):
    r = hash32(seed, pos, key)
    hi = (r >> np.uint64(16)).astype(np.float32); lo = (r & np.uint64(0xFFFF)).astype(np.float32)
    return (hi * np.float32(1.52587890625e-05) + (lo + np.float32(0.5)) * np.float32(2.3283064365386963e-10)).astype(np.float32)
def log1p_neg(v):
    v = v.astype(np.float32); p = np.float32(1/8)
    for c in (1/7, 1/6, 1/5, 1/4, 1/3, 1/2, 1.0):
        p = np.float32(c) + v * p
    series = -v * p
    direct = np.log(np.maximum(np.float32(1.0) - v, np.float32(5.960464477539063e-08)))
    return np.where(v < 0.25, series, direct).astype(np.float32)
def gumbel(seed, pos, key):
    return (-np.log(-log1p_neg(uniform32(seed, pos, key)))).astype(np.float32)

rows = [json.loads(l) for l in open(sys.argv[1])]
prompts = {json.loads(l)["id"]: len(json.loads(l)["ids"]) for l in open(sys.argv[2])}
def predict(r, pos):
    ids = np.array([int(k) for k in r["tops"]], dtype=np.uint64)
    lp = np.array(list(r["tops"].values()), dtype=np.float32)
    return int(ids[np.argmax(lp + gumbel(np.full(len(ids), r["seed"]), np.full(len(ids), pos), ids))])
conv = {"pos=len": lambda r: prompts[r["pid"]] + r["pos"], "pos=len-1": lambda r: prompts[r["pid"]] + r["pos"] - 1,
        "pos=j": lambda r: r["pos"], "pos=0": lambda r: 0}
res = {}
for name, f in conv.items():
    res[name] = sum(predict(r, f(r)) == r["emitted"] for r in rows[:2000])
    print(f"{name:10s} match {res[name]}/2000")
bname = max(res, key=res.get); best = (bname, conv[bname])
print("convention:", bname)

# Step 2: calibration of the function itself on the recorded distributions (fresh seeds, same pos convention)
rng = np.random.default_rng(1); S = int(sys.argv[3]) if len(sys.argv) > 3 else 400
obs = exp = var = 0.0; n = 0
for r in rows:
    ids = np.array([int(k) for k in r["tops"]], dtype=np.uint64)
    lp = np.array(list(r["tops"].values()), dtype=np.float32)
    p = np.exp(lp.astype(np.float64)); pmax = p.max(); top = np.argmax(lp)
    seeds = rng.integers(0, 2**62, S); pos = best[1](r)
    g = gumbel(np.repeat(seeds, len(ids)), np.full(S * len(ids), pos), np.tile(ids, S)).reshape(S, len(ids))
    win = np.argmax(lp[None, :] + g, axis=1)
    # renormalise expectation to the top-20 subset the replay can see (tail tokens are not in the replay)
    psub = pmax / p.sum()
    obs += (win == top).sum(); exp += psub * S; var += psub * (1 - psub) * S; n += S
print(json.dumps({"draws": n, "obs_argmax": obs / n, "exp_argmax_top20_renorm": exp / n,
                  "diff_pp": (obs - exp) / n * 100, "z": (obs - exp) / math.sqrt(var)}, indent=1))
