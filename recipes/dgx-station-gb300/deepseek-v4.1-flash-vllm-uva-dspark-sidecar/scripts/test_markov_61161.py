#!/usr/bin/env python3
"""test_markov_61161.py <fastpath_dir> <out.json> — GPU exactness test for the #61161 seed mix in Garner's fused draft
samplers (libw6mk.so via w6_markov, and w5_markov's Triton loop), against production's eager draw.

Reference per (trial, step s): production's overlay gumbel_sample (L3 gumbel.py, with spec_step_seed) on the very q-cache
row the kernel wrote, with logits_cache_col = s, is_drafting = True, pos = sample_pos - 1. A ported kernel must equal it
at every step; an unported one equals it only at step 0 (col 0 leaves the seed unchanged). At T = 0 no noise is drawn,
so stock and patched must agree everywhere (compared across the two runs by the caller).
"""
import json, sys
import torch

FP, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, FP)
import w6_markov, w5_markov  # noqa: E402
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample  # noqa: E402  (overlay: L3 gumbel.py)

torch.manual_seed(61161)
dev = torch.device("cuda")
V, RANK, S, MAXR = 129280, 256, 5, 8
TRIALS = int(sys.argv[3]) if len(sys.argv) > 3 else 40
w1 = (torch.randn(V, RANK, device=dev) * 0.05).to(torch.bfloat16).contiguous()
w2 = (torch.randn(V, RANK, device=dev) * 0.05).to(torch.bfloat16).contiguous()


def reference(cache, st, s, temp, seeds, pos):
    scratch = torch.empty(MAXR, S, V, dtype=torch.bfloat16, device=dev)
    logits = cache[st, s, :V].unsqueeze(0).contiguous()
    return int(gumbel_sample(logits, torch.tensor([st], dtype=torch.int32, device=dev), temp, seeds,
                             torch.tensor([pos], dtype=torch.int64, device=dev), apply_temperature=True,
                             is_drafting=True, logits_cache=scratch,
                             logits_cache_col=torch.tensor([s], dtype=torch.int32, device=dev))[0])


def inputs(R, T):
    base = (torch.randn(R, S, V, device=dev) * 2.0).to(torch.bfloat16)
    input_ids = torch.randint(0, V, (64,), dtype=torch.int32, device=dev)
    anchor = torch.randint(0, 64, (R,), dtype=torch.int64, device=dev)
    sts = torch.randperm(MAXR, device=dev)[:R].to(torch.int32)
    idx_map = sts.repeat_interleave(S).contiguous()
    p0 = torch.randint(100, 900000, (R,), dtype=torch.int64, device=dev)
    sample_pos = (p0[:, None] + 1 + torch.arange(S, device=dev)[None, :]).reshape(-1).contiguous()
    temp = torch.full((MAXR,), T, dtype=torch.float32, device=dev)
    seeds = torch.randint(-(2 ** 62), 2 ** 62, (MAXR,), dtype=torch.int64, device=dev)
    return base, input_ids, anchor, idx_map, sample_pos, temp, seeds


res = {"fastpath": FP, "w6": {}, "w5": {}}
loop = w6_markov.FusedMarkovLoop(w1, w2, MAXR, S)
w5s = w5_markov.FusedMarkovSampler(MAXR, V, w1, False)
for T in (1.0, 0.0):
    for name in ("w6", "w5"):
        match = [0] * S
        n = [0] * S
        toks = []
        for _ in range(TRIALS):
            R = 1 if name == "w6" else 4
            base, input_ids, anchor, idx_map, sample_pos, temp, seeds = inputs(R, T)
            cache = torch.zeros(MAXR, S, V, dtype=torch.bfloat16, device=dev)
            draft = torch.full((MAXR, S), -1, dtype=torch.int64, device=dev)
            if name == "w6":
                loop.run(base, input_ids, anchor, idx_map, sample_pos, temp, seeds, cache, draft)
            else:
                embed = w5s.embed_anchor(input_ids, anchor, R)
                for s in range(S):
                    bias = embed @ w2.t()
                    w5s.sample_step(base, bias, s, idx_map, sample_pos, temp, seeds, cache, draft)
            torch.cuda.synchronize()
            for r in range(R):
                st = int(idx_map[r * S])
                for s in range(S):
                    got = int(draft[r, s])  # draft_tokens rows are batch rows 0..R-1; cache rows are req states
                    if T > 0:
                        ref = reference(cache, st, s, temp, seeds, int(sample_pos[r * S + s]) - 1)
                    else:
                        ref = int(cache[st, s, :V].float().argmax())
                    match[s] += got == ref
                    n[s] += 1
                    toks.append(got)
        res[name][f"T{T:g}"] = {"match_per_step": match, "n_per_step": n,
                                "tokens_sha": __import__("hashlib").sha1(json.dumps(toks).encode()).hexdigest()[:12]}
        print(name, f"T={T:g}", "match/step", match, "of", n, res[name][f"T{T:g}"]["tokens_sha"], flush=True)
json.dump(res, open(OUT, "w"), indent=1)
