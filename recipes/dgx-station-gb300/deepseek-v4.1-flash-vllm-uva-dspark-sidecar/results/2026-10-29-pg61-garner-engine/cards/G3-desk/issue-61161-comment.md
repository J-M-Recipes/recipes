One more, related to sampling rather than memory safety (checked by diffing files, plus a GPU test running now):

**vLLM #61161 ("Avoid reusing random draws across block verification rounds") is also absent from v0.31.0.** `sample/gumbel.py` (sha1 `70ae04f2`), `spec_decode/rejection_sampler_utils.py` (`fb99972e`) and `dflash2/speculator.py` (`7a2333ce`) are byte-identical to the pre-fix files. With `rejection_sample_method: block` + `draft_sample_method: probabilistic`, positions after the accepted prefix are re-drafted and re-verified next round with the same `(seed, pos)` key, so they reuse the exact draw that was just rejected. Upstream measured a biased output marginal (block: P = 0.417 vs 0.500 expected; probabilistic + block: 0.356 vs 0.300). This only matters at T > 0 — but the recipe's `--override-generation-config` makes T = 1.0 the default.

The fused draft samplers reimplement the unmixed draw too (`w5_markov.py` / `w5_markov_gemv.py` say "no `spec_step_seed`"; `w6mk.cu` loads `a.seeds[st]` directly), so overlaying upstream's files alone would leave the fast path unfixed. The port we're testing is the #61161 `gumbel.py` + `rejection_sampler_utils.py` (with the #60945 guard) in the 58132 overlay, plus the same one-line mix in each fused kernel, where the q-cache column is the step:

```c
// w6mk.cu, per-(req, step) parameter load; i = m * S + s
p_seed[i] = st >= 0 ? (a.seeds[st] ^ (static_cast<long long>(i % a.S) << 32)) : 0;
```
```python
# w5_markov.py / w5_markov_gemv.py, after the seed load
seed = seed ^ (tl.cast(step, tl.int64) << 32)
```

Step 0 is unchanged, and T = 0 never reads the seed. I'll follow up with the GPU exactness result (fused vs the eager #61161 draw).
