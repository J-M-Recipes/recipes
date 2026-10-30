Follow-up with the GPU result. Test: synthetic inputs (V = 129,280, rank 256, S = 5, T = 1). The reference is the eager #61161 `gumbel_sample` (with `spec_step_seed`) applied to the q-cache row each kernel wrote, `logits_cache_col = s`.

| kernel | with the one-line mix, matches per step | as shipped, matches per step |
|---|---|---|
| `libw6mk.so` (w6mk.cu, R = 1, 30 trials) | 30 / 30 / 30 / 30 / 30 | 30 / 0 / 0 / 0 / 0 |
| `w5_markov` Triton (R = 4, 30 trials) | 120 / 120 / 120 / 120 / 120 | 120 / 0 / 1 / 0 / 0 |

T = 0 token streams are identical with and without the change. End to end on the Station (fp8 KV, 1M), the patched build ran at the same step rate as unpatched (119.3 vs 119.4 steps/s).

Separately, in case it's useful: on a peer tier v3 wait timeout (`PEER3_SPINS`, default 20,000,000), the finish kernel and both folds skip that call's cold-expert adds and keep generating; the count lives only in device memory. We run `PEER3_SPINS=1000000000000` so a stalled RTX hangs the GB300 instead, plus a host watchdog that kills the server when the v3 done tag stops following the published tag. With the sidecar host frozen (`docker pause`), output stopped within 0.1 s and the watchdog tripped in about 3 s.
