# Cards G1b–G1f — the C32 crash, its cause, and Garner-under-our-policy with the fix (2026-10-29)

## Verdict

1. **Cause of G1's C32 illegal address: vLLM #60945, unguarded in stock v0.31.0.** The rejection sampler's
   `_compute_local_residual_mass_kernel` reads `draft_sampled_ptr + logit_idx + 1` with no bound; with block
   verification and k=3 at 9–32 seqs, a full 32-request step is 32 × 4 = **128 logits × 4 B = exactly one 512 B
   allocator block**, so the read lands in whatever the caching allocator put next to it. Whether that faults depends
   on layout — which is why turning off `lowb_aux` (a component that does not even run at T=128) flipped it.
   Our production sidecar lane has carried the guard since the AP2 round; his release does not overlay that file.
2. **Fix verified causally (V1):** the exact config that died in G1e + the 3-line guard (sha1 `4c6250af`) and nothing
   else → 3 × 60 s C32 windows, server up.
3. **Garner's tree under our policy + the guard (P′ = "Pg") passes every cell G1 owed** and beats h2-REF everywhere
   it was measured. Not a promotion: BFCL held-out, GSM8K-200 paired, replay turns, T3, soak and a fail-closed sidecar
   watchdog for peer3 are still owed.

## The bisect

| card | arm | config | C32 ×3 |
|---|---|---|---|
| G0 | G | his release as shipped (512K) | survived ×1 (only one window run) |
| G1 | P | our policy (fp8 KV, 1M, 48 GB host KV, no overrides, minus 5 components) | **died** |
| G1 | K | P + nvfp4 KV + his indexer config | **died** |
| G1b | X1 | his release @1M | survived |
| G1b | X2 | P @512K | **died** |
| G1c | Y1 | X1 + P's component list | **died** |
| G1c | Y2 | X1 + P's non-component knobs (48 GB, no effort default) | survived |
| G1d | Z1 | X1 − {w8oproj, lowb_aux, tail_drafthead_mxfp8} | **died** |
| G1d | Z2 | X1 − {ds41_sampling, ds41_stop} | survived |
| G1e | W1 | X1 − tail_drafthead_mxfp8 | survived |
| G1e | W2 | X1 − w8oproj | survived |
| G1e | W3 | X1 − lowb_aux | **died** |
| G1f | V1 | W3 + #60945 guard | **survived** |
| G1f | Pg | P + #60945 guard | **survived** (all cells + extra C32) |

Every death: `torch.AcceleratorError: CUDA error: an illegal memory access`, scheduler dump 32 running reqs ×
4 tokens = 128, KV usage < 1 %. `lowb_aux`'s row limits are 12–24 (`fastpath_lowb_aux.py:33-36`), so it is inert at
T=128; its absence only perturbs allocation. "Survived" is a 3-window bar, not a proof of absence — the guard is the
fix because it removes the read, not because the windows were clean.

## Pg vs h2-REF (controls A, A2 from Card G1, same day, same instruments, controls within 1.2 % on steps/s)

| cell | A | A2 | **Pg** | Pg / ctrl |
|---|--:|--:|--:|--:|
| fixed coding prompt tok/s (his bench, greedy) | 446.9 | 447.4 | **632.5** | 1.41× |
| steps/s (ms/step) | 94.7 | 95.8 | **119.4 (8.38 ms)** | 1.25× |
| tokens/step, fixed | 4.71 | 4.69 | 5.28 | 1.12× |
| GSM8K T=1.0 thinking, C1 | 334.2 | 335.7 | **450.3** | 1.34× |
| GSM8K greedy, C1 | 397.4 | 400.9 | **546.7** | 1.37× |
| 8 streams aggregate | 1,120 | 1,124 | **1,393** | 1.24× |
| 32 streams aggregate | 1,490 | 1,463 | **2,586 / 2,601** | 1.75× |
| per-class C1, temp 1.0 (ours, 2 reps) | | | prose 1.30 · structured 1.35 · code 1.40 · shell 1.36 · tool_json 1.24 | **mean 1.33, min 1.24** |
| per-class C8, temp 1.0 (ours, 2 reps) | | | prose 1.10 · structured 1.12 · code 1.10 · shell 1.11 · tool_json 1.08 | **mean 1.10, min 1.08** |
| GPU KV tokens at 1M | 4,400,013 | | **11,696,582** | 2.66× |
| 1M needle (999,034 tok) | 33.2 s published | | **PASS**, TTFT 36.54 s (27,340 tok/s), post-probe OK, depth decode cache hit | TTFT 1.10× slower |

## Quality cells

| gate | control (A vs A2) | Pg vs A | Pg vs A2 | bar | result |
|---|---|---|---|---|---|
| TF ref-confident flips, agent+tool | **1** | 0 | 0 | 0 | PASS (the control itself fails this bar) |
| TF ref-confident prose flip rate | 0.003 % | 0.011 % | 0.006 % | ≤ 0.05 % | PASS |
| TF margin-weighted W per 1k | 3.021 | 2.912 | 2.699 (calib 2.636) | ≤ 1.25 × calib | PASS both |
| TF raw flips (record only) | 2.18 % | 2.27 % | 2.27 % | — | |
| perplexity over the corpus | 1.5209 → 1.5185 | → 1.5185 | 1.5185 → 1.5185 | — | |
| gauntlet (12 tool tasks) | 12/12, 12/12 | **12/12** | | 12/12 | PASS |
| 1M needle + post probe | — | **PASS** | | | PASS |

## Still owed before anything replaces h2-REF

- BFCL held-out sibling, GSM8K-200 paired, replay turns ×6, T3 promotion bench, 48 h soak.
- **A fail-closed watchdog for peer3.** Our `sidecar_watchdog.py` watches the v2 peer server; peer3's sentinel traps
  on a lost peer, but timeout behaviour (fail-open = wrong tokens) is unverified on this stack.
- 1M prefill is 10 % slower than h2-REF's published 33.2 s; prefill tok/s at 64K/128K not measured.
- The temperature > 0 seed-mix (#61161) is in our production overlay set; his tree on v0.31.0 has not been checked
  for it.
- Packaging: Pg = his `8043a681` tree + 5 no-op-unless-set hooks in `serve.sh`/`recipe.env` (`make_pol_tree.py`) +
  the guard file in a 58132 overlay copy. A recipe needs this as a reproducible launcher, with credit.

## For Garner (draft, not sent)

His release on stock `vllm/vllm-openai:v0.31.0` runs the unguarded #60945 read on a full 32-request step under his
own k-schedule (`[[1,4,5],[5,8,3],[9,32,3]]` + block verification). On our box (GB300 + RTX PRO 6000 Max-Q, his
8043a681 at 1M) it faults with an illegal address whenever the allocator layout shifts — reproduced 6 times across
4 configs; the 3-line upstream guard fixes it (3 × 60 s C32 clean on the config that died). Suggest overlaying the
guarded `rejection_sampler_utils.py` (patch attached) until a release carries #60945.
