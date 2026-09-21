# 2026-09-21 — rebase to SGLang v0.5.20 (sgl-project/sglang#37818, DFlash/KDA checkpoint fix)

Four boots in one window on `:30001`, same weights (`nvidia/GLM-5.3-Flash-NVFP4@09b04e5e`), same draft (`incoai/GLM-5.3-Flash-DFlash2@7d74cdd8`), same flags (`--max-mamba-cache-size 48`, block 7, fp8_e4m3, 1M ctx, MAXBS 16): pinned nightly `20260911-00143e9c` and tagged **`v0.5.20-cu130`** (`sha256:06e4f2ed21af…`), each in AR (`SPEC=none`) and DFlash2. Runner `cardD-glmf-v0520-rebase-2026-09-21.sh`; new instrument `long_greedy.py` (8 prompts × 2500 tokens, T=0, effort low). Receipts in `receipts/<oldar|olddf|newar|newdf>/`.

## Gates

| gate | pinned AR | pinned DFlash2 | v0.5.20 AR | v0.5.20 DFlash2 |
|---|--:|--:|--:|--:|
| TF logprob on w2-a reference text (2,857 tokens, `tf_noninferiority.py`) vs pinned-AR self-score | self (0.0) | max Δ **0.0** | max Δ **0.0** | max Δ **0.0** |
| greedy 20×200 vs w2-a (DFlash reference) | 1/20 | **20/20** | 1/20 | **20/20** |
| tool harness (10 prompts, ×2) | 10/10 | 10/10 | 10/10 | 10/10 |
| C1 recipe-method history-essay 512 / code 400 / shell 200 (tok/s, effort low) | 138.2 / 136.7 / 104.3 | 193.3 / 276.1 / 154.7 | 138.9 / 137.3 / 104.1 | **201.9 / 288.2 / 162.3** |
| accept (`accept_probe`, mean len / rate; agentic) | — | 3.47 / 0.41; 3.37 / 0.39 | — | 3.37 / 0.39; 3.34 / 0.39 |
| C8 agg (3 reps) | 668 (3%) | 507 — one 134 rep, spread 112%; reps 684/704 | 667 (3%) | **699 (6%)** |
| long greedy 8×2500, DFlash vs AR *same image* | — | 0/8 identical, first divergence char 27–362 | — | 0/8, **identical offsets** (27–362) |
| long greedy, pinned DFlash vs v0.5.20 DFlash | | 1/8 identical; divergence at char 5,694–9,337 (≈ tokens 1,500–2,500) | | |
| long greedy, pinned AR vs v0.5.20 AR | 0/8; divergence at char 4,705–9,193 | | | |

## Reading

**v0.5.20 passes every gate the pinned image passes, and is faster with DFlash2: C1 +4–5% on every prompt class (history 193→202, code 276→288, shell 155→162), C8 ~700 vs ~690 (pinned's 507 mean carries a 134 tok/s outlier rep).** Acceptance is flat (0.41 → 0.39 mean rate). TF prefill logprobs are bit-identical across all four servers — the model function did not change between images; the differences are decode-path numerics.

**The #37818 signature is not observable with this instrument.** The bug (DFlash missing a KDA state checkpoint when accepted tokens cross a 64/128/256 tracking boundary) would show as DFlash text diverging from AR text *late* on the pinned image and *not* on v0.5.20. What we see instead: (a) DFlash and AR diverge within the first 30–360 characters on **both** images at byte-identical offsets — that is DFlash2's verify path picking a different argmax where the target's top-2 logits are near-tied (fa4 draft attention, batched verify), present before and after the fix and unrelated to KDA checkpoints; (b) old-vs-new *DFlash* outputs stay identical for 1,500–2,500 tokens and then diverge at about the same depth old-vs-new *AR* outputs do (4,700–9,300 chars), which is image-numerics drift, not a DFlash-specific effect. So the fix is in the image, the image is not worse on anything measured, and we cannot show the bug it fixes with 8 prompts — a boundary-crossing reproducer needs controlled accepted-token runs, which is what upstream's regression test does.

**A correction this bundle forces:** the recipe has said DFlash2 is "lossless at T=0" because its greedy gate compared DFlash boots to a DFlash reference (w2-a). DFlash2 vs **AR** on the same image is **not** token-identical on GLM-5.3-Flash — 1/20 on 200-token prompts, 0/8 on 2,500-token prompts, first divergence typically inside the first 100 tokens. Teacher-forced logprobs of the *same* text are identical (Δ 0.0), so this is argmax tie-breaking in the verify kernel, not a quality change; but "target-verified, every emitted token accepted by the full model" is the correct claim and "byte-identical to autoregressive" is not. `limits` updated.

## Decision

Recipe image pin moves to `lmsysorg/sglang:v0.5.20-cu130` (`sha256:06e4f2ed21afde4ff513cda65070124e727ba23ccaeff7712b8c40e1097d611f`, SGLang `0.5.20`, released 2026-09-18). Old row retained as `— nightly 00143e9c baseline`. Draft stays `7d74cdd8`; the newer `bf582e4e` draft remains an untested axis. `:30001` is not a serving lane today; the pin is the recipe contract.
