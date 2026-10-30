Thanks for publishing this. We ran 8043a681 on a DGX Station GB300 + RTX PRO 6000 **Max-Q (300 W)** and it reproduces: 135.9 steps/s on the fixed prompt (yours: 135.4), 731 tok/s. 1M context costs nothing (135.9 steps/s at `DS41_MAXLEN=1048576`).

**One latent bug worth guarding.** Stock `vllm/vllm-openai:v0.31.0` ships `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py` without the vLLM #60945 guard, and the release overlays (1017, 58132) don't touch that file. `_compute_local_residual_mass_kernel` reads `draft_sampled_ptr + logit_idx + 1` with no bound. With block verification and the recipe's k-schedule (`[9,32,3]`), a full 32-request decode step is 32 × 4 = 128 logits × 4 B — exactly one 512 B allocator block — so the read lands in whatever the caching allocator placed next to `draft_sampled`.

On your config as shipped we saw **no fault** in 4 × 60 s C32 windows (`bench_full.py --conc 32`). It faulted (`torch.AcceleratorError: CUDA error: an illegal memory access`, scheduler dump: 32 running requests × 4 scheduled tokens = 128) **6 times on variants that only shift allocation**. The cleanest one: your release at 1M with `lowb_aux` removed from `DS41_FASTPATH` — a component that is inert at T=128 (its row limits are ≤ 24), so removing it changes allocation, not math. Adding the 3-line guard to that exact config gave 3 × 60 s C32 windows clean.

Suggested fix until a release carries #60945: overlay the guarded file alongside the 58132 overlay.

```diff
--- a/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py
+++ b/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py
@@ -425,6 +425,9 @@
     PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
 ):
     logit_idx = tl.program_id(0).to(tl.int64)
+    # Adaptive verification can end before num_speculative_steps.
+    if logit_idx + 1 >= tl.num_programs(0):
+        return
     draft_step_idx = tl.load(expanded_local_pos_ptr + logit_idx)
     if draft_step_idx == 0 or draft_step_idx >= num_speculative_steps:
```

Guarded file sha1 on v0.31.0: `4c6250af032ac3dda8a5755dfa9e395e7bfc1ae3` (stock: `fb99972e662e8d8187282e3cb951cee874a907a9`).

Box: driver 595.99.02, `vllm/vllm-openai:v0.31.0` (`sha256:c1c9f6fd5c10…`), checkpoint DeepSeek-V4.1-Flash-1017, your tree at 8043a681 with `DS41_PORT` changed only.
