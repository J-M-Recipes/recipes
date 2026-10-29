# 2026-10-29 · Card TR1 (desk) — the sidecar's h2-REF round, audited for what transfers to the GB300-only recipe

**Verdict: `TRANSFER-PARTIAL` — no launch change on this lane.** Of the sidecar's three h2-REF deltas, one is a
latent defect that is *already on this recipe's release* (#60945), one is a sampling-bias fix that matters here
only at temperature > 0 (#61161), and one is not portable at all (split-KV). The sidecar's earlier 48 GiB Grace
prefix-KV tier is not available on this recipe's image of record. One speed idea does transfer and is owed a card
(the hook's L2 attention/indexer trims).

Nothing was booted. `:30006` (the sidecar's pair) and this recipe's release container
(`dsv41-vllm-1017-v28-prob-BOUND-REF`, stopped-and-kept) were not touched.

## Method

Read the sidecar's published patches, then read **this lane's image of record** — `vllm/vllm-openai:deepseekv41-flash-0909`,
`sha256:00d577a6…`, engine `0.1.dev20904+g179dd0fa9` — out of the box at `$HOME/dsv41/1017/p2c-desk/O` on the reference box
(the pre-extracted trees from the 2026-10-18 rebase desk check), and apply each patch with `patch -p1 --dry-run`
against that tree. Every "applies / does not apply" claim below is the dry-run or a line number in
[`raw/0909-source-excerpts.txt`](raw/0909-source-excerpts.txt).

## Verdicts

| sidecar h2-REF item | on 0909 (this recipe's image) | exposed on this recipe's release? | action |
|---|---|---|---|
| vLLM **#60945** rejection-sampler guard (3 lines) | **applies clean, 1/1 hunks** ([receipt](raw/patch-dryrun-60945-on-0909.txt)) | **yes** — v28-prob launches the kernel; the read is ahead of the temperature guard | document; a guard card is optional (TR1a) |
| vLLM **#61161** fresh per-step draws | 5/6 hunks: `gumbel.py` and `dflash2/speculator.py` clean, only the `rejection_sampler_utils.py` *import* hunk rejects (0909 has no watermark import) ([receipt](raw/patch-dryrun-61161-on-0909.txt)) | **yes, at temperature > 0** | document; a card is owed before any temperature>0 quality claim (TR1b) |
| h1 hook L2 attention/indexer trims (`attn_share.py`) | not a patch transfer — it is written against af7f9488's MegaAttn wrapper | no (nothing changes) | **Card TR2**: the call structure exists on 0909 (`flashmla.py:201`), so the idea lands; ~+1% C1, bit-identical in principle, needs its own bracket |
| split-KV decode attention (4 splits) | **does not apply** — side extension next to af7f9488's `_flashmla_C` plus a #57711-derived overlay; 0909 has no split-KV entry point | no | record as not-portable; reopen only on a rebase |
| 48 GiB Grace prefix-KV tier (`--kv-offloading-*`) | **not available** — the OffloadingConnector is not in this image | no | record; reopen only on a rebase |
| v25m map (`rowmap-mixed-ar-v1` family) | not a mechanism transfer (that map places *computed* cold experts on the 6000; this lane *reads* 89 cold experts from Grace) | no | none; this lane already runs the same map family as its seed map |

## 1. #60945 is wider than the sidecar's story: it is on this recipe's release, not just on block

The sidecar found the out-of-bounds read under `rejection_sample_method: block` and attributed it to block
verification. On this image it is launched by a **different** condition:

```python
# vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:959 (0909)
if has_draft_logits := draft_logits is not None:
...
# :1052 — launched only for full draft distributions; one-hot drafts use a
# closed-form residual mass instead.  Block verification is NOT a precondition.
        if has_draft_logits:
            _compute_local_residual_mass_kernel[(num_logits, vocab_num_blocks)](...)
```

`draft_logits` is the **draft-logits cache**, and it exists precisely because this recipe's release sets
`draft_sample_method: probabilistic` (v28-prob's own limits entry records the cache: KV −0.56%, ~31 MB at
24 seqs × k=5 × 129,280 vocab). So **v28-prob runs this kernel every decode step**, with block verification off.

The unguarded lookahead is in the kernel head, and it sits *ahead* of the temperature check:

```python
# rejection_sampler_utils.py:426-435 (0909)
logit_idx = tl.program_id(0).to(tl.int64)
draft_step_idx = tl.load(expanded_local_pos_ptr + logit_idx)
if draft_step_idx == 0 or draft_step_idx >= num_speculative_steps:
    return
if tl.load(draft_sampled_ptr + logit_idx + 1) < 0:      # <- one past the end
    return
req_state_idx = tl.load(expanded_idx_mapping_ptr + logit_idx).to(tl.int64)
temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
if temp == 0.0:
    return
```

A temperature-0 request therefore does not shield the read: the launch is decided at boot by the draft-logits
cache, and the `temp == 0` return comes after it.

**Why it has not crashed here.** The grid x-dimension is `num_logits`, and `num_logits` on this lane is bounded by
the k-schedule `[[1,8,5],[9,24,1]]` at `--max-num-seqs 24`: at most `max(8 × 6, 24 × 2) = 48` logits, i.e. 192 B
of int32. The caching allocator rounds a 192 B request up to a 512 B block, so the read at byte 192 lands inside
the allocation and returns a stale value instead of faulting. The sidecar hit it at `32 × 4 = 128` logits = exactly
512 B, where the read falls into the next block and faults once that block ends a 2 MiB segment — the sanitizer
output in the PR shows `1 bytes after the nearest allocation … of size 512 bytes`, thread 66 of block 127.
This alignment model is the one that fits both lanes' observations; it is **not** a receipt, and Card TR1a is how
it gets one. It is also consistent with the BR3–BR6 record: eleven block arms on this lane, 0 Xid, 0 engine errors.

**What this does not do:** it does not reopen the block verdicts. The read is silent here, and the BR3–BR6 trips
were speed/parity cells on shell_ops, not crashes. The block series stays closed on this lane.

## 2. #61161: v28-prob's temperature > 0 output carries a known cross-round sampling bias

0909's cached Gumbel draw is keyed on `(request seed, position)` and nothing else:

```python
# vllm/v1/worker/gpu/sample/gumbel.py:177-183 (0909)
seed = tl.load(seeds_ptr + req_state_idx, mask=is_valid_req, other=0)
pos = tl.load(pos_ptr + token_idx)
return gumbel_noised_argmax(logits, block, mask, seed, pos, temp, ...)
```

and the DSpark drafter reaches it through the logits cache with a per-step column
(`dspark/speculator.py:140-150`: `gumbel_sample(..., logits_cache=self.draft_logits,
logits_cache_col=self._step_cols[step])`). When the target accepts only part of a block, the next round revisits
an uncommitted position with the **same** `(seed, position)` draw, so that draw is conditioned on the previous
round's decision. #61161 mixes the draft step into the seed's upper 32 bits and reports the resulting marginal
error, on a 10,000-request synthetic target where P(third token = 0) should be 0.3000:

| draft | verification | expected | before | after |
|---|--:|--:|--:|--:|
| deterministic | standard | 0.5000 | 0.5039 | 0.5039 |
| deterministic | block | 0.5000 | 0.4171 | 0.4989 |
| **probabilistic** | **standard** | 0.3000 | **0.3083** | 0.3055 |
| probabilistic | block | 0.3000 | 0.3558 | 0.2898 |

The row that matters here is the third: **probabilistic drafting with standard verification** is exactly v28-prob,
and it is the one configuration where the fix changes token sequences ("standard probabilistic output does not
retain its original hash") while the target distribution is retained. Direction of the speed result is unaffected
(fresh draws do not change the verify shape), but no temperature>0 quality claim on this lane is distribution-clean
until the fix is in. At temperature 0 nothing changes — the fix is inert by construction.

The hand-port to 0909 is small: the `gumbel.py` hunk applies clean **and is sufficient for the DSpark drafter**
(0909's drafter flows through `gumbel_block_argmax` with `logits_cache_col`, which is the hunk's `col`). The
verifier half of the patch (the `rejection_sample_method: block` draw) needs the import line hand-written, and only
matters if block is ever re-enabled.

## 3. The one speed transfer: h1's L2 attention/indexer trims

h1 is −0.117 ms/step (about +1% C1) and bit-identical on the sidecar because it stops recomputing the same thing
38 times per decode step: `compute_global_topk_indices_and_lens` is a pure gather/count over identical inputs for
every layer that shares a KV source, and `positions.to(int32)` is re-derived in every layer's `forward_mqa`.

0909 has the same shape. In `vllm/models/deepseek_v4_1/nvidia/flashmla.py:201`, `_forward_decode` calls
`compute_global_topk_indices_and_lens` **once per non-SWA-only layer**, and the layers are already grouped by
`self.compress_ratio` (0/1/2) — so the sidecar's key (kv-source, index-source, ratio) exists here too, and the
call count collapses from ~38 to 3.

It is not a patch transfer: `attn_share.py` is installed by `mega_peer_hook.install()`, which this lane does not
run, and its graph-safety argument rests on af7f9488's `@eager_break_during_capture` in `_sparse_indexer_and_attn`.
0909 captures through a different path. **Card TR2** therefore has to re-establish, on this image: (a) pointer-share
distribution on the real fixture (the sidecar needed its relaxed mode `MEGA_SHARE_TOPK=2` because
`token_to_req_indices`/`is_valid_token` are per-SWA-group buffers with equal content), (b) graph-capture lifetime
and replay safety, (c) bit-identity (TF flips at the control floor, parity), then (d) the per-class C1/C8 cells.

## Not transferable, with reasons

- **Split-KV decode attention** — built as a side extension next to af7f9488's untouched `_flashmla_C` plus a
  #57711-derived overlay; 0909's FlashMLA has no split-KV entry point, and the shipped quarantine only lets it
  engage at one request. Porting it means rebasing, which this recipe has already declined twice (P2f/P2g are
  candidates, not the runtime).
- **The 48 GiB Grace prefix-KV tier** — the OffloadingConnector (`--kv-offloading-size/--kv-offloading-backend`)
  postdates this image. This is the one sidecar win that is structurally out of reach without the rebase.
- **The v25m map** — it is a placement map for experts that are *computed* on the 6000. This lane's 89 cold experts
  are read from pinned Grace, and its seed map already comes from the same traffic-fitted family.

## What a card still has to prove

- **TR1a (guard):** apply `vllm-60945-rejection-sampler-guard.patch` as a one-file overlay on 0909; prove
  bit-identity at temperature 0 on the per-class fixture and in the teacher-forced bracket; `compute-sanitizer
  --tool memcheck` on a decode step at 24 seqs (v28-prob) shows 0 invalid reads, and the same run on the unpatched
  image shows the read. Cheap: one arm plus one instrumented pass, no speed claim needed.
- **TR1b (fresh draws):** the `gumbel.py` hunk only, bracketed on temperature-1.0 per-class cells (the fix changes
  token sequences by design, so the speed cells must be re-measured, not assumed) plus a seeded-marginal instrument
  to show the bias is gone. A temperature-0 arm is the parity control (must be identical).
- **TR2 (L2 sharing):** see above; the only item with a real speed upside (~+1% C1, bit-identical in principle).

## Files

- [`raw/patch-dryrun-60945-on-0909.txt`](raw/patch-dryrun-60945-on-0909.txt) — clean apply on the 0909 tree.
- [`raw/patch-dryrun-61161-on-0909.txt`](raw/patch-dryrun-61161-on-0909.txt) — 5 of 6 hunks; the import hunk rejects.
- [`raw/0909-source-excerpts.txt`](raw/0909-source-excerpts.txt) — kernel head, launch condition, cached-draw seed,
  DSpark cache call, and the per-layer topk call site, with line numbers.

Source trees read from `$HOME/dsv41/1017/p2c-desk/O` on the reference box on the Station (2026-10-29 03:50–03:56 CDT). Upstream
claims quoted from the two PR bodies, not re-run here.