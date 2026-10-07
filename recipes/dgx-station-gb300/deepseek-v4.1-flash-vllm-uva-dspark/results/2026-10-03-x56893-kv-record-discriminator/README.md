# 2026-10-03 — vllm#58391 discriminator: the r4→r5 teacher-forced shift is #56893 (V4.1 MXFP8 KV record)

Method (hclsys, vllm#58391): same Card E r5 rung (`vllm/vllm-openai:nightly-cd10ed6f`, v15 hook, off54, `fp8_ds_mla`, seqs 24) with one change —
`_use_v41_mxfp8_kv_record()` forced to `False` via a read-only bind-mount of the image's own `attention.py` (diff + sha256 in receipts),
so the GB300 (SM103, family 100) uses the pre-#56893 584 B record (RoPE dims bf16) instead of the 528 B all-fp8 record.
Same teacher-forced corpus (39 docs, 79,544 positions, `tf_logprob.py`, MAXPOS 4096) and same v14 no-hook reference as Card E.

| vs v14nohook | r4 dc36fcce | r5 cd10ed6f | r5 + gate=False (this run) |
|---|---|---|---|
| top-1 flips | 1.038% | 2.234% | **1.085%** |
| mean abs Δlogp | 0.0280 | 0.0601 | **0.0290** |
| agent_tool flips (1074 pos) | 1.77% | 12.66% | **3.72%** |
| heldout flips (2868 pos) | 0.00% | 9.00% | **0.00%** |
| prose flips (75602 pos) | 1.07% | 1.83% | **1.09%** |

r4 → r5x directly: flips 1.103%, mean abs Δ 0.0290, ppl 1.5075 → 1.5075.
Receipts that the record actually changed: in-container `_use_v41_mxfp8_kv_record()` returned `False`; GPU KV cache 2,647,958 tokens
(r5: 2,878,193; r4: 2,547,505). Autotune hash hit (189 configs loaded), boot 530 s, smoke prose/tool/think OK. Candidate stopped-and-kept.
