# Failure ledger — every wall between "hardware works" and "first token"

Recorded verbatim so you can grep your error into ours. Same order we hit them.

| # | Attempt | Failure |
|---|---|---|
| 0 | `docker pull lmsysorg/sglang:blackwell` | tag does not exist (guessed) |
| 1 | pinned v0.5.16 digest + `CUDA_VISIBLE_DEVICES=1` | container saw 0 GPUs — DGX OS docker config already hides the display GPU; drop the env var |
| 2 | mounted archive wrapper dir | `Unrecognized model in /model. Should have a 'model_type' key in its config.json.` — mount the `original/` snapshot inside the wrapper |
| 3 | pinned v0.5.16 + correct mount | `model type 'glm5_next' but Transformers does not recognize this architecture` — image transformers too old |
| 3b | `:latest` (transformers 5.12.1) | same as #3 |
| 4 | `:latest` + `pip install -U transformers` | `ValueError: 'qwen3_asr' is already used by a Transformers config` — SGLang v0.5.x hard-registers qwen3_asr; new transformers ships it natively |
| 5a | `:latest` + `transformers==5.16.0` (checkpoint's declared version) | same collision — the declared version is what the quantizer ran (a dev build), not a release that knows the arch |
| 5b | pinned image + `transformers==5.16.0` | same collision from the pinned image too |
| 6 | `:latest` + sed `exist_ok=True` patch + transformers 5.16.0 | past the collision, but 5.16.0 genuinely lacks `glm5_next` |
| 7 | + latest transformers + patch | arch recognized! …no native model in release SGLang: `Unsupported TP style 'mla_kv_a_proj'` from the generic Transformers backend |
| 8 | SGLang **main** from source | same TP failure — GLM-5.3-Flash not merged to main (as of 2026-09-01) |
| 9 | **PR #36507 branch** + patch | `SyntaxError: keyword argument repeated: exist_ok` — the branch already fixed registrations; drop the sed patch |
| 10 | PR #36507 branch, clean | **loads, serves, 141.8 tok/s** |

## Bonus: full GLM-5.3 (704 GB FP8) does not fit one Station

Three attempts with `--cpu-offload-gb` (500/620/560) all OOM'd — GPU-side, then
host-side twice. dmesg showed the scheduler peaking at **1.37 TB total-vm with 741 GB
in shm**: SGLang's offload loader stages the full weight set through host shared
memory, so peak host demand ≈ model size + buffers > the machine's 744 GiB total.
Not tunable around with stock SGLang. Verdict for the FP8 checkpoint: dual-Station or
streaming-loader territory. NVFP4 Flash *is* the proven single-Station model.

Scope note: this verdict covers **FP8 (704 GB)** only. The full GLM-5.3 in **NVFP4
(~433 GB)** passes the loader's shm-staging math (433 + buffers < 744 GiB total) and
would need ~170 GB of Grace offload — untested as of this writing, expected slow,
but not excluded by the failure above.

## Also worth knowing

- Benchmark TTFT on a thinking model must count `reasoning_content` deltas, or
  "TTFT" silently includes thinking time. Our first prefill probe was garbage for
  this reason.
- The v1 "concurrency cliff" (15 s TTFT at C8+) was first-hit kernel autotune per
  batch shape, not a server defect. Warm up after every start (see README).


## Round 2 (2026-09-11) — what failed, and what round 1 got wrong

- **Round-1 conclusion "DFlash2 loses to AR at C16+" was wrong.** SGLang's default KDA state budget capped `max_running_requests` at 7; the C16/C32 rows were 16–32 clients sharing 7 seats. With `--max-mamba-cache-size` raised, DFlash2 beats AR per running request through C24. Lesson: read `max_running_requests=` in the boot log before interpreting any concurrency ceiling on a hybrid-attention model.
- **Round-1 "FP8 KV cache" claim was half true.** The boot log allocates two pools; the MLA KV is bf16. We had been reading the indexer pool's line.
- `--max-mamba-cache-size 660` (AR) and `320`/`200` (DFlash2) → `Loaded weights leave no GPU memory for the KV cache`. DFlash2 slots cost ~0.14 GB state + 7 × 0.03 GB intermediate, fp32. Use the ratio form or size from the boot log constants.
- `--mamba-ssm-dtype bfloat16` + DFlash2 → `lower_bound (safe gate) target verify is only supported by TritonKDAKernel; got FlashInferKDAKernel`. Works on AR with `--linear-attn-decode-backend triton`.
- `--mamba-radix-cache-strategy no_buffer` → `AssertionError: no_buffer only supports page_size=1`.
- AR-mode CUDA graph capture on the nightly image died with inductor `Could not find an active GPU backend` — compile-worker race; `TORCHINDUCTOR_COMPILE_THREADS=1` fixes it.
- SGLang `--cpu-offload-gb` on the FP8 original: `functional_call got multiple values for keys ['self_attn.attn.dt_bias', 'self_attn.dt_bias'], which are tied`. Oracle was served with vLLM UVA offload instead (120 GB, 9-min boot, 21 autotune configs).
- SGLang `/v1/completions` rejects `max_tokens=0`; teacher-forced scoring uses `max_tokens=1, echo=true` and drops the last token.
- First `x_search`-style checks of community claims: ebfio's 54–67% acceptance is on a vLLM stack with block 7; ours is 0.39 mean on SGLang. Block 7 matched his C1 gain but not his acceptance. Unresolved.
- Staging to a root-owned `/models` with `mkdir -p` silently no-ops and the size-verify reports every file as `MISMATCH … 0`. Use `sudo mkdir && chown` first.

## Round 3 (2026-09-16) — external corroboration: tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark

Independent 2x DGX Spark (GB10, SM121) vLLM TP2 recipe; tweet 2100290762155970877.
Portable takeaways for our stack:

- **ModelOpt NVFP4 quants emit intermittent corrupted token IDs** (vLLM #54150) —
  nearly invisible in English, but a corrupted token inside a tool-call block desyncs
  the parser and can spiral into repetition lock. Their fix: prefer
  **compressed-tensors** quants (RedHatAI/GLM-5.3-Flash-NVFP4) over ModelOpt builds
  (LibertAIDAI, keys-ablit). Probe method ports directly: Korean-Hangul probe,
  temperature 0, non-streaming, 3 passes, count U+FFFD (their data: ModelOpt 4/9/8,
  compressed-tensors 0/0/0). **TODO: run this probe against our FP4 checkpoints on .9 —
  cheap canary for the tool-call desync history.**
- **Never pin `--kv-cache-memory`** (vLLM): profiler still runs but never subtracts
  the measured activation peak (gpu_worker.py:475-495) — boots fine, dies on first
  long prompt. Let the profiler size the pool. (Station: less acute with 250 GiB HBM,
  but same rule.)
- **Headline "KV cache size" inflates with --max-model-len** — it is
  `int(max_concurrency * max_model_len)`, not bytes. Compare configs only by
  blocks x block_size / bytes-per-token.
- **Pool = min across ranks; rank 0's "Available KV" line can lie** (logged 6.25 GiB,
  bound 2.29 GiB). Read on every rank. (Moot for single-rank Station; matters on TP.)
- **Poll `/health`, never `/v1/models`** — the latter returns 200 with a dead engine.
- **Draft acceptance tracks output type**: structured/list/tool-args ~0.9, freeform
  prose ~0.33; agentic traffic lives in the high-acceptance zone. Also temp 0 = free
  throughput (+13-21%, exact top-1 sampler) and thinking-off drafts better (+8%).
  Matches our own acceptance observation (ebfio discrepancy note, round 2).
- **DFlash2 economics: +91% decode for -40% KV pool** (drafter costs ~4.8 GiB KV
  headroom vs 2.2 GiB weights). K=7 optimal, don't sweep.
- Their SM121 top-k kernel crash, NCCL NIC env, drop_caches/swap UVM livelock, and
  InstantTensor direct-IO loader instability are GB10/unified-memory-only — no
  transfer to the Station (HBM-only, Grace staging).

Cross-check vs round 2: their vLLM acceptance (0.40-0.53 at C1-C6, block 7) sits
between ebfio's claim and our SGLang 0.39 mean; direction consistent.
