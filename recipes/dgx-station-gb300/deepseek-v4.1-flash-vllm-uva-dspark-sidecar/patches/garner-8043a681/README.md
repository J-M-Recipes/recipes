# Patches on Garner McCloud's DS4.1 release (Pg61; Pg62 adds `06` + `w1-exact/`)

The engine these patches apply to is **Garner McCloud's**
[`garnermccloud/dgx-station-recipes`](https://github.com/garnermccloud/dgx-station-recipes) at
[`8043a6811a5feda70fdfb6e827eebb2ec25c65d1`](https://github.com/garnermccloud/dgx-station-recipes/tree/8043a6811a5feda70fdfb6e827eebb2ec25c65d1/deepseek-v4.1-flash),
directory `deepseek-v4.1-flash/` (MIT). His code is **not** vendored here. Clone it at that commit and point
`GARNER_SRC` at the checkout. `scripts/launch-pg61.sh build` refuses any other HEAD.

`launch-pg61.sh build` applies the files below in name order with `patch -p1 --forward` to a copy of
`deepseek-v4.1-flash/`. All five were re-applied on the desk to a clean checkout at the pin. The result matches the
frozen build tree that serves `:30006` file for file
([`results/2026-10-29-pg61-garner-engine/patch-desk-check.txt`](../../results/2026-10-29-pg61-garner-engine/patch-desk-check.txt)).

| file | touches | why |
|---|---|---|
| `01-policy-hooks-serve.patch` | `serve/serve.sh` | Five hooks, each a no-op unless its variable is set. `DS41_NO_KEY=1` serves without his API key file. `DS41_ATTN_CFG` (unset = no `--attention-config`) drops his MXFP4 indexer K / sparse-logits config. `DS41_GEN_OVERRIDE` (unset = no `--override-generation-config`) drops his forced T=1.0 / top_p 0.95. `DS41_EFFORT_DEFAULT=0` drops his `reasoning_effort` default. `DS41_PEER3_SPINS` forwards `PEER3_SPINS` into the server container (Card G3-desk). |
| `02-policy-hooks-recipe-env.patch` | `config/recipe.env` | `DS41_FASTPATH_COMPONENTS_OVERRIDE`: the component list can be overridden without editing his file. Pg61 runs 24 of his 29 components. It drops `ds41_sampling` and `ds41_stop` (sampling policy) and `w8oproj`, `lowb_aux` and `tail_drafthead_mxfp8` (each changes numerics by design). |
| `03-61161-w5-markov.patch` | `fastpath/w5_markov.py` | Ports vLLM [#61161](https://github.com/vllm-project/vllm/pull/61161) `spec_step_seed` into his fused Triton draft sampler: `seed ^ (step << 32)`. |
| `04-61161-w5-markov-gemv.patch` | `fastpath/w5_markov_gemv.py` | The same line in the GEMV variant. |
| `05-61161-w6mk.patch` | `kernels/w6-moe/markov/w6mk.cu` | The same seed mix in the CUDA draft sampler. `build` recompiles it into `fastpath/libw6mk.so` (`nvcc -gencode=arch=compute_103a,code=sm_103a` inside the image, CPU only). |

Why 03–05: v0.31.0 ships the pre-#61161 sampler, and his fused drafters reimplement the same unmixed draw ("no
`spec_step_seed`"). With block verification and probabilistic drafting, re-drafted positions then reuse the draw that
was just rejected. That biases every temperature > 0 sample. Card G3's GPU exactness test
([`scripts/test_markov_61161.py`](../../scripts/test_markov_61161.py)) compares both kernels with production's eager
reference at T=1. Patched, they match at every step: w6 30/30 per step, w5 120/120. Stock, they match only at
step 0. At T=0 the token streams are identical.

## The vLLM overlay (`overlay-vllm/`)

`launch-pg61.sh build` copies his `docker/overlay-58132/vllm` (his #58132 overlay: `config/`, `models/`). It then
copies `$PATCHES/overlay-vllm/` on top, which carries two files. Both are **the same bytes that h2-REF mounts**, built
from patches this recipe already publishes:

| file in `overlay-vllm/` | build from | sha1 (launcher checks) | sha256 |
|---|---|---|---|
| `v1/worker/gpu/sample/gumbel.py` | stock + [`../vllm-61161-block-verification-fresh-draws.patch`](../vllm-61161-block-verification-fresh-draws.patch) | `6eac3e1348339d157ba0e2184114859aca207df1` | `ee5072d2…` |
| `v1/worker/gpu/spec_decode/rejection_sampler_utils.py` | stock + the #61161 patch + [`../vllm-60945-rejection-sampler-guard.patch`](../vllm-60945-rejection-sampler-guard.patch) | `edc8f8f4c425f51a962840fc2e0e7aa32059f57a` | `2d43d513…` |

The stock files in `vllm/vllm-openai:v0.31.0` are sha1-identical to af7f9488's (`gumbel.py` 70ae04f2,
`rejection_sampler_utils.py` fb99972e; Card G3-desk), so the h2-REF patches apply unchanged. Desk check:
[`overlay-desk-check.txt`](../../results/2026-10-29-pg61-garner-engine/overlay-desk-check.txt). The two files **ship in
`overlay-vllm/`** (launcher sha1-gates them); to re-derive or audit the bytes:

```bash
P=patches/garner-8043a681; IMG=vllm/vllm-openai@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3
DP=/usr/local/lib/python3.12/dist-packages; T=$(mktemp -d)
for f in v1/worker/gpu/sample/gumbel.py v1/worker/gpu/spec_decode/rejection_sampler_utils.py \
         v1/worker/gpu/spec_decode/dflash2/speculator.py; do
  mkdir -p $T/vllm/$(dirname $f); docker run --rm --entrypoint cat $IMG $DP/vllm/$f > $T/vllm/$f; done
patch -d $T -p1 < patches/vllm-61161-block-verification-fresh-draws.patch
patch -d $T -p1 < patches/vllm-60945-rejection-sampler-guard.patch
mkdir -p $P/overlay-vllm/v1/worker/gpu/sample $P/overlay-vllm/v1/worker/gpu/spec_decode
cp $T/vllm/v1/worker/gpu/sample/gumbel.py $P/overlay-vllm/v1/worker/gpu/sample/
cp $T/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py $P/overlay-vllm/v1/worker/gpu/spec_decode/
# (dflash2/speculator.py is patched too but not mounted: it is not on the DSpark path)
```

Why the guard is mandatory on this engine: his release on stock v0.31.0 runs the unguarded #60945 read. Under his
k-schedule (`[[1,4,5],[5,8,3],[9,32,3]]` + block verification), a full 32-request step is 32 × 4 = 128 logits ×
4 B = exactly one 512 B allocator block. The read lands past it, and whether that faults depends on allocator layout.
It killed both G1 arms and four bisect arms (Cards G1–G1e), and turning off an inert component (`lowb_aux`) was enough
to flip it. The 3-line guard fixed it with nothing else changed (Card G1f V1). Reported upstream to Garner as
[issue #1](https://github.com/garnermccloud/dgx-station-recipes/issues/1).

## Pg62 (2026-11-14): `06` + `w1-exact/` — applied only with `DS41_PG62=1`

Pg62 = Pg61 + two changes to W1 decode attention (round bundle:
[`results/2026-11-14-pg62-w1-exact/`](../../results/2026-11-14-pg62-w1-exact/README.md)). `launch-pg61.sh build` applies
`01`–`05` only (its glob is `0[1-5]-*.patch`); `DS41_PG62=1 launch-pg61.sh build` then runs the Pg62 block, which uses
the files below. Nothing here vendors Garner's source or binaries: the two `.so` files are rebuilt from his pinned source
in the pinned image.

| file | applies to | what it does | sha256 |
|---|---|---|---|
| `06-w1-fp8-flag-serve.patch` | `serve/serve.sh`, after `01` | Two lines: passes `DS41_W1_COMPRESSED_FP8` (0 or 1, anything else refuses) into the server container. Reproduces the box's `raw/promote/live/serve.sh.pg61-to-pg62.diff` byte for byte. | `7d15187f…` |
| `w1-exact/w13-fp8-compressed-records.patch` | `deepseek-v4.1-flash/` (`-p1`), after `01`–`06` | The "W13" FP8 routing: `fastpath/w1attn/{formats,integrate,ops}.py`, `kernels/w1-attn/{build.sh, csrc/w1_attn.cu, include/w1/*.cuh}`. With `DS41_W1_COMPRESSED_FP8=1`, target layers 2–39 at 1–16 decode rows run W1 on `fp8_ds_mla` 528 B records through a second extension `w1_attn_fp8_ext.so`; 17+ rows stay on the stock op. | `8366c92f…` |
| `w1-exact/fp8-variants.patch` | the W13-patched `kernels/w1-attn/` (`-p1`) | Compile-time switches for the FP8 extension. `-DW1_VARIANT_STOCK_ALL` = sink term unfused + ex2 without FTZ + the `osc` normalizer divide (`1/(l + 2^(sink·log2e − m))`) without FTZ + output-scale divide `div.rn`: the stock op's rounding. | `758f25a4…` |
| `w1-exact/incumbent-variants.patch` | a copy of the pinned `fastpath/w1attn` + `kernels/w1-attn/{csrc,include}` (`-p1`) | The same switches for his incumbent extension `w1_attn_ext.so` (window-only layers 0–1 and drafter layers 40–42). | `eac3d500…` |
| `w1-exact/build_candidate_F.py` | — | Builds `w1_attn_fp8_ext.so` (CPU only, `W1_EXTRA_CFLAGS`). | `587514df…` |
| `w1-exact/build_incumbent_driver.py` | — | Builds `w1_attn_ext.so` through his own `ops.build()` (CPU only, `W1_EXTRA_CFLAGS`). | `fce6d1b6…` |

Both served extensions were built with `-DW1_VARIANT_STOCK_ALL`: `w1_attn_fp8_ext.so` `de87450648e96214…` and
`w1_attn_ext.so` `3bca6cac288d7795…` (in-container sha256 after the cutover:
`raw/promote/live/in-container-sha256.txt`). The served Pg61 incumbent was his committed prebuilt `7201643d…`.

**Why.** W1 computed the softmax sink term `sink*log2e - m` as one fused multiply-add; the stock FlashMLA op multiplies,
then subtracts. Sometimes that one float step puts a pre-quantization value on the other side of an FP8 rounding
midpoint, and ONE output data byte flips. The fixture probe (`raw/builds/fp8/*/F-probe.json`) shows the sink switch on
its own moves the failing byte from 111 back to stock's 110 (`CONFIRMS`). The other three switches, each on its own,
leave it at 111 (`NO-CHANGE`). In the served shadow the sink-only incumbent build was also clean (0 of 17,733 calls).
Only the STOCK_ALL builds were speed-measured and promoted.

**Desk check** ([`patch-desk-check.txt`](../../results/2026-11-14-pg62-w1-exact/patch-desk-check.txt)):
- `06` applies after `01`–`05` and reproduces the box diff. Resulting `serve.sh` sha256 = the box manifest's.
- The Pg61 W1 source (`milo-takeover/results/I/pg61-src`, 17 files) plus `w13-fp8-compressed-records.patch` equals
  the W13 source tree for all 15 files both trees have. Not checkable: `kernels/build_all.sh` and `kernels/common.sh`
  (not in the W13 source copy; the patch does not touch them).
- A stub-docker `DS41_PG62=1` build matches the box's `pg62-BUILD-SHA256SUMS` on 112 of 117 files. The five
  misses: three `.so` (the stub has no nvcc), the watchdog (known published-copy difference) and `policy.env` (equal
  once `WORK` is the box path).
- **Launcher build verified 2026-11-15 on the reference box.** The real `DS41_PG62=1 build` (two nvcc builds in the
  image), from a fresh clone at the pin, produced both `.so` byte-identical to the served files (`3bca6cac…`,
  `de874506…`; cubins `3a5a0123…`, `bb8b96a8…`) and an identical `tree/`:
  [`raw/launcher-build/`](../../results/2026-11-14-pg62-w1-exact/raw/launcher-build/). The served tree itself was
  assembled by hand; no lane has been booted from the launcher-built one.

Not in the recipe: the cluster-kernel (cx) FP8 port for 17–76 rows (byte-exact in shadow, no throughput gain; the
`results/G/` drafts stay in the research tree).

## Not in this directory

- The fail-closed watchdog [`../../scripts/sidecar_watchdog_p3.py`](../../scripts/sidecar_watchdog_p3.py). On the
  box it sat next to these patches. The published launcher reads it from `scripts/` (`WATCHDOG=`).
- `PEER3_SPINS=1000000000000` is set by the launcher's `policy.env` through hook 01. It is not a patch.
