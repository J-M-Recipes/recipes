# Patches on Garner McCloud's DS4.1 release (Pg61)

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
[`overlay-desk-check.txt`](../../results/2026-10-29-pg61-garner-engine/overlay-desk-check.txt). To populate the overlay
before `build`:

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

## Not in this directory

- The fail-closed watchdog [`../../scripts/sidecar_watchdog_p3.py`](../../scripts/sidecar_watchdog_p3.py). On the
  box it sat next to these patches. The published launcher reads it from `scripts/` (`WATCHDOG=`).
- `PEER3_SPINS=1000000000000` is set by the launcher's `policy.env` through hook 01. It is not a patch.
