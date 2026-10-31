# overlay-vllm — the two vLLM files `build` mounts over Garner's `overlay-58132`

`launch-pg61.sh build` copies `docker/overlay-58132/vllm` out of Garner's tree into `$WORK/overlay-vllm`, then copies
this directory on top of it, then sha1-gates both files (`WANT_GUMBEL`, `WANT_RSU` in the launcher). Two files ship:

| path (under `v1/worker/gpu/`) | sha1 | sha256 |
|---|---|---|
| `sample/gumbel.py` | `6eac3e1348339d157ba0e2184114859aca207df1` | `ee5072d245ea850ff4efa60ec548c172484019415f5d687b805d2d6af650ade1` |
| `spec_decode/rejection_sampler_utils.py` | `edc8f8f4c425f51a962840fc2e0e7aa32059f57a` | `2d43d5133c1fb96399e186ed10c5a26b54c12e6a019d9d163cbff1adbcfbcc39` |

**Provenance.** Both are stock `vllm/vllm-openai:v0.31.0` files (Apache-2.0, SPDX headers retained) with this recipe's
two upstream backports applied in order:

1. [`../../vllm-61161-block-verification-fresh-draws.patch`](../../vllm-61161-block-verification-fresh-draws.patch) —
   vLLM #61161: block verification draws fresh randoms per speculative step instead of reusing one. Inert at T = 0;
   removes a temp > 0 bias. Kernel-exact against the eager reference (Card G3: w6 30/30, w5 120/120 per step).
2. [`../../vllm-60945-rejection-sampler-guard.patch`](../../vllm-60945-rejection-sampler-guard.patch) — vLLM #60945:
   three lines at the top of `_compute_local_residual_mass_kernel` (`return` when `logit_idx + 1 >= num_programs`).
   Without it, stock v0.31.0 reads 4 B past a 512 B tensor at 32 requests × k = 3; it killed both Card G1 arms.

The stock inputs those patches apply to are sha1-identical on af7f9488 and v0.31.0 (Card G3-desk), so the same two
patch files build the h2-REF overlay (`patches/vllm-*` at the recipe root) and this one. The rebuild is receipted in
[`results/2026-10-29-pg61-garner-engine/overlay-desk-check.txt`](../../../results/2026-10-29-pg61-garner-engine/overlay-desk-check.txt):
stock `gumbel.py` `70ae04f2`, stock `rejection_sampler_utils.py` `fb99972e`, both `patch -p1` clean, results equal the
files the launcher mounts.

**To audit the bytes without a GPU:** take the two stock files from a local `vllm/vllm-openai:v0.31.0` image
(`/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/`), apply the two patches above with `patch -p1`, and
compare sha1/sha256 against the table. **What is not claimed:** these files are ours to maintain only as a backport —
when an image ships both PRs, the overlay should be dropped, not carried forward.
