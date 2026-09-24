# 2026-09-24: image input fix (transformers 5.16.1 derived image)

**Reported and diagnosed by [Paul Torruella](https://x.com/PaulTorrue36658).** He ran this recipe as the first inference engine on his DGX Station. Image input was silently broken: requests returned HTTP 200 with no error, but the prompt was text-only. He traced it to transformers, built the derived image, and sent the report. This bundle reproduces his finding on our Station and gates his fix.

Same weights (`nvidia/GLM-5.3-Flash-NVFP4@09b04e5e`), same draft (`incoai/GLM-5.3-Flash-DFlash2@7d74cdd8`), same daily flags (DFlash2 block 7, `--max-mamba-cache-size 48`, fp8_e4m3, mem 0.85, 1M ctx, MAXBS 16) as [Card D `newdf`](../2026-09-21-v0520-rebase/README.md), which is the reference for every text comparison here. Page cache dropped before each cold boot. Box paths are replaced by `$HOME` and the LAN IP by `<station-lan-ip>`.

## Cause

`lmsysorg/sglang:v0.5.20-cu130` ships **transformers 5.12.1**. GLM-5.3-Flash (`glm5_next`) was first released in **transformers v5.16.1**. On 5.12.1, `AutoProcessor.from_pretrained` returns a bare `TokenizersBackend`, so SGLang has no image processor and drops the image part of a chat message. Nothing is logged.

We lost this on the 9/21 rebase. The previous frozen image (`glm53-nvfp4-sglang:gb300-v2`) ran `pip install --upgrade transformers` and had 5.16.1, so images worked on it. All our gates were text-only, so none of them noticed. SGLang main still pins transformers 5.12.1, so moving to a newer SGLang tag would not fix this.

## Fix

[`Dockerfile.tf5.16.1`](../../Dockerfile.tf5.16.1) (sha256 `8cee99bdc9576884e394c2d636a2dbad5b1c5f1a668e2a48c02ca62bb04005b9`) is `FROM` the pinned v0.5.20 digest plus `pip install transformers==5.16.1 tokenizers==0.23.2`. The `pip freeze` diff against the base has exactly two lines ([`raw/image/`](raw/image/)): transformers 5.12.1 → 5.16.1 and tokenizers 0.22.2 → 0.23.2. The `pip check` complaints (protobuf, pillow) are identical in the base. Built locally on the Station, image ID `sha256:98dabd1b8b8816484c65c3edc8e29ec3f59b126ef362d97dbb5e6e51dd76ee3d`. It is not pushed to any registry; build it yourself (CPU-only, about 5 s on top of the base).

## Image input

`scripts/vision_gate.py` uses `scripts/vision-gate.png` (red square, blue circle, "HELLO"). It fails if the prompt grows by fewer than 100 tokens, if `image_tokens` is 0, or if the answer does not read HELLO.

| image | transformers | processor | prompt tokens text / +image | `image_tokens` | answer |
|---|---|---|---|---|---|
| stock `v0.5.20-cu130` | 5.12.1 | `TokenizersBackend`, no image processor | 29 / **32** | not reported | *"The image shows the Coca-Cola logo … red circle … white script"*. **FAIL**: the model invents an image. |
| derived | 5.16.1 | `Glm5NextProcessor` + `Glm5NextImageProcessor` | 29 / **392** | **361** | red square top-left, blue circle top-right, "HELLO". **PASS** |

The derived image passed on every boot (20 image requests across 6 server starts). Paul measured 543 prompt / 512 image tokens on a different image; the count depends on image size.

## Text path: unchanged

| check | result | raw |
|---|---|---|
| chat template render + token IDs, 6 cases (± tools, unicode, CJK) | byte-identical on 5.12.1, 5.16.1 (old frozen image) and the derived image | `scripts/tokparity.py` |
| SGLang's loaded `Glm5NextConfig` (serialized + attributes) | identical except new unused transformers attributes (`per_layer_config`, `is_heterogeneous`, fsdp plan) | `raw/image/cfg/` |
| teacher-forced vs the FP8 original (vLLM oracle text, 40 × 128 tokens) | stock **0.1360** / derived **0.1360** mean \|Δlogp\|, top-1 0.8439 on both, the same as the 2026-09-11 run | `raw/cardH-2026-09-24/h2/` |
| **H6: pinned rebuild, fresh boot** | TF vs Card D **max Δ 0.000000 over 2,857 tokens** · greedy **20/20** · tools 10/10 ×2 · accept len 3.46 / rate 0.41 · C8 699.4 agg (spread 3%) | `raw/cardH-2026-09-24/h6/` |
| text after image traffic (H3) | clean → after 1 image → after flush → after 11 images: TF identical at every step, greedy 20/20 before and after | `raw/cardH-2026-09-24/h3/` |

C1 answer-only (`reasoning_effort: low`, median of 3, tok/s):

| boot | history 512 | prose 300 | code 400 | shell 200 |
|---|--:|--:|--:|--:|
| Card D `newdf` (stock, 2026-09-21) | 201.9 | 151.0 | 288.2 | 162.3 |
| H5 r1 derived (Card H order, with images) | 202.3 | 151.0 | 288.6 | 161.5 |
| H5 r2 derived (no images) | 201.4 | 151.0 | 288.5 | 160.9 |
| **H6 pinned rebuild** | 200.4 | 149.8 | 285.9 | 157.7 |
| Card H `tf5` first boot (see below) | 204.1 | 154.4 | 313.6 | 160.2 |

## One unexplained boot

The first fresh boot of the derived image (Card H `tf5`) served a slightly **different model function**. Compared with Card D, greedy output matched **2/20**. Teacher-forced mean |Δlogp| was 0.146 (max 6.6) and differed from token 0. Code C1 was 313.6. Tools passed 10/10 and vision passed.

It did not reproduce anywhere else:
- a `docker start` of the same container (H2, H3);
- a fresh boot with an image as the first request (H4x);
- a fresh boot with text first, then an image (H4y);
- two fresh boots replaying Card H's exact request order, with and without images (H5 r1, r2);
- the pinned rebuild (H6).

All seven match Card D (greedy 20/20 on the six where greedy ran; TF within the bar on all seven, and Δ 0.0 in H3 and H6 where the exact delta was computed). Boot logs show the same KV/Mamba allocation, the same autotune passes and the same 62 persisted CuTe-DSL kernels on each boot.

So it is not transformers, not image traffic and not request order. It is one server process that came up with different numerics and kept them until it was restarted. We do not know the cause. We record it because a single boot's text gate is only evidence about that boot: **gate text quality after every boot.** The `tf5` receipts are kept in `raw/cardH-2026-09-24/tf5/`. A process-level nondeterminism in kernel selection is the leading guess, and it is untested.

## Decision

The recipe image moves to the derived image. Image input is now gated on every boot with `vision_gate.py`. The Card D text numbers carry over unchanged, with H6 as the receipt.
