# DeepSeek-V4.1-Flash on one GB300, no sidecar GPU (gracetier)

Status: **experimental**, first published 2026-11-09. Card: `card.png`. Blog:
<https://al-engr.com/gb300-dsv41-decode.html>. Code:
[jmeadlock/gb300-dsv41-decode @ fadc3d2](https://github.com/jmeadlock/gb300-dsv41-decode/commit/fadc3d2eb7289189d82ba40cbbdc9b3f10b357a2)
(Apache-2.0).

> **Read this first.** We set a release bar for this recipe before measuring it: at least 1.25x our October
> single-GPU recipe at one stream. It measured 1.11-1.16x. The recipe is published anyway, as the measured answer to
> "how fast is Garner McCloud's DS4.1 engine without the second GPU", with the miss stated. It is not a standing
> service yet: the launcher boots, measures and restores.

## At a glance

All three columns are the same harness on the same night (2026-11-09): 40 general prompts, 256 new tokens each,
greedy, thinking off. Receipts in `results/2026-11-09-gt-a5/`.

| | **this recipe** (1 GPU) | October recipe, v28-prob (1 GPU) | Pg61 (GB300 + RTX PRO 6000) |
|---|---|---|---|
| 1 stream, set A / set B, tok/s | **261.5 / 260.5** | 226.4 / 223.5 | 414.2 / 410.4 |
| 1 stream, set A again after the concurrent cells | 273.5 | 247.0 | 414.4 |
| engine step at 1 stream, ms (A / B) | 11.80 / 11.49 | 13.67 / 13.52 | 7.65 / 7.64 |
| 8 streams, aggregate tok/s | 489.4 | 452.3 | 909.0 |
| 32 streams, aggregate tok/s | 717.5 | 683.2 (admits 24, 8 queue) | 1,733.3 |
| first token at 32 streams, median / max, s | 1.43 / 1.65 | 2.47 / 8.91 | 1.23 / 1.38 |
| KV cache at 1M context, tokens | 8,403,455 (8.0x) | 2,481,694 (2.4x) | not re-read that night |
| tool gauntlet | 12/12 | not run that night | 12/12 |

Ratios, this recipe over the October recipe: 1.155 and 1.166 at one stream (1.107 on the later pass, when both had
adapted further), 1.08 at 8 streams, 1.05 at 32. Pg61 over this recipe: 1.52-1.58 at one stream, 1.86 at 8, 2.42
at 32.

| time to first token by prompt length (median of 3) | 31 tok | 286 | 954 | 7,621 | 60,954 |
|---|---|---|---|---|---|
| this recipe, s | 0.050 | 0.094 | 0.097 | 0.183 | 1.337 |
| Pg61, s | 0.020 | 0.123 | 0.127 | 0.184 | 1.474 |

## What this runs

Garner McCloud's DS4.1 engine for the DGX Station computes the 129 least-used experts of each layer on an RTX PRO
6000 in the same chassis. This recipe removes that card from the path:

- **Every layer runs all 384 experts in one MegaMoE call.** 255 per layer sit in HBM as before. The other 129 sit in
  pinned Grace memory (85.26 GiB), and each has a fixed slot in one HBM scratch area that all 40 layers share.
- **One kernel per layer (`gt_stage`) copies in what the step needs.** It reads the router's picks on the device and
  copies only the cold experts that were picked from Grace into their slots over NVLink-C2C, inside the captured
  graph. Nothing is computed on the CPU.
- **Adaptive residency moves the popular experts into HBM.** The same kernel counts picks per expert. Between
  engine steps a manager exchanges a hot expert with a cold one (three copies and a scale-row swap; every address
  stays fixed, so no graph is re-captured). The decision rule is the one our October recipe ships, unchanged, with
  hysteresis raised from 1.5 to 3.0.

A kernel-level profile on the static map (card A2) showed where the single-GPU cost is: of a 10.49 ms step,
`gt_stage` is 2.60 ms and everything else is 7.89 ms, against 7.75 ms for the whole two-GPU step. Residency exists
to shrink that one kernel's work.

## Hardware

One NVIDIA DGX Station GB300: 250.7 GiB HBM visible, 494.5 GiB host memory, driver 595.99.02, Ubuntu 24.04.5,
kernel 7.0.0-1024-nvidia-64k (`results/2026-11-09-gt-a5/system.json`).

- The reference box also has an RTX PRO 6000 installed, driving the display. This recipe does not pass it to the
  container. A Station without that card was not tested.
- Host memory in use by the server: 189.13 GiB Engram + 85.26 GiB cold experts + 48 GiB host KV tier = 322.4 GiB.
  The component refuses to boot under 32 GiB of headroom; the receipts show 124-126 GiB.
- Checkpoint on local NVMe.

## Software

- `vllm/vllm-openai:v0.31.0`, digest `sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3`,
  stock. Nothing is built into an image.
- Garner's engine: `garnermccloud/dgx-station-recipes` @ `8043a6811a5feda70fdfb6e827eebb2ec25c65d1`, as built by the
  sibling sidecar recipe (`launch-pg61.sh build`: his tree plus our lane patches and the vLLM #61161 / #60945
  overlay). 23 of his fastpath components run: Pg61's 24 minus `coldskip` and `peer3`, plus `gracetier`.
- The gracetier component and its runtime modules from `jmeadlock/gb300-dsv41-decode` @ `fadc3d2`
  (`port/garner/fastpath/fastpath_gracetier.py`, `runtime/gracetier_{vmm,pool,stage,res}.py`,
  `runtime/residency_policy.py`), and three small patches to Garner's tree in `patches/` (`SHA256SUMS` covers
  `patches/` and `scripts/`).
- Checkpoint `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e10`, weights as shipped.

## Launch

There is one launch path and it is a measurement card, not a service:

```bash
# prerequisite: the sibling sidecar recipe's Pg61 tree is built on this box
git clone https://github.com/jmeadlock/gb300-dsv41-decode && cd gb300-dsv41-decode && git checkout fadc3d2
GT1_WINDOW=my-window GT1_RES=1 GT1_RES_CELLS=a5 GT1_SEED_CACHE=1 \
  GT1_RES_CONFIG='{"mode": "adaptive", "freeze": false, "hyst": 3.0}' \
  bash bench/box/card-gt1.sh
```

It refuses unless the box is in the expected state, stops Pg61, builds the gracetier tree (a patched copy of the
Pg61 tree), boots it on `:30006` as `dsv41-flash-gracetier`, runs the cells in `scripts/a5-cells.sh`, stops it and
restores Pg61. Boot to serving took 4 min 19 s to 4 min 39 s with seeded JIT caches. `scripts/` holds the files that
ran, with box paths replaced by `$HOME`.

Variant: `GT1_HOT_MAP=docs/maps/hot-garner255-plus-275.json` boots 275 hot experts per layer. It is about 1 ms per
step faster on general prompts and leaves 2.76M KV tokens instead of 8.40M. Use it for one user, not for many.

## Verify

- Boot log: `DS41_FASTPATH ACTIVE gracetier`, `components ACTIVE 23`, `ENGAGED: 40 layers, H [255], G = 384`, a
  `HOST_RAM_BUDGET [post-load ...] -> fits` line, `residency: manager over 40 layers`, and
  `GPU KV cache size: 8,403,455 tokens`.
- `curl -fs http://localhost:30006/v1/models` lists `dsv41-flash-gracetier`.
- Byte audit: write `{"audit": {"moved": true, "random": 64}}` as a control document
  (`scripts/gt_res_drive.py ctrl`); the acknowledgement must report `mismatches 0`. It did in all 8 audits.
- Residency telemetry (`telemetry.jsonl` in the control directory): one line per 64 engine steps with the cold
  share of picks and the swap counts.

## Results

Run `2026-11-09-gt-a5`; index in `results/2026-11-09-gt-a5/README.md`, every number in `throughput.csv`.

**Three-way (card A5)** is the table at the top.

**How the single-GPU number was built up, same evening:**

| step | lab step prompt, ms/step | general prompts, ms/step |
|---|---|---|
| static 255-hot map (cards GT1, GT4) | 10.45-10.50 | 13.00-13.25 |
| + adaptive residency, hysteresis 1.5 (GT3, GT4) | 8.20 (one prompt repeated) · 9.60 (held-out prompt, frozen map) | 11.42-12.03 |
| + hysteresis 3.0 (A5) | not run | 11.27-11.80 |
| 275 hot + residency, hysteresis 1.5 → 3.0 (A4) | not run | 10.46-10.92 → 10.45-10.80 |
| Pg61, two GPUs | 7.75 | 7.64-7.68 |

- On the 8-prompt lab fixture the cold share of expert picks fell from 4.3 % to 1.6 % and reached 90 % of that drop
  in 6.4 s. On general prompts it settled at 5.5-7.5 % with 255 hot and 3.7-5.4 % with 275.
- Restoring the boot map brought the step back to within 0.8 % of the control each time.

**Output checks:**

| check | result |
|---|---|
| swap byte audit (every moved expert + 64 random per layer, against boot digests) | 0 mismatches, 8 audits, 5 boots |
| greedy fixture, adapted map vs itself / vs the boot map | 8/8 / 8/8 identical |
| greedy fixture vs Pg61 (static map, card GT1) | 0 top-1 flips before divergence; 2/8 streams identical; max abs Δlogp 0.376 (Pg61 vs itself: 7/8, 0.121) |
| tool gauntlet | 12/12 (Pg61 same night 12/12) |
| 32-stream burst | 64/64 completed, server alive |

## Known limits

- **Missed its own speed bar** (1.11-1.16x the October recipe at one stream against a pre-registered 1.25x).
- **Not a standing service.** No launcher that leaves it serving, no health cron, no soak. Each boot served only its
  card's cells.
- **Needs the sidecar recipe's built tree** as its starting point; a build on a box that never ran that lane was not
  attempted. The reference box has the RTX card installed.
- **General prompts are slower than our lab prompts**, because the row map was fitted to lab traffic. Quote the
  general-prompt rows.
- **Residency keeps swapping on mixed traffic.** Hysteresis 3.0 halved the swap rate with equal or slightly better
  step time in one pass each; it is a direction, not a tuned value.
- **Concurrency is the weak axis against two GPUs**: 0.54x at 8 streams, 0.41x at 32.
- **Quality gates not run:** teacher-forced bracket, GSM8K, BFCL, needle, a 1M-token prompt, temperature > 0
  sampler calibration. Greedy streams differ from Pg61's on 6 of 8 lab prompts.
- **First token measured to 61K prompt tokens only.** The first run at a new length is slower (5.5 s at 61K).
- **One evening, one run per cell.** No run-to-run spread for the concurrent cells.
- **The 275-hot variant was measured at one stream only.**
- **Wait sites not audited for fail-open behaviour.** There is no sidecar to time out, but nobody has read every
  wait in the path for what it does on a stall.

The full list with detail is `limits:` in `recipe.yaml`.

## Rollback

The card's exit path is the rollback: on any exit after Pg61 is stopped it stops the gracetier containers and runs
the Pg61 restore (`launch-pg61.sh up`, watchdog in enforce mode). It returned 0 in all seven card windows on
2026-11-09. Nothing else on the box is changed.

## Credits

**What this recipe runs**

- **Garner McCloud** — the DS4.1 engine this is built on: `serve.sh`, the fastpath component framework, the MegaMoE
  fork, the fused boundary, the row map. [garnermccloud/dgx-station-recipes](https://github.com/garnermccloud/dgx-station-recipes/tree/8043a6811a5feda70fdfb6e827eebb2ec25c65d1)
  (MIT). gracetier is one component in his framework and three small patches to his tree. It has not been proposed
  upstream yet.
- **original-el8** — the hook lineage Garner's engine and our earlier lanes share.
- **DeepSeek** — the model, the in-checkpoint draft head, DeepGEMM and FlashMLA.
- **vLLM** — the serving engine (v0.31.0).
- **Ours** — the gracetier component, the VMM scratch layout, the stage kernel, the residency manager, and the
  v26ar residency rule from the October recipe.

**What it is measured against**

- **Pg61**, the sibling sidecar recipe (Garner's engine with the RTX sidecar, under our policy) — the two-GPU
  column.
- **v28-prob**, the sibling single-GPU recipe — the one-GPU control.

## Finalisation checklist

- [ ] A launcher that leaves gracetier serving, with a lane check, and a soak.
- [ ] Teacher-forced bracket against Pg61 with a same-map control; GSM8K, BFCL, needle.
- [ ] A served 1M-token prompt.
- [ ] Repeat brackets for the C1 / C8 / C32 cells.
- [ ] A build from a clean checkout on a Station without the sidecar lane.
- [ ] Hysteresis and hot-count sweep on general traffic at 8 and 32 streams.
