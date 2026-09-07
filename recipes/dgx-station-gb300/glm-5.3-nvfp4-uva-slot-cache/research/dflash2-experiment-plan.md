# Queued experiment: DFlash2-over-UVA acceptance probe

Status: **QUEUED** (2026-09-07). Idea credit: keys (drowzeys, GitHub). Not started; no GPU time spent.
Sourcing note: raised in session `milo/20260907_095808_e77b25` on 2026-09-07 after the 512K daily-profile rollout.

## The idea

Replace GLM-5.3's native MTP(1) with incoai's DFlash2 block-diffusion draft model
(`incoai/GLM-5.3-DFlash2`, 2B bf16) on the sc13g slot-cache build, to raise decode
throughput on the one-GB300 UVA recipe.

## Why it is plausible

- inco's model card: DFlash2 beats native MTP on every evaluated task (acceptance
  5.94 vs 5.12 on GSM8K; 3.2x vs 2.6x speedup at C1), measured on 4x GB300 TP4.
- catid/dgx_station_benchmarks: 154.9 tok/s C1 / 742 C16 with DFlash2 on 2x GB300
  PP2 (HBM-resident).
- Our own vLLM image (`vllm-glm53-uva:v0.28.0-2cf0a691`) already contains
  `vllm/v1/spec_decode/dflash.py` and the `dflash` speculative config method —
  verified live 2026-09-07. Not engine-blocked.
- DFlash2 is verified-lossless by construction — a cleaner quality position than
  MTP(1), whose promotion was blocked on an INCONCLUSIVE quality audit.

## Why it is not a drop-in

- Every published DFlash2 number is HBM-resident. Nobody has published DFlash2
  over UVA offload.
- Verification width: MTP(1) verifies 2 tokens/step; DFlash2 K7 verifies 8.
  Per verification step that is up to 4x the offload traffic of MTP(1) and up to
  8x plain decode, through 256-expert layers whose cold rows are in Grace memory.
  Slot-cache amortization vs thrash is unknown.
- Our UVA/slot-cache patches have never been exercised with spec decode enabled.

## Bar to beat (measured rows in this recipe)

| Config | C1 | C4 agg | C8 agg |
|---|---:|---:|---:|
| sc13g no MTP | 43.1 | 92.0 | 95.6 |
| sc13g + MTP(1) | 54.7 | 107.8 | 102.9 |

## Gated sequence (do in this order)

1. **Static geometry audit** — verify `incoai/GLM-5.3-DFlash2` matches our NVFP4
   target geometry (catid's `current-target.json` approach; his audit says the
   draft matches the 78-layer full GLM-5.3 geometry). Free; kills the idea fast
   if mismatched.
2. **Acceptance-length probe** — sc13g + DFlash K4 on the prose/code battery.
   **Stop gate: acceptance < ~3 means the offload penalty eats the win; close
   the experiment there.** HBM-resident ceiling is ~5-6 (inco); ours will be lower.
3. **Full bench only if step 2 passes** — C1/C4/C8 vs the table above, same
   harness as the recipe's existing rows.
4. **Teacher-forced divergence check** — "verified-lossless by construction" is
   not a measurement (same standard we applied in the Flash DFlash2 recipe).
5. **Publish with receipts** — recipe + blog update, first-party numbers only.

Realistic expectation if the inco MTP->DFlash2 ratio (~1.25x) transfers:
~65-70 tok/s C1. Meaningful, not transformative; the honest ceiling on this box
is UVA bandwidth, not draft efficiency.

## License note

`incoai/GLM-5.3-DFlash2` is CC BY-NC-ND 4.0 (research/evaluation). Our use on our
own hardware is research/evaluation; any commercial serving off this recipe needs
a separate license from inco.ai.

## Preconditions

- Do not run against the live 512K daily-profile container. This experiment gets
  its own launch window via `scripts/launch-slotcache-portable.sh` with
  `--speculative-config` swapped from `mtp` to the `dflash` method (exact flag
  shape to be confirmed against the image's `speculative.py` at run time).
- Decode-at-occupancy re-bench of the 512K profile is queued separately and
  should land first (it is the missing baseline for any decode-speed comparison).
