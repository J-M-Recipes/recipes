# T3b5 — cache-pin discriminator (nightly `dee37d89` + v15 hook, T3b4 fast set loaded)

Worker: grok-4.6 · xai-oauth. Evening 21:16–21:51 CDT 2026-09-19. Runner `scripts/t3b5-cachepin-runner-2026-09-19.sh` (already staged; not rewritten). Receipts: `receipts/` (window logs/json/metrics, boot/smoke, parity, three `autotune_configs.json` + sha256s). Campaign log: box `$BOX_HOME/dsv41/overnight-campaign-2026-09-18.log`.

Question: does loading T3b4's live-tune cache (196 C1) reproduce 196 (recipe = pin a good cache) or 172 (load-path tax)?

## 0. Preconditions / launch

- Box 21:15: only `dsv41-vllm-v18-cgsizes-BOUND-REF` Up on `:30006` (`dsv41-flash-uva`); no `glm53-*` / `dsfv-*`; no `STOP-CAMPAIGN`; MemAvailable 130058816 kB (log: 123 GiB); cache dirs `bba7410c…`, `ed692e15…`, `ed692e15….from-T3b4` present.
- Detached: `nohup bash $BOX_HOME/pin-hot-experts/scripts/t3b5-cachepin-runner-2026-09-19.sh > $BOX_HOME/dsv41/t3b5-nohup.out 2>&1 &` + `disown`. PID 234333. `ps -eo args | grep '^bash $BOX_HOME/pin-hot-experts/scripts/t3b5'` (not `pgrep -f`).
- `place_fast`: slow set → `.aside-T3b5`; FAST copied to canonical (original `.from-T3b4` kept). First `sha256sum` as `milo` logged Permission denied (root-owned json). `cachepin-sha-T3b5.txt` later holds three hashes; independently re-hashed with `sudo sha256sum` (same values):
  - fast / `.from-T3b4` / `.after-T3b5`: `5440f02d5ba960ecd5a0b29db6d7ae6377b239c3766f01e6e62592550bbe4241`
  - slow (canonical after restore): `17abc02e60a7fc8d6c2cab7bf94fc3bb154122324b8a42694bbc822a9cbbb16a`
- Runner has `set -f`; `mv "$OUT"/*"$t"*` did not glob, so window json stayed in `overnight-T1T2/` (copied to Mac receipts; later `cp` into `overnight-T3/`). `led` grep for `window-$t.log` failed the same way. Windows themselves completed.

## 1. Boot facts (hash hit — mandatory)

| boot | bind | autotune | HBM_expert / pinned | host_avail | KV | graphs |
|---|---|---|---|---|---|---|
| **T3b5** 21:24:18 | **480 s** | **Loaded 189**; Autotuning process ends; **Saved 189 (0 new, 189 from previous config)** | 206.61 / 62.33 | 19.46 GiB | **4,530,275 tok (4.32× at 1M)** | 73 s / 1.29 GiB then 15 s / 1.12 GiB |
| **T3b5b** 21:43:34 | **480 s** | **Loaded 189**; same Saved 189 (0 new) | 206.61 / 62.33 | 19.52 GiB | **4,530,275 tok (4.32×)** | 69 s / 1.29 GiB then 15 s / 1.12 GiB |
| v18ctl6 21:32:34 | 310 s (`docker start`) | — | v18 residency (unchanged) | — | v18 2.50M (not re-logged this window) | — |

Neither boot ran a live 15–17 min tune. Cache file used: `…/103a/ed692e15…/autotune_configs.json` (the FAST copy). Rehome `hbm_layers=31 uva_layers=9`. Smoke both: models `dsv41-flash-uva`, ARITH 323, COUNT ok.

v18 restore after T3b5b: BOUND 21:51:34 after 290 s. `:30006` = `dsv41-vllm-v18-cgsizes-BOUND-REF` serving `dsv41-flash-uva`. T3b5 / T3b5b stopped-and-kept. Slow set back at canonical; `.from-T3b4` untouched; `.after-T3b5` kept. `bba7410c…` untouched. Nothing `docker rm`'d.

## 2. Windows (same instrument as T3b; C1 = knee r1/r2/knee6; C8, C16 = r1/r2)

| boot | autotune path | C1 | C4 r1/r2 (k6) | C8 | C12 r1/r2 | C16 | replay n=4 | Δ C1 vs v18ctl6 |
|---|---|---|---|---|---|---|---|---|
| T3b4 (ref, T3b README) | live tune, 15 min | **195.8 / 195.6 / 195.4** | 442 | 700 / 733 | 891 | 1011 / 1031 | 374 / 468 | +13.7% vs v18ctl5 |
| v18ctl5 (ref, T3b README) | — | 172.0 / 172.2 / 172.2 | 413 | 642 / 659 | 803 | 746 / 942 | 321 / 416 | — |
| **T3b5** 21:24 fresh `docker run`, FAST cache in place | **Loaded 189, 8 min, 0 new** | **187.8 / 188.2 / 188.0** | 373 / 423 (425) | 677 / 666 | 843 / 847 | 974 / 983 | 312 / 460 | **+9.6%** (three-C1 mean 188.02 vs 171.52) |
| v18ctl6 21:32 `docker start` | — | 171.5 / 171.6 / 171.5 | 414 / 414 (414) | 657 / 657 | 809 / 811 | 758 / 952 | 331 / 437 | — |
| **T3b5b** 21:43 fresh `docker run`, same FAST cache | **Loaded 189, 8 min, 0 new** | **155.3 / 187.8 / 187.9** | 335 / 418 (423) | 668 / 673 | 842 / 846 | 965 / 977 | 286 / 467 | r2/k6 = T3b5 (Δ C1 r2 0.21%, k6 0.06%); r1 collapsed (inner runs 186.5/124.2) |

C16 r1 is ~25% under r2 on v18ctl6 (758 vs 952); quote paired only.

C4/C12 vs the loaded-cache hole (T3b2/T3b3 C4 346/350, C12 672/669): T3b5 r1 C4 373 is still −9.9% vs v18ctl6 r1 414; T3b5 r2/k6 C4 423/425 = +2.2% / +2.7% vs ctl. T3b5 C12 843/847 vs ctl 809/811 (no −17% hole). T3b5b r1 C4 335 / C2 123 is the same cold-first-shape collapse as C1 r1, not a second window.

Fixture (tok/s · accept): v18ctl6 prose 202.4·0.349 / structured 234.3·0.525 / code 234.9·0.661 / shell 267.7·0.908 / tool_json 241.3·0.878 (wacc 0.628). T3b5 181.1·0.288 / 215.0·0.442 / 243.5·0.688 / 263.1·0.829 / 217.2·0.810 (wacc 0.582). T3b5b 180.3·0.288 / 213.8·0.442 / 240.5·0.688 / 260.2·0.829 / 219.3·0.810 (wacc 0.582). Acceptance matches T3b4's live tune, not v18.

Replay acc/step: T3b5 2.62/2.68; v18ctl6 2.70/2.59; T3b5b 2.56/2.64.

## 3. Greedy parity (`e2c_parity.py`, 18 prompts: 2 tool `agent-tool-*`, 16 text; seed 42, 256 tok)

Compared `content` + `tool_norm`.

| pair | exact | text-prompt mismatches (of 16) |
|---|---|---|
| **T3b5 vs T3b5b** (same cache, two boots) | **17/18** | **0** — only `agent-tool-1` |
| **T3b5 vs T3b4** (loaded FAST vs the live tune that saved it) | **17/18** | **0** — only `agent-tool-1` |
| T3b5b vs T3b4 | 17/18 | 0 — only `agent-tool-1` |
| T3b5 vs v18ctl6 | 2/18 | 16 (cross-image; not a bar) |

Text 16/16 identical T3b5 ↔ T3b4 ↔ T3b5b. Do not apply 18/18 vs v18.

## 4. Power (`nvidia-smi -i 1`; draw, cap, SM, mem, temp, clocks_event_reasons.active)

| line | draw W | cap W | SM MHz | mem MHz | temp C | throttle |
|---|---|---|---|---|---|---|
| power-pre T3b5 | 286.39 | 1300.00 | 2070 | 3996 | 38 | **0x0000000000000000** |
| power-post T3b5 | 565.71 | 1300.00 | 2070 | 3996 | 48 | 0x0 |
| power-ref / power-pre v18ctl6 | 207.14 | 1300.00 | 2070 | 3996 | 38 | 0x0 |
| power-post v18ctl6 | 609.24 | 1300.00 | 2070 | 3996 | 50 | 0x0 |
| power-pre T3b5b | 299.31 | 1300.00 | 2070 | 3996 | 40 | 0x0 |
| power-post T3b5b | 590.73 | 1300.00 | 2070 | 3996 | 49 | 0x0 |
| power-end (v18 restored) | 205.80 | 1300.00 | 2070 | 3996 | 38 | 0x0 |

No throttle reason ≠ `0x0`. Cap held 1300 W. First powered receipts on this lane.

## 5. Bars (report, do not promote)

- **Hash hit:** PASS (Loaded 189, 0 new, 480 s bind, both boots).
- **Residency / KV:** PASS (206.61/62.33 identical to v18; KV 4,530,275 on nightly).
- **CACHE-PIN CONFIRMED** (C1 two-window mean ≥ +10% vs v18ctl6 **and** C4 ≥ ctl −3% **and** T3b5 vs T3b5b within 2% at C1): **not met as written.** T3b5 three-C1 mean +9.6% (188.02 vs 171.52). C4 r1 −9.9% vs ctl (r2/k6 +2.2/+2.7%). T3b5 vs T3b5b C1 r2/k6 within 0.2%; T3b5b r1 155.3 is a collapsed inner pair.
- **LOAD-PATH TAX CONFIRMED** (C1 within ±2% of v18ctl6 with C4/C12 hole): **not met.** C1 +9.6%, not ±2%; C12 hole absent.

Measured C1 with the FAST set loaded is **188**, between the 172 slow-loaded boots and the 196 live tune that produced this file. Not promoted.

VERDICT T3b5: NEITHER FORK AS WRITTEN — hash-hit Loaded 189 (0 new, 480 s) both boots · FAST set pin sha256 5440f02d… · C1 T3b5 187.8/188.2/188.0 vs v18ctl6 171.5/171.6/171.5 (**+9.6%**, bar +10%) · T3b5b C1 r2/k6 187.8/187.9 (within 0.2% of T3b5); r1 155.3 (inner 186.5/124.2) · C4 r1 373 (−9.9% vs ctl 414); r2/k6 423/425 (no −15% hole) · C8 +1.5–3.0% · C16 paired r2 +3.2% (r1 +28% vs ctl's cold 758) · replay 312/460 vs 331/437 · parity T3b5 vs T3b5b 17/18 (text 16/16), T3b5 vs T3b4 17/18 (text 16/16) · power cap 1300 W, throttle 0x0, draw 207→566–609 W · rehome 206.61/62.33 · KV 4,530,275 · v18 restored 21:51 on :30006 · stopped-and-kept `…-T3b5-…-CACHEPIN-EXP` and `…-T3b5b-…-CACHEPIN-EXP` · `.from-T3b4` untouched, slow set canonical, `.after-T3b5` kept.
