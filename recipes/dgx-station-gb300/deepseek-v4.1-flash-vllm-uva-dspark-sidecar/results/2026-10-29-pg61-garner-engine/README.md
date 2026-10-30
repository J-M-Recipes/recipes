# 2026-10-29 — Pg61: Garner McCloud's DS4.1 engine under our policy, and its promotion to `:30006`

**Result:** Pg61 has served `:30006` on the reference box **since 2026-10-29 10:14pm CDT**. James approved the
promotion and asked that it stay **marked as still being tested**: a 48 h soak runs to **Saturday 2026-10-31 10:14pm CDT**,
and the recipe stays `experimental`. Rollback is one command back to h2-REF (stopped and kept).

**Pg61 is Garner McCloud's engine.** It runs [`garnermccloud/dgx-station-recipes`](https://github.com/garnermccloud/dgx-station-recipes)
@ [`8043a6811a5feda70fdfb6e827eebb2ec25c65d1`](https://github.com/garnermccloud/dgx-station-recipes/tree/8043a6811a5feda70fdfb6e827eebb2ec25c65d1/deepseek-v4.1-flash)
(`deepseek-v4.1-flash/`, MIT) on stock `vllm/vllm-openai:v0.31.0`, started by his `serve/serve.sh`. His work includes
the 29-component `DS41_FASTPATH`, peer tier v3, the MegaMoE scheduling fork, the h255 rowmap, his FlashInfer tune file
and his overlays. Our layer is small and listed in full:

- five patches ([`../../patches/garner-8043a681/`](../../patches/garner-8043a681/README.md)): six policy hooks that
  do nothing unless their variable is set, plus the vLLM #61161 seed mix in his three fused draft samplers;
- a two-file vLLM overlay, byte-identical to what h2-REF mounts: #61161 plus the #60945 guard;
- a fail-closed watchdog for peer tier v3 ([`../../scripts/sidecar_watchdog_p3.py`](../../scripts/sidecar_watchdog_p3.py)),
  with `PEER3_SPINS=1000000000000`;
- our serving policy. Stock weights; `fp8_ds_mla` KV (his release uses `nvfp4_ds_mla`); the default indexer; a
  1,048,576 context (his: 524,288); a 48 GB host KV tier; no sampling override; no reasoning-effort default; no API
  key; 24 of his 29 components.

The card write-ups below are scrubbed copies of the research-tree READMEs (`cards/`); the raw receipts are under `raw/`.
**Not vendored:** Garner's tree (link above), the private per-class prompts, BFCL data and per-item BFCL rows,
replay transcripts, and any key file.

## Cards

| card | question | verdict | write-up | receipts |
|---|---|---|---|---|
| **G0** | Does his release, as shipped, reproduce on our GB300 + RTX PRO 6000 **Max-Q (300 W)**? | **REPRODUCES**: 135.9 steps/s on his fixed coding prompt vs his published 135.4 (731.3 vs 733.6 tok/s). Nothing adopted | [`cards/G0/`](cards/G0/README.md) | [`raw/G0/`](raw/G0/) |
| **G1** | His tree under our policy (P), and P + his KV/indexer config (K), bracketed by h2-REF (A → P/K → A2) | **INCOMPLETE, CRASH**: both P and K died with `CUDA error: an illegal memory access` on a 32-request × 4-token step. Single-stream cells before the crash: P 1.26× h2-REF steps/s | [`cards/G1/`](cards/G1/README.md) | [`raw/G1/`](raw/G1/) |
| **G1b–G1e** | Bisect the crash: context length, component list, numerics-changing components | 6 deaths in 4 configs; turning off an inert component (`lowb_aux`) was enough to flip survival → **layout-dependent** | [`cards/G1f/`](cards/G1f/README.md) | [`raw/G1-bisect/`](raw/G1-bisect/) (runner logs, crash excerpts) |
| **G1f** | Cause and fix | **Cause = vLLM #60945, unguarded in stock v0.31.0.** The 3-line guard alone fixes the config that died (V1: 3 × 60 s C32 clean). **Pg** = P + guard passes every cell G1 owed: per-class C1 mean 1.33× and C8 1.10× h2-REF; TF at the floor; gauntlet 12/12; 1M needle PASS | [`cards/G1f/`](cards/G1f/README.md) | [`raw/G1f/`](raw/G1f/) |
| **G2** | Promotion-tier gates on Pg (T0/T2/T1, BFCL held-out, replay, prefill) | **PASS on every quality gate**; BFCL held-out 78.26 % vs v27 77.27 % (p = 0.011, better). **One speed regression: prefill −8 to −10 %** | [`cards/G2/`](cards/G2/README.md) | [`raw/G2/`](raw/G2/) |
| **G3-desk** | Does v0.31.0 carry #61161? Does his peer tier v3 fail closed? | **#61161 absent** in v0.31.0 and in his fused drafters (every T > 0 draw is biased). **A v3 wait timeout skips the cold-expert add silently** (wrong tokens). Fix = `PEER3_SPINS` ≥ 1e12 + a watchdog that refuses to arm without it. Desk-built, 13/13 unit tests | [`cards/G3-desk/`](cards/G3-desk/README.md) | — (desk) |
| **G3 / G3b** | #61161 port exactness on a GPU; the watchdog's trip path under fault injection | **Both PASS.** Patched kernels equal the eager reference at every step (w6 30/30, w5 120/120 per step; stock matches only step 0). G3b paused the sidecar under 4 streams: the last tokens came 0.10 s after the pause and none after the stall, the watchdog **TRIPPED after ~3 s** and the server was killed. Pg61 = Pg + port + spins + watchdog becomes the candidate | [`cards/G3/`](cards/G3/README.md) | [`raw/G3/`](raw/G3/) |
| **G4** | T3 promotion bench, Pg61 **booted by the published launcher** vs h2-REF in the same window | **PASS on all six pre-registered bars** ([PREREG](cards/G4/PREREG.md)). `docker inspect` args identical to G3; needle 500/500; GSM8K-1319 and MMLU-Pro-1000 no paired loss (MMLU-Pro +2.4 pt, p < 0.001); T1 C1 1.21–1.31×, C4 1.12–1.18× (the card README rounds 757.4/678.8 to 1.11×); 0 trips | [`cards/G4/`](cards/G4/README.md) | [`raw/G4/`](raw/G4/) |
| **G5** | Does the #61161 port cost C1 at T = 1? What is the headline vs h2-REF? | **Port: no effect** (Pg61/Pg = 1.012). **Headline: Pg61 = 1.31× h2-REF**, per-class C1, T = 1.0, mean of 3 seeds, 1.26–1.36× per class ([PREREG](cards/G5/PREREG.md)) | [`cards/G5/`](cards/G5/README.md) | [`raw/G5/`](raw/G5/) |
| **cutover** | Promotion | h2-REF pair stopped (kept) 10:07:54pm; launcher `verify OK` 10:14:12pm; watchdog armed 10:14:15pm; smoke OK and **serving 10:14:16pm CDT** | — | [`promote/`](promote/) |

## Headline (Card G5): per-class C1, temperature 1.0, mean of 3 seeds

`perclass_decode_t.py` ran 40 private prompts per class, 256 forced tokens, `--temperature 1.0 --top-p 1.0`, with
seed bases 42/43/44. Each arm had one boot in the same window (Pg61 → Pg → h2-REF), plus one unscored natural-length
warm pass. acceptance_len = 1 + Σaccepted / Σdraft-steps over the three seeds, from the instrument's spec-decode
counter deltas (bonus token included).

| class | Pg61 tok/s | Pg tok/s | h2-REF tok/s | Pg61 / Pg | **Pg61 / h2-REF** | acceptance_len Pg61 / Pg / h2-REF |
|---|--:|--:|--:|--:|--:|---|
| prose | 367.2 | 367.2 | 289.7 | 1.000 | **1.27×** | 3.02 / 3.03 / 2.96 |
| structured | 396.9 | 401.2 | 314.0 | 0.989 | **1.26×** | 3.27 / 3.33 / 3.21 |
| code | 494.9 | 482.8 | 368.8 | 1.025 | **1.34×** | 4.30 / 4.19 / 3.82 |
| shell_ops | 530.4 | 532.1 | 388.9 | 0.997 | **1.36×** | 4.65 / 4.68 / 4.03 |
| tool_json | 463.8 | 442.8 | 356.5 | 1.047 | **1.30×** | 3.97 / 3.79 / 3.68 |
| **mean** | | | | **1.012** | **1.31×** | |

All 45 cells ran with 0 errors, 0 foreign tokens and `max running 1/1`, and the p3 watchdog never tripped. The card
README's "tokens/step" column is the instrument's `acc/step` (bonus token excluded); the table above adds 1.

## T3 (Card G4): Pg61 vs h2-REF, same window, M4 over the LAN, C ≤ 8

| bar | Pg61 | h2-REF | verdict |
|---|---|---|---|
| 5 launcher reproduces G3 | fresh `build` → `up`: hashes OK, `docker inspect` Args **identical** to G3 ([`args-diff.txt`](raw/G4/box/Pg61/args-diff.txt)), 24 components ACTIVE, KV 11,696,582 | — | PASS |
| 1 needle-checksum 500 (C4) | EXACT 500/500 | EXACT 500/500 | PASS |
| 2 GSM8K-1319 (C4) | 96.97 % (1279); vs v25m baseline +0.15 pp, McNemar p = 0.73 (3 lost / 5 gained) | 96.89 % (1278); paired 4 lost / 5 gained, p = 1.0 | PASS |
| 3 MMLU-Pro-1000 stratified (C4) | **87.7 % (877)**; 14 runs hit the 131K max_tokens | 85.3 % (853); 20 hit max_tokens. Paired **10 lost / 34 gained, p < 0.001** | PASS (better; cause not investigated) |
| 4 T1 C1 @ 0 / 32K / 128K | 324.7 / 322.1 / 320.5 tok/s | 247.3 / 266.4 / 250.4 | PASS: 1.31× / 1.21× / 1.28× |
| 4 T1 C4 @ 0 / 32K / 128K | 773.7 / 757.4 / 803.8 | 689.7 / 678.8 / 678.5 | PASS: 1.12× / 1.12× / 1.18× |
| T1 C8 (no bar) | 1,287.8 / 1,258.9 / 1,262.0 | 1,124.4 / 1,127.1 / 1,124.1 | 1.15× / 1.12× / 1.12× |
| 6 watchdog / survival | 0 trips; server up after the suite | — | PASS |
| tool-eval hardmode ×2 (no bar) | 89 (163/184) · 91 (168/184) | 89 (163, safety-capped) · 93 (171) | same band; one run cannot rank |

T1 is llm-inference-bench 0.7.7 synthetic filler at the server's default temperature. Per the repo's labelling
rule, it never replaces the per-class headline.

## Known costs, measured

| cost | evidence |
|---|---|
| **Prefill is slower.** His `bench_full.py` prefill, uncached, median of 3: 14K / 57K / 114K = 39.9K / 41.6K / 41.0K tok/s vs h2-REF same day 44.0K / 44.4K / 44.6K → **−8 to −10 %**. The G4 T1 scout prefills (N = 1 each) read **−10 to −17 %** (8K 38,865 vs 42,968; 32K 39,954 vs 47,218; 64K 40,399 vs 47,054; 128K 39,775 vs 48,062) | [`raw/G2/box/*/prefill.md`](raw/G2/box/), [`raw/G4/m4/*/t3-t1.stdout.txt`](raw/G4/m4/) |
| **1M first token 36.54 s** on 999,034 tokens (27.3K tok/s), vs **33.2 s** for the v25 base launch (2026-10-22, a different day and launch). h2-REF's own AP2 needle was 900,372 tokens in 31.17 s (28.9K tok/s) | [`raw/G1f/Pg/onem-needle.json`](raw/G1f/Pg/onem-needle.json) |
| Probable cause (not measured): his h255 rowmap puts **129 cold experts per layer** on the RTX (vs our 89), and T > 64 still uses peer tier v2. A rowmap sweep (h270 / h285 under our policy) is planned after promotion, at a KV-token cost | [`cards/G2/`](cards/G2/README.md) |

## Not exercised

- **Temperature > 0 quality.** Every quality gate here is T = 0 or teacher-forced. The #61161 port is proven exact
  at the kernel level (G3) and costs no C1 (G5). There is still no T > 0 quality score: no full-vocabulary KLD over
  forced tokens and no seeded GSM8K at T = 1.
- **The 48 h soak.** It is running now, from 10:14pm CDT Thursday 2026-10-29 to 10:14pm CDT Saturday 2026-10-31. Any
  restore or card resets the clock.
- **Decode beyond 128K**, except the one G1f 1M needle (on Pg, before the port; found, post-probe OK).
- **The watchdog fix after G3b.** G3b's trip ran the watchdog *before* a log-ordering fix (log the server kill first,
  unpause, then stop the sidecar). The fixed file (`812c47c0`, published as `scripts/sidecar_watchdog_p3.py` with
  only its default paths scrubbed) has passed the 13 unit tests. It ran armed through G4, G5 and the cutover with 0
  trips, but it has **not itself fired under fault injection on a GPU**.
- **libw6mk.so build hash.** The launcher's fresh build produces `libw6mk.so` sha256 `b5416822…`. G3 tested
  `b072ee09…`, which was built by hand from the same patched `w6mk.cu`. The host-side bytes differ (nvcc embeds build
  paths); the **device code does not**: [`libw6mk-cubin-identity.txt`](libw6mk-cubin-identity.txt) (`cuobjdump -xelf`,
  added 2026-10-29 after the first draft of this bundle) shows both builds carry the same two sm_103a cubins
  (`11b55da2…`, `4d07c093…`); Garner's shipped `ab677a99…` differs only in the patched kernel's cubin (`d45e7760…`).
  G4 and G5 measured the `b5416822` build, which is what serves.

## Promotion

[`promote/cutover-20261029.log`](promote/cutover-20261029.log) records `scripts/cutover-h2ref-to-pg61.sh`. It
checked preconditions (h2-REF serving, no card runner, `BUILD-SHA256SUMS` intact) and raised MAINTENANCE. At 10:07:54pm
it stopped and kept the h2-REF pair (the RTX container needed SIGKILL after the 60 s stop timeout,
`Exited (137)`). It dropped caches and ran `launch-pg61.sh up`: sidecar `prepared 40 layers in 58s`, KV 11,696,582
tokens, `verify OK` at 10:14:12pm, watchdog ENFORCE at 10:14:15pm. It then ran the smoke (OK, 10:14:16pm), froze the
build with `chmod a-w`, cleared MAINTENANCE and wrote an owner line. The freeze printed one `Operation not
permitted` on `libw6mk.so`: the nvcc build runs inside a container, so that file is root-owned. Fixed by hand at
about 10:45pm with `sudo chmod a-w` (0 writable files left under `tree/` and `overlay-vllm/`); the lane check's
`BUILD-SHA256SUMS -c` covers it either way. The published cutover script should `sudo chmod` that file.

A read-only snapshot at 2026-10-30T03:20Z ([`promote/live-check-2026-10-30T0320Z.txt`](promote/live-check-2026-10-30T0320Z.txt))
shows:

- `ds41` and `ds41-sidecar` up on `sha256:c1c9f6fd…`, the h2-REF pair exited and kept;
- `PEER3_SPINS=1000000000000` in the server env;
- one ENFORCE watchdog with `trip: null` and v2/v3 timeouts 0;
- `BUILD-SHA256SUMS` OK (114 files), `/v1/models` bound and 0 Xid since the cutover.

An M4 cron runs [`../../scripts/pg61_lane_check.sh`](../../scripts/pg61_lane_check.sh) every 10 min. It prints
nothing while the lane is healthy.

## Files

| path | what |
|---|---|
| `throughput.csv` | every speed cell × arm × card (G0 / G1 / G1f / G3 `bench_full`, G2 prefill, G4 T1, G5 per-class per seed and mean, ratios), built by the copy-and-scrub step from the raw files it names |
| `image.txt` | image of record: `vllm/vllm-openai:v0.31.0@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3` (pullable release; repo digest = image id on the box) |
| `cards/` | scrubbed copies of PLAN.md and the G0, G1, G1f (covers G1b–G1f), G2, G3-desk, G3, G4 (+PREREG) and G5 (+PREREG) READMEs, plus the issue drafts sent to Garner ([issue #1](https://github.com/garnermccloud/dgx-station-recipes/issues/1) and two comments, filed as jmeadlock) |
| `overlay-desk-check.txt` | the published #61161 + #60945 patches applied to the stock files reproduce the two overlay files byte for byte |
| `patch-desk-check.txt` | the five Garner-tree patches applied to a clean checkout at 8043a681 reproduce the frozen build tree on the box (5/5 sha1) |
| `raw/G0/`, `raw/G1/`, `raw/G1-bisect/`, `raw/G1f/` | `bench_full` md/json per arm, per-class tables, TF compares, gauntlet summaries, 1M needle, C32 windows, runner logs, crash excerpts |
| `raw/G2/` | box: prefill per arm, replay **per-pass counts only** (`replay-counts.log`; the transcript turns are withheld). m4: T0, T1, T2 (gsm8k/needle/tool-eval `.json.gz`), BFCL held-out summary + log (per-item rows not published), engine-load |
| `raw/G3/` | U: #61161 unit test JSON/logs, including the first harness run (`unit-v1bad-*`, which read the wrong rows and was kept). Pg61 cells, both fault injections (`fault-client.json`, `wdp3.*`), and m4 T2/BFCL summaries |
| `raw/G4/` | box: the launcher's `launch-up.log`, args/env/args-diff, gzipped boot log, watchdog status. m4: per-arm T0/T1, gzipped T3 JSONs, stdout tables, engine-load |
| `raw/G5/` | per-arm per-class md + JSON (`rows` set to null with a `rows_withheld` note; prompt file sha1 `618898aac432…` is private), cell logs, runner log |
| `promote/` | the cutover log, with one `serve.sh` curl-hint line removed because it names a key-file path, and the live snapshot |

**Scrub:** box home-directory paths → `$HOME`, desk home paths → `~`, LAN addresses → `<gb300>` / `<m4>` / `<lan>`.
Authorization headers and key-file names are redacted. Every per-request `rows` array is withheld. JSON was
re-serialised with `indent=1`, and large JSONs are `gzip -9n`. A grep over the tree, and over `gzip -dc` of every
`.gz`, for LAN prefixes, home-directory paths, auth headers and key-file names returns nothing.
