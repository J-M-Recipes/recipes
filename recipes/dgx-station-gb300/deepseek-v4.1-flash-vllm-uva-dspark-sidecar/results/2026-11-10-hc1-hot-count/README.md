# Card HC1 — hot experts per layer 255 → 275 on Pg61 (2026-11-10): measured, not promoted

**Verdict: not a promotion candidate as registered.** With 275 hot experts per layer (109 cold on the RTX instead of
129) a 61K-token prompt reaches its first token 11–12 % sooner, but single-stream decode read 0.9796× the
controls against a pre-registered bar of 0.98×, and the KV pool halves. Pg61 keeps the h255 map.

One variable changed: the rowmap. Everything else is the Pg61 tree, launched by the unmodified `launch-pg61.sh`
(`docker inspect` Args and Env of the arm: [`raw/arm-args.json`](raw/arm-args.json), [`raw/arm-env.json`](raw/arm-env.json)).

## Arms

| arm | what | when (CST) |
|---|---|---|
| control A | live Pg61, no window | 05:19–05:21 |
| H = 275 | Pg61 tree + [`rowmap-mix-v1-h275.json`](rowmap-mix-v1-h275.json) + the one `config/recipe.env` rowmap line | boot 05:22:58 → up 05:29:07, cells to 05:31:41 |
| control A2 | Pg61 restored by `launch-pg61.sh up` | 05:38:56–05:41:23 |

Lane dark 05:21:27 → 05:38:56 (17 min 29 s; the arm served `:30006` for part of it). The p3 watchdog ran in ENFORCE on
every arm with no trip. Bars were written before the arm booted: [`PREREG.md`](PREREG.md).

## Results

| cell | control A | **H = 275** | control A2 |
|---|--:|--:|--:|
| C1 mixed prompts, tok/s (mean of 4 passes) | 412.59 | **404.15** | 412.53 |
| C1 set A / set B, tok/s | 414.5 / 410.6 | 397.8 / 410.5 | 414.3 / 410.7 |
| C1 ms per step | 7.637 | 7.674 | 7.64 |
| C1 tokens per step, set A / set B | 3.125 / 3.091 | 3.011 / 3.109 | 3.125 / 3.091 |
| C8 aggregate tok/s, 3 reps sorted | 891 / 1,027 / 1,099 | 881 / 1,038 / 1,090 | 844 / 1,084 / 1,104 |
| C32 aggregate tok/s, 3 reps sorted | 1,625 / 1,784 / 1,806 | 1,442 / 1,870 / 1,885 | 1,582 / 1,813 / 1,818 |
| first token, 7,621-token prompt, s (3 reps in order) | 0.710 / 0.186 / 0.182 | 3.405 / 0.168 / 0.168 | 0.710 / 0.184 / 0.185 |
| first token, 60,954-token prompt, s (3 reps in order) | 1.592 / 1.499 / 1.488 | **3.978 / 1.318 / 1.312** | 1.600 / 1.482 / 1.473 |
| tool gauntlet | 12/12 | 12/12 | 12/12 |
| KV pool at boot | 11,696,582 tokens (26.54 GiB, 11.15×) | **5,524,336 tokens** (12.54 GiB, 5.27×) | 11,696,582 tokens |
| RTX PRO 6000 memory used | 95,682 MiB (preflight) | 82,442 MiB | — |

Cells: `gt_mix_drive.py` from [jmeadlock/gb300-dsv41-decode](https://github.com/jmeadlock/gb300-dsv41-decode) @ `476176d`
(`pass` = 20 mixed prompts × 256 tokens, one stream; `conc` = 40 salted requests on 8 streams / 64 on 32, a new salt
per rep; `ttft` = `max_tokens 1`, a new salt per run), over loopback, driven by [`scripts/hc1-cells.sh`](scripts/hc1-cells.sh).
Every number is in [`throughput.csv`](throughput.csv) with its raw file.

## Bars

| # | bar (registered before the boot) | result |
|---|---|---|
| 1 | C1 mixed ≥ 0.98× the control mean | **MISS** — 0.9796× |
| 2 | a gain outside the A-vs-A2 spread on C32, C8 or the 64K first token | **PASS** — 64K first token -11.0 / -12.0 % vs A / A2 (reps 2–3; the controls differ by ~1 %) |
| 3 | no cell worse than the worse control beyond the spread | **MISS** — C1 set A 397.8 vs 414.3 tok/s |
| 4 | gauntlet 12/12, server alive, watchdog trip null | **PASS** |
| 5 | KV at boot ≥ 4,400,013 tokens | **PASS** — 5,524,336 |

## Reading it

* **Long prompts are where the cold experts cost.** The 7.6K first token moves -8.7 / -8.9 % and the 61K one
  -11.0 / -12.0 % (reps 2–3). The sidecar's own counters agree: its peer tier v2 call in the 8,192-row bucket
  averaged 5,368 µs on control A2 (7,029 rows mean) and 4,382 µs on the arm (6,730 rows mean).
* **Single-stream decode does not wait on the RTX.** At 6 rows (one stream, k = 5) the GB300's exposed wait per layer
  call was 3.6 µs on control A2 and 0.7 µs on the arm; distinct cold experts per call 2.47 vs 2.0. The
  step is 7.674 ms against 7.637 / 7.64 (+0.5 %).
* **The C1 miss is acceptance, on one prompt set.** Set A accepted 3.011 tokens per step on the arm against 3.125 on both
  controls, identically on both passes; set B accepted slightly more (3.109 vs 3.091). That is one deterministic
  sample per set. Whether it is a cost of the map or the arbitrary sign of a near-tie text divergence is **not
  resolved** here; the per-class temperature-1.0 three-seed instrument (Card G5's) would settle it.
* **C8 and C32 are inside the run-to-run spread.** Three reps per arm overlap on every arm.

## Not measured, and caveats

* First rep of the 7.6K and 61K prompts on the arm took 3.405 s and 3.978 s (controls 0.71 and 1.592). The arm ran
  on its own state directory seeded from Pg61's kernel caches; first-use compilation is the likely cause, not confirmed.
* One boot per arm, one morning. No teacher-forced, GSM8K, BFCL, needle or 1M-prompt run on the arm.
* H = 265 and H = 285 were not run.
* The arm's map keeps Garner's 255 hot experts and promotes 20 per layer from his cold list, preferring experts in our
  `v26ar` seed map; it was not re-calibrated on traffic.
* Control A's sidecar counters ([`raw/ctrlA-peer3_stats.json`](raw/ctrlA-peer3_stats.json)) cover the whole previous boot, not only these cells;
  the paired comparison uses control A2 and the arm, which saw the same cells from a fresh boot.

## Files

* [`PREREG.md`](PREREG.md), [`throughput.csv`](throughput.csv), [`rowmap-mix-v1-h275.json`](rowmap-mix-v1-h275.json) (sha256 `60c3108c08dd765d…`; derived from
  Garner McCloud's `rowmap-mix-v1-h255.json`, Apache-2.0, see the recipe's Credits).
* [`scripts/`](scripts/): the runner as it ran (box paths replaced by `$HOME`; it is a card runner for our box, not a
  standalone launcher), the cells, and the rowmap builder.
* [`raw/`](raw/): per-cell JSON and logs for the three arms, boot facts, sidecar counters, gauntlet summaries, the arm's
  server and sidecar logs, [`raw/runner.log`](raw/runner.log). LAN client addresses are replaced by `lan-client` and box paths by `~`;
  the gauntlet kit's per-task files are not vendored.
