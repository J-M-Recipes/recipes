# Pg62 T2/T3 quality suite on the live lane (2026-11-15): all six pre-registered bars PASS

Run from the M4 over the LAN at concurrency ≤ 4 against the serving Pg62 lane, 06:53:58–07:50:25 CST, inside its 48 h
soak (nothing was started or stopped on the box). Bars were written before the first request:
[`PREREG.md`](PREREG.md) (sha256 `205f2a4c…`, logged on the START line of [`campaign.log`](campaign.log)).

**The reference is Pg61's stored per-item results from 2026-10-29 (Cards G3 and G4), paired by item id.** It is a
cross-day comparison, not a same-window control: Pg61 cannot be re-run without ending the soak. Pg62 changes
decode-attention bytes on some calls relative to Pg61, so near-tie items are expected to move both ways.

| bar | Pg62 | Pg61 reference | verdict |
|---|---|---|---|
| T0: model id, exact-word chat, one parsed tool call | 3/3 | — | PASS |
| needle-checksum 500 (C4) | EXACT 500/500 | EXACT 500/500 | PASS |
| GSM8K-1319 (C4), paired | 1284/1319 = 97.35 % | 1279/1319 = 96.97 % | PASS: +0.38 pt, 5 lost / 10 gained, McNemar p = 0.30 |
| BFCL held-out 1,311 (CONC 4), paired; bar ≥ −1.0 pt | 1024/1311 = 78.11 % | 1026/1311 = 78.26 % | PASS: −0.15 pt, 17 lost / 15 gained, p = 0.86 |
| MMLU-Pro-1000 (C4), paired | 871/1000 = 87.1 % | 877/1000 = 87.7 % | PASS: −0.6 pt, 19 lost / 13 gained, p = 0.38 (no significant loss) |
| p3 watchdog after the suite | trip null, 0 timeouts, both containers up | — | PASS |
| tool-eval hardmode ×1 (no bar) | 93/100 (172/184) | 89 and 91 | reported |

None of the three paired differences is distinguishable from item-level noise. They do not show Pg62 is better or
worse than Pg61; they show no measured loss at these sample sizes.

## What this does not cover

- Temperature > 0 quality (everything here is T = 0), the teacher-forced bracket and the gauntlet were not run on Pg62.
- The reference is from another day and another boot; there is no same-window control and no Pg62 self-repeat to
  size the flip floor.
- Contexts beyond the needle profile's. Long-context evidence for Pg62 is the byte-equality shadow to 571K tokens
  in [`../2026-11-14-pg62-w1-exact/`](../2026-11-14-pg62-w1-exact/README.md).
- The gate stamps stay unstamped until the 48 h soak completes (Monday 2026-11-16 11:11 PM CST).

## Files

[`PREREG.md`](PREREG.md) · [`run_all.sh`](run_all.sh) · [`command.txt`](command.txt) · [`campaign.log`](campaign.log) ·
[`analysis.txt`](analysis.txt) · [`engine-load.tsv`](engine-load.tsv) · [`lane-after.txt`](lane-after.txt) ·
`t0.json` · `t3-needle.*` · `t3-gsm8k1319.*` · `t3-mmlupro1000.*` · `t3-tooleval.*` · `bfcl-pg62-heldout*` ·
[`facts.txt`](facts.txt) (grader and baseline hashes). Box paths and LAN addresses are scrubbed.
