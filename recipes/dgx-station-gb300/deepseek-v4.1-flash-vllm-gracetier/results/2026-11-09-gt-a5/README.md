# results/2026-11-09-gt-a5

Receipts for the gracetier recipe, all measured on the reference DGX Station GB300 on 2026-11-09 (CST). Everything
under `raw/` is a copy of a file from `jmeadlock/gb300-dsv41-decode` at `fadc3d2` (`docs/<card>/`), with three scrubs
applied by a script: box home paths become `~`, one LAN scraper address in server logs becomes `<lan-host>`, and the
launcher's generated API-key line is redacted. Boot logs are included as the `DS41_FASTPATH` / KV lines only (the full
logs are in the repo). `throughput.csv` was generated from the cell JSONs by the same script. `system.json` is
`scripts/collect_system_snapshot.sh` on the box that evening, hostname replaced.

| dir | card | what it holds |
|---|---|---|
| `raw/a5/` | A5, 22:11-23:00 | the three-way: per-cell JSON for gracetier (`a5-gracetier-20261109-221120-*`), the October recipe (`a5-v28-*`) and Pg61 (`a5-pg61-*`); cells `warm-a/b`, `c1-a`, `c1-b`, `c8`, `c32`, `c1-a2`; `a5-table.json` is the extracted table. Second gracetier window (`-224750`): TTFT sweep and tool gauntlet; Pg61's TTFT and gauntlet from the live lane. Driver and card logs, audits, boot lines. |
| `raw/gt1/` | GT1, 14:26-14:39 | first boot on the static 255-hot map: boot gate, greedy parity against Pg61, step cell, 32-stream smoke, TTFT |
| `raw/gt1/a2/` | A2, 14:42-14:55 | nsys kernel anatomy of the gracetier step (184 steps) |
| `raw/gt3/` | GT3, 18:58-19:13 | adaptive residency: static control, adaptation on the lab fixture, held-out and adaptive step cells, parity captures, audits, restore, 32-stream smoke, telemetry |
| `raw/gt4/` | GT4, 19:53-20:09 | general prompts (sets A / B): static, adapting, first sight, interleaved, restore; Pg61 on the same passes (`gt4-pg61-*`) |
| `raw/a4-h275/` | A4, 21:09-21:25 | the same passes with 275 hot experts per layer, plus the hysteresis 3.0 cells (`m6a`, `m6b`) |

Corrections recorded in the repo ledger that these receipts settle: the first A5 gauntlet attempt on both arms failed
on the runner (it pre-created the output directory); the gauntlet results quoted are from the re-runs. The cold-share
figures in the telemetry count uncached prefill picks on first-sight passes, so compare ms/step across cells, not
cold share between a first-sight and a repeated pass.
