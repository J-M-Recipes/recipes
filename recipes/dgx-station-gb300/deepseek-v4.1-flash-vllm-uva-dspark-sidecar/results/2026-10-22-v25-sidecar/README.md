# 2026-10-22-v25-sidecar — what is in this bundle

The measurement round that produced the sidecar lane: bring-up gate (SC0), bracketed speed (SC1),
quality gates (SC2), cold-expert placement (SC3), 1M fit and power (SC5). Everything was measured on
**one** DGX Station GB300 + RTX PRO 6000 Max-Q, 1017 weights, image `sha256:3af6a45b…` (nightly-af7f9488),
original-el8 hook @87af662, uniform `rowmap-static-v1.json` (295 hot / 89 cold).

| file | what it is |
|---|---|
| `throughput.csv` | every speed cell quoted in the README, with its arm (v23 control / sidecar) and the receipt it came from |
| `power-by-phase.csv` | GB300 GPU, Station module, `sw_power_cap` and RTX draw by phase, sidecar vs the v23 incumbent |
| `system.json` | `scripts/collect_system_snapshot.sh` on the Station (both GPUs, caps, driver, kernel, OS) |
| `raw/sc0-cosine-gate.md` | SC0: bring-up gate on real prompts — 1,720 post-startup checks, 1,341 with ≥32 cold rows, 40/40 layers, min cos **0.99906**, 0 timeouts |
| `raw/sc1-speed-bracketed.md` | SC1: same-window CTL1 → CAND1 → CAND2 → CTL2 (plus an M3G Grace-only arm), real-text and knee ladders |
| `raw/sc2-quality-gates.md` | SC2: the 12 pre-registered daily-driver gates (TF flips, BFCL, gauntlet, GSM8K, needle ladder, per-class decode, tool emission, replay, soak, 0 timeouts) |
| `raw/sc3-placement.md` | SC3/SC3b: 285/99 nested map vs 295/89 — the 285 map FAILED its pre-registered cosine bar, so 295/89 serves |
| `raw/sc5-1m-and-power.md` | SC5: 1M-token fit, needle, cold 128K prefill, C16 after 1M, plus the per-phase power table and the v0.31.0 candidate arm |
| `raw/soak-and-watchdog.md` | the fail-closed guard: why a timeout is a correctness bug, the fault test, the live counter read, the soak contract |
| `raw/v25-boot-facts.txt` | the serving boot's own lines (peer mode, rowmap, KV tokens, replay line, sidecar prep) |
| `raw/weights-1017-receipt.txt` | checkpoint identity: revision, shards, total bytes, config/index sha256 |
| `raw/watchdog-status.json` | one watchdog status read from the live lane |

Numbers are copied from these files, not from memory. Where a cell was measured on a different harness it
says so in `throughput.csv` (`RT-C*` = the 8.2K-in/1,024-out real-text harness; `knee-*` = the agent-text
knee harness; `class-*` = the 40-prompt-per-class fixture).

Known gaps, stated so nobody reads more into the bundle than it holds:

- **Per-class speculative acceptance is not in this round.** Acceptance is reported for the real-text C1 cell
  (2.96 tokens per verify step). Per-class acceptance on the sidecar image was not instrumented.
- **The 285/99 map's speed is unmeasured** (its gate failed first, so the placement arms were skipped).
- **Power bars were set too tight for this chassis** — the v23 incumbent fails them too in prefill. The
  numbers are reported; the bars are recorded in the recipe's limits as a correction owed to the platform.
- **Tool-emission counts are low-power** (a temp-0 replay turns 15/14 vs 15/15 on the control; the earlier
  12/24 flag did not reproduce).
