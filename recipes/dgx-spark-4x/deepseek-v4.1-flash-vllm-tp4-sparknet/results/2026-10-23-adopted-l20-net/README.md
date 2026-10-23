# results/2026-10-23-adopted-l20-net — the adopted configuration, measured

Boot 2026-10-23 10:54:38 UTC → HTTP ready 10:56:19 UTC (101 s), on Sparks 5–8. Config: KV 10.5 GiB/rank,
32 seats, `--long-prefill-token-threshold 3856`, late admission at layer 20 (four-file overlay), sparknet 0.5.0
(mounted over the image's 0.3.0).

## What is in here

| file | what it is |
|---|---|
| `summary.json`, `verdict.json` | the arm's row: headline numbers and every needle verdict |
| `bench-L20-NET-adopted-main.json` | v41bench v1, C1–C6: aggregate and per-stream tok/s, mean TTFT per level |
| `bench-L20-NET-adopted-c1.json` | the C1 pass on its own, 8-category mix, T=0, thinking off |
| `concurrency.json` | streaming short-chat ladder 1/4/8/16/24/32 — a **different instrument**, not comparable to C1–C6 |
| `mixed-warm-cache.json` | 5 samples of a guest turn over a 68K resident prefix + 1.5K suffix (`mixed_warm.py`) |
| `teacher-forced-verdict.json` | teacher-forced Δlogprob vs the kept 8 GiB control, 45 segments / 5,485 positions |
| `gates-functional.json` | /v1/models, exact-reply smoke, get_weather tool round, preemption metric |
| `greedy40.json` | 40-prompt greedy raw-completion probe, byte-compared to the control |
| `launch-command.txt` | the exact rendered `docker run` this boot used |
| `render-diff-spark5.json` | factory render vs the durable config: what the boot actually changed |
| `orchestrator.log` | boot timeline, hygiene checks, battery phases |

`scripts/verify_launcher.py` re-derives the launcher from the recipe and diffs it against the live container
(80 args, 103 env pairs, 8 mounts on ranks 1 and 3).

## Headline

- C1 **105.21** per stream / **93.89** aggregate; C4 233.16; C6 291.75; mean TTFT 0.158 / 0.231 / 0.286 s.
- 93K cold prefill **5,338.1** tok/s at 32 seats / threshold 3856.
- Warm-cache guest turn **2.147 s** median (from ~3.14 s on the 8 GiB control).
- Guest turn behind a 413K **cold** prefill **1.196 s** with L20 vs 1.885 s without.
- KV pool **8,580,566** tokens (8.18× at 1M).
- qeval 71/75, greedy 40/40 identical, 0 preemptions, teacher-forced **PASS 0.0**.

## Second boot (11:55 UTC, same launch line)

A deliberate stop/start cycle produced an independent boot of the identical configuration, which is what lets the
recipe say "two boots agree" rather than "one boot said":

| | boot 1 (10:54) | boot 2 (11:55) | spread |
|---|---:|---:|---:|
| C1 aggregate tok/s | 93.89 | 94.34 | +0.5% |
| C4 / C6 aggregate | 233.16 / 291.75 | 234.55 / 290.15 | +0.6% / −0.5% |
| 93K cold prefill | 5,338.1 | 5,372.0 | +0.6% |
| KV pool | 8,580,566 | 8,580,566 | — |
| qeval | 71 | 71 | — |
| boot s | 101 | 101 | — |
| MemAvailable floor GiB | 20.79 | 21.08 | — |
| preemptions | 0 | 0 | — |
| 413K mixed short-turn TTFT | 0.93–1.49 s (11 samples, L20) | 0.901 / 1.352 / 0.935 / 1.262 s | reproduces |

Receipts: `boot2-summary.json`, `boot2-v41bench-main.json`, `boot2-concurrency.json`, `boot2-greedy40.json`,
`boot2-gates-functional.json`, `boot2-verdict.json`.

Note the pattern worth remembering: C1 repeats to ~0.6% across boots, which is why only paired C1 claims are quoted
here. C4 and above moved by more than their own CI in earlier A/A work, so a single boot's C4+ number is a sample,
not a measurement.

## What this round did NOT measure

- No needles at 300K on this config (the 3/3 PASS is the 2026-10-20 shape).
- No interleaved C4+ harness, so the C8 TTFT flag at layer 20 (0.463 s vs 0.359–0.364 s) is unresolved.
- No T>0 sampling comparison: teacher forcing holds the token stream fixed by construction.
- The `+1.37%` C1 on this boot vs the 92.62 control is larger than L20 alone (+0.08%) plus sparknet alone (+0.81%).
  Different boots, so it is encouraging, not established additivity.
