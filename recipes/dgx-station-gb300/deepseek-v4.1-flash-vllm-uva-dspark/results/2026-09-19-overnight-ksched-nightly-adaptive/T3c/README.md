# T3c — nightly `dee37d89`, hook **OFF**, `--cpu-offload-gb 60`, `--kv-cache-dtype fp8_ds_mla` — the flag-only Station row: 91.3 tok/s C1 live / 90.5 loaded; fp8 KV restores DSpark acceptance hook-off too

Purpose: every flag in the proposed `vllm-project/recipes` `dgx_station_gb300` row for DeepSeek-V4.1-Flash must be measured with no out-of-tree patch. T3 (2026-09-19) was the hook-off nightly on its default `nvfp4_ds_mla` KV; T3b6 showed fp8 KV recovers draft acceptance hook-**on**. T3c closes the gap: hook-off + fp8.

Runner: `../t3c-nohook-fp8kv-runner-2026-09-20.sh` (Milo, nohup on the box, 05:22–07:26 CDT 2026-09-20). Verdict read from receipts by Milo (Hermes `milo` profile, claude-fable-5-1 via anthropic). Receipts: `receipts/` (knee/knee6/agentfix/replayc JSON, window logs, boot/smoke excerpts, parity captures, power lines, cache-dir snapshots, the new autotune set with sha256).

Windows, same box, in order: **T3c** (fresh `docker run`, live tune) → **T3rs** (`docker start` of the kept T3 container — hook-off nvfp4, cache loaded: the direct peer) → **v18ctl8** (reference) → **T3cb** (fresh `docker run` loading T3c's saved set — the servable configuration). `run_window_T1T2.sh` each: knee ×2, knee6, `agent_fixture_o`, replay n=4 ×2, metrics; then `e2c_parity.py` (18 prompts).

## Boot facts

| boot | bind | autotune | KV | graphs |
|---|---|---|---|---|
| T3c 05:22 | **82 min** (4930 s) | new hash `7dbc24b6…` (fp8 KV changes the hook-off kernel-shape hash; nvfp4 hook-off was `bba7410c…`); live tune 05:29–06:44, `Saved 189` | `fp8_ds_mla` · **3,092,486 tok (2.95× at 1M)** | 67 s / 1.21 GiB + 10 s / 1.03 GiB |
| T3rs 06:50 | 270 s (`docker start`) | `bba7410c…` loaded | `nvfp4_ds_mla` · 4,405,597 tok (4.20×) | — |
| v18ctl8 07:01 | 300 s | — | 0909 fp8 · 2.50M (not re-logged) | — |
| T3cb 07:08 | **450 s** | `7dbc24b6…` **Loaded 189**, `Saved` 0 new | `fp8_ds_mla` · 3,092,486 tok | 68 s / 1.21 GiB + 12 s / 1.03 GiB |

Both fp8 boots: `Total CPU offloaded parameters: 61.17` GiB (positional off60), Engram 2 × 94.42 GiB pinned. Smoke 5/5 on both (arith, count, prose, tool call parsed, think parsed). Cache dirs 6 → 7 (one new; nothing moved or removed). New set: `receipts/autotune_configs-nohook-fp8kv-7dbc24b6-from-T3c.json`, sha256 `2b0cf542…`.

## Knee (agg tok/s; r1 / r2 / k6 where all three ran)

| | C1 | C2 | C4 | C8 | C12 | C16 (r2 / k6) | replay n=4 (r1 / r2) |
|---|--:|--:|--:|--:|--:|--:|--:|
| **T3c** hook-off fp8, live tune | **91.3 / 91.4 / 91.2** | 125–126 | 170–171 | 305 / 307 / 306 | 384–385 | 427† / 431 | 207 / 236 |
| T3rs hook-off nvfp4, loaded | 95.7 / 95.6 / 95.6 | 126 | 173–175 | 305 / 307 / 307 | 362 | 415 / 415 | 189 / 230 |
| v18ctl8 (0909 + hook, fp8) | 171.0 / 171.3 / 171.3 | 272–274 | 412–413 | 656 / 659 / 657 | 800–811 | 947 / 945 | 327 / 451 |
| **T3cb** hook-off fp8, loaded | **90.5 / 90.5 / 90.6** | 124 | 168 | 301 / 302 / 303 | 377–378 | 423 / 416 | 194 / 228 |

†T3c r2 C16 was 378 agg because run 1 of the pair collapsed to 329 (run 2: 427); r1 and k6 agree at 424–431. Quote C16 as 424–431 for T3c.

## Agent fixture (tok/s · accept · acc/step)

| class | v18ctl8 | T3rs (nvfp4) | **T3c** (fp8, live) | **T3cb** (fp8, loaded) |
|---|---|---|---|---|
| prose | 201.5 · 0.349 · 1.75 | 94.5 · 0.288 · 1.44 | 100.1 · **0.309** · 1.54 | 100.0 · 0.309 · 1.54 |
| structured | 232.7 · 0.525 · 2.63 | 112.0 · 0.442 · 2.21 | 123.5 · **0.479** · 2.40 | 122.2 · 0.479 · 2.40 |
| code | 231.8 · 0.661 · 3.30 | 156.5 · 0.688 · 3.44 | 148.4 · 0.609 · 3.04 | 147.0 · 0.609 · 3.04 |
| shell_ops | 264.4 · 0.908 · 4.54 | 149.2 · 0.829 · 4.14 | 166.0 · **0.908** · 4.54 | 163.8 · 0.908 · 4.54 |
| tool_json | 241.6 · 0.878 · 4.39 | 140.0 · 0.810 · 4.05 | 153.9 · **0.855** · 4.27 | 152.6 · 0.855 · 4.27 |
| weighted accept | 0.628 | 0.582 | 0.581 | 0.581 |

Acceptance is identical between T3c and T3cb (same config, same tokens) — as in every prior live/loaded pair.

## Parity (`e2c_compare_parity.py`, 18 prompts)

| pair | exact | note |
|---|---|---|
| **T3c vs T3cb** (same config, live tune vs its loaded cache) | **17/18** | only `agent-tool-1` (the tool prompt that jitters on every pair incl. control-vs-control) |
| T3c vs T3rs (fp8 vs nvfp4 KV, same image) | 3/18 | different KV dtype → different tokens; not a bar |
| T3c vs v18ctl8 · T3rs vs v18ctl8 | 2/18 · 2/18 | cross-image; not a bar (documented in T3b) |

## Power (`nvidia-smi` GB300; cap 1300 W, throttle `0x0` on every line)

T3c 362 → 385 W (pre was still warm from the tune) · T3rs 293 → 377 · v18ctl8 207 → 624 · T3cb 319 → 361 · 207 W at restore. Hook-off windows draw ~250 W less under load than v18 — the lane is waiting on C2C, not computing.

## Reading

1. **fp8 KV restores DSpark acceptance hook-off exactly as it did hook-on** (T3b6): prose 0.288 → 0.309, structured 0.442 → 0.479, shell 0.829 → 0.908, tool_json 0.810 → 0.855 vs the nvfp4 peer. Fixture tok/s: shell +11%, tool_json +10%, structured +10%, prose +6%, **code −5%** (code acceptance dropped 0.688 → 0.609 — the one class that went the other way; one fixture, one boot, unexplained). The mechanism is the image + KV dtype, independent of the hook.
2. **Cost of fp8 on the prose knee: C1 −4.6%** (91.3 vs 95.7), C4 −1–2%, C8 wash, **C12 +6%, C16 +2–4%**, replay +3–10%. Half the KV bytes per token (nvfp4) buys single-stream prose; fp8 buys the agent classes and mid-concurrency. For an agent lane, fp8.
3. **Live-vs-loaded, fourth pair: −0.9%** (91.3 → 90.5, 17/18 parity). The three hook-on pairs were −6 / −4 / −5.4%. Hook-off, the tax nearly vanishes. Consistent with the tax living in the hook's extra kernel shapes (two `do_finalize=False` MoE calls + finalize), not in FlashInfer's loader generically. Filed with the RFC #3920 comment as a discriminating observation.
4. **The servable flag-only number for one Station at 1M context, nightly image: 90.5 tok/s C1 · 302 C8 · ~420 C16 · KV 3.09M tokens · 7.5-min bind on a cache hit.** This is the row. The hook lane (v18, 171 C1 / 657 C8 / 947 C16) is a bind-mounted patch and is linked as a design note, not shipped as flags.

VERDICT T3c: flag-only nightly + fp8 KV **binds and serves** at 1M · C1 **91.3 live / 90.5 loaded** vs T3rs nvfp4 95.7 (−4.6%) vs v18 171 · C8 305 / 302 · C16 424–431 / 416–423 · fp8 restores acceptance hook-off (tool_json 0.855 vs 0.810, shell 0.908 vs 0.829; code 0.609 vs 0.688 the exception) · KV 3.09M vs 4.41M · parity T3c/T3cb 17/18 · live-vs-loaded **−0.9%** (4th pair; hook-on pairs were −4…−6%) · power cap 1300 W / 0x0 · v18 restored 07:26 · T3c/T3cb/T3 stopped-and-kept · new cache dir `7dbc24b6…` left in place · **recipes row = T3cb numbers**.
