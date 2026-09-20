# T3b6 — nightly `dee37d89` + v15 hook + `--kv-cache-dtype fp8_ds_mla` — the nvfp4 KV was the acceptance drop; nightly + fp8 KV is +12.6% C1 live / +6.5% loaded

Worker: runner `../t3b6-fp8kv-runner-2026-09-19.sh` (Milo, nohup on the box, 22:10–23:01 CDT 2026-09-19). Verdict read from receipts by Milo (Hermes `milo` profile, claude-fable-5-1 via anthropic); the grok-4.6 worker dispatched to write this file was terminated after the runner finished and produced no artifact. Brief: `../briefs/HANDOFF-T3b6-fp8kv-grok-4.6.md`. Receipts: `receipts/` (knee/knee6/agentfix/replayc JSON, window logs, boot/smoke excerpts, parity captures, power lines, the new autotune set with sha256).

## Why

T3b5 (same evening) showed nightly + hook + a pinned autotune set is +9.6% C1 over v18 but loses the agent fixture (prose −10.5%, tool_json −10%, structured −8%) because the nightly accepts fewer DSpark drafts on every boot (prose 0.288 vs 0.349, tool_json 0.81 vs 0.88). The boot logs held the candidate cause: the nightly logs `Using DeepSeek's nvfp4_ds_mla KV cache format` where 0909 logs `fp8_ds_mla` — vLLM #56935's SM100 default — which also explains KV 2.50M → 4.53M tokens (half the bytes per token). `fp8_ds_mla` is still accepted by the nightly backend. T3b6 = the T3b5 launch plus `--kv-cache-dtype fp8_ds_mla`. Launcher `launch-t3-nightly.sh` gained a `KVDTYPE` passthrough.

## Boot facts

Both boots: `Using DeepSeek's fp8_ds_mla KV cache format` ✓ · `GPU KV cache size: 3,179,961 tokens` (3.03× at 1M; 0909 2.50M, nightly nvfp4 4.53M) · rehome 40 layers, `hbm_layers=31 uva_layers=9` (off54), residency 206.61 / 62.33 GiB as v18 · Engram 2 × 94.42 GiB pinned.

**The KV dtype changes the FlashInfer kernel-shape hash.** The pinned T3b4 set (`ed692e15…`) was placed and ignored; T3b6 tuned live for 15 min into a new dir `ddf01704…` (`Saved 189`). So the pinned-cache comparison from T3b5 does not apply here. T3b6b then **loaded** that set (`Loaded 189` from `ddf01704…`, 8-min bind) — which makes the pair a third live-vs-loaded observation. New set: `receipts/autotune_configs-fp8kv-ddf01704-from-T3b6.json`, sha256 `24a96383…` (box and Mac copies match).

## Windows — same-window T3b6 → v18ctl7 → T3b6b (`run_window_T1T2.sh`: knee ×2, knee6, agent_fixture_o, replay n=4 ×2)

| | C1 (r1 / r2 / k6) | C2 | C4 | C8 (r1 / r2) | C12 | C16 (r1 / r2) | replay n=4 (r1 / r2) |
|---|--:|--:|--:|--:|--:|--:|--:|
| **T3b6** live tune | **193.7 / 193.9 / 194.0** | 282–289 | 443–446 | 700 / 742 | 898–902 | 1041 / 1052 | 380 / 484 |
| v18ctl7 | 172.1 / 172.2 / 172.3 | 267–276 | 417–418 | 666 / 665 | 809–822 | 759 / 966 | 319 / 434 |
| **T3b6b** cache loaded | **183.3 / 183.2 / 183.2** | 257–263 | 417–419 | 629 / 630 | 823–848 | 973 / 986 | 316 / 485 |

C1 vs control: T3b6 **+12.6%**, T3b6b **+6.5%**. C8 paired r2: +11.5% / −5.3%. C16 paired r2: +8.9% / +2.1%. Replay: T3b6 +19% / +12%; T3b6b wash. C16 r1 of the control is the usual warm-cache low (759); pair r2/r2.

**Agent fixture** (`agent_fixture_o.sh`, tok/s · accept · acc/step):

| class | v18ctl7 | T3b6 | T3b6b | T3b5 (nvfp4 KV, for reference) |
|---|---|---|---|---|
| prose | 203.4 · 0.349 | 200.0 · **0.336** | 191.5 · 0.336 | 181.1 · 0.288 |
| structured | 235.7 · 0.525 | 240.1 · **0.492** | 226.6 · 0.492 | 215.0 · 0.442 |
| code | 237.4 · 0.661 | 248.5 · 0.650 | 234.5 · 0.650 | 243.5 · 0.688 |
| shell_ops | 269.9 · 0.908 | 278.2 · 0.829 | 262.8 · 0.829 | 263.1 · 0.829 |
| tool_json | 245.8 · 0.878 | 231.0 · **0.838** | 224.1 · 0.838 | 217.2 · 0.810 |
| weighted accept | 0.628 | 0.601 | 0.601 | 0.582 |

Acceptance is identical between T3b6 and T3b6b (same configs, same tokens) — as it was between T3b5 and T3b5b. The acceptance is a property of the image + KV dtype, not of the autotune set.

**Parity** (`e2c_compare_parity.py`, 18 prompts): T3b6 vs T3b6b **17/18** (the one agent-tool prompt that jitters on every pair); T3b6 vs T3b5 2/18 (different KV dtype → different tokens); T3b6 vs v18ctl7 2/18 (different image). Cross-image parity is not a bar here (documented in T3b).

**Power** (`nvidia-smi` GB300, before/after each window): cap 1300 W, throttle `0x0` on every line. T3b6 434 → 554 W (pre was still warm from the tune), v18ctl7 207 → 513 W, T3b6b 302 → 609 W; 207 W at restore.

## Reading

1. **The nvfp4 KV was the acceptance drop.** Prose 0.288 → 0.336 (control 0.349), tool_json 0.810 → 0.838 (0.878), structured 0.442 → 0.492 (0.525). Most of the gap closed; a residual 1.3–4 points remains and is the nightly's own kernel numerics (parity vs 0909 is 2/18 on every nightly boot).
2. **The nightly's speed was not the fp4 KV.** With fp8 KV the C1 gain *grew*: 193.7 live vs T3b5's 187.8 pinned-nvfp4 vs T3b4's 195.7 live-nvfp4. The +7–14% is the merged DSV4.1 kernels on the image.
3. **Live-vs-loaded, third observation.** T3b6 (the process that ran the tune) 193.7 → T3b6b (loading that tune's own output) 183.3 = **−5.4%**, tokens identical. Prior: −4% (T3b4→T3b5), −6% (T3b→T3b3). This is now a stable ~5% tax on FlashInfer 0.6.18.post1, mechanism unknown, filed upstream.
4. **Fixture verdict.** Against the bars: tool_json −6% (bar −3%) FAIL; every other class within bar (prose −1.7% / −5.9% on the loaded boot, structured +1.9% / −3.9%, code +4.7% / −1.2%, shell +3.1% / −2.6%). On the loaded boot — the one you can serve — prose and structured also slip past −3%. The lane's traffic is tool-heavy.
5. **Outcome (a) of the brief, with two caveats**: acceptance recovered and C1 ≥ +5% on both boots; but (i) the served number is 183, not 194, and (ii) tool_json is still −6%. This is the first nightly configuration that is a **promotion candidate**; it is not promoted on one pair.

## What it would take to promote (v19 candidate = nightly + hook + off54 + fp8 KV + pinned `ddf01704` set)

- One more same-window pair on the **loaded** config (fresh `docker run` with `ddf01704…` in place) to confirm 183 is not a low draw.
- The **fund harness** (`iwbench.py`, the workload v18 was promoted for) at C16/C24 — the knee is not the lane's job.
- A decision on tool_json −6% vs C1 +6.5% / C16 +2% / KV 3.18M vs 2.50M. James's call.

VERDICT T3b6: OUTCOME (a) — fp8_ds_mla logged both boots · KV 3,179,961 · **hash changed** (new dir `ddf01704…`, T3b6 live tune 15 min `Saved 189`; T3b6b `Loaded 189`) · prose accept 0.336 (v18 0.349, nvfp4 0.288) · tool_json 0.838 (0.878, 0.810) · structured 0.492 (0.525, 0.442) · C1 **193.7 live / 183.3 loaded** vs 172.1 (+12.6% / +6.5%) · C4 445 / 417 vs 417 · C8 r2 742 / 630 vs 665 · C16 r2 1052 / 986 vs 966 · replay 380–484 / 316–485 vs 319–434 · fixture tool_json −6.0% (FAIL −3% bar), others within bar on the live boot · parity T3b6 vs T3b6b 17/18 · live-vs-loaded −5.4% (third observation) · power cap 1300 W / 0x0 · v18 restored 23:01 · T3b6/T3b6b stopped-and-kept · `.from-T3b4` untouched, slow set back at canonical, post-run copy `.after-T3b6` · not promoted; **v19 candidate**.
