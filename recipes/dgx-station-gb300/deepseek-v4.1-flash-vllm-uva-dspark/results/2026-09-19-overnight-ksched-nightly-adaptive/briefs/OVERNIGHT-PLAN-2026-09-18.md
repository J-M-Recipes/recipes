# OVERNIGHT PLAN — DSV4.1-Flash / GB300 — 2026-09-18 → 19

Orchestrator: Milo H. Workers: grok-4.6 · xai-oauth, one brief per test, from `HANDOFF-grok-4.6.md` rules (never `:30003`, never `glm53-*`, never `docker rm`/`pull`, drop caches, stop means stop, findings only). Box: `$BOX_USER@$BOX_HOST`. Lane: `:30006`. Live reference: `dsv41-vllm-v18-cgsizes-BOUND-REF` (v15 hook + seqs 24 + `[[1,4,5],[5,24,1]]` + lpt 6144 + token capture sizes). Verdict instrument: `knee.sh` ×2 (C1/C8/C16). Secondary: `agent_fixture_o.sh` (fixed CAT_ORDER), `replay_c.py` n=4 ×2, `e2c_heldout.py` (math + JP/DE), greedy parity 18 prompts. Every test is same-window **cand → v18ctl → cand**. Every candidate is stopped-and-kept with a verdict suffix. v18 is restored at the end regardless.

Hash facts (verified from source on main and the 0909 image, `vllm/config/speculative.py:600-646`, `vllm/config/vllm.py:458+`): the FlashInfer autotune key hashes `max_num_seqs`, `max_num_batched_tokens`, `max_model_len`, `cudagraph_capture_sizes`, offload/engram config, compilation config. It does **not** hash `num_speculative_tokens`, the batch-size k-schedule, `enable_adaptive_verification`, `long_prefill_token_threshold`, `prefill_schedule_interval`, `async_scheduling`, or `policy`. `seed_autotune.sh` is armed on every boot anyway (the campaign's `cbf5ea8f` retune on `[[1,3,5],[4,16,1]]` is unexplained by the source; treat a miss as expected-not-failure, budget 80 min).

Why these tests: the v15 static map made the recorded 3.7× k=5 unique-expert byte penalty collapse from 2.08× to 1.80× at C8–C16 (computed on `counts-20260917-183654.npz`), but v18's schedule still forces k=1 from 5 streams — a decision made under the positional map on 2026-09-12. Externally, neko-legends/spark-bench measured k=3 beating k=5 on chat (30.2→34.4 C1, 58.6→67.8 agg@4). That is the lever. Nightly `dee37d89` carries every landed DSV4.1 kernel PR (#56935 FlashMLA mega, #56962 Mega-mHC, #56266 Mega-Gate, #56464, #56568/#57204 MegaMoE) — the image is the second lever, and the overnight is exactly when a 74-min autotune is affordable.

## Order and time budget (~8 h)

| # | Test | Axis | Boot cost | Window | Est. |
|---|---|---|---|---|---|
| T1 | k-schedule `[[1,4,5],[5,12,3],[13,24,1]]` | spec K only | hash-safe (~6 min) | knee ×2, fixture, replay ×2 | 1.6 h (shares ctl with T2) |
| T2 | k-schedule `[[1,8,5],[9,24,3]]` | spec K only | hash-safe | same | +0.9 h |
| T3 | **nightly image swap** `vllm/vllm-openai:nightly-dee37d89115db4c94a820a79a78a7828e141c910`, v18 flags, **hook off** first | image | new hash, 74 min | smoke, knee ×2, fixture | 2.5 h |
| T3b | *(only if T3 binds and hook target class still exists)* nightly + v15 hook | image+hook | hash-hit from T3 | knee ×2, parity 18 | +1.0 h |
| T4 | E4b D1 counter **unfrozen**, gated at C8/C16, off-profile held-out | adaptive placement | hash-hit `9ac7b387` → v18 sizes: new hash likely | knee ×2, held-out JP/DE+math, parity | 1.5 h |
| — | v18 restore + verdicts + ledger | | | | 0.3 h |

T1/T2 first because they are the cheapest and the most likely to move the number. T3 is the long pole and runs when nothing else needs the box. T4 last; it is the lowest expected value and the only one whose tax is already known (−1.26% C8, −0.94% C16).

## T1 / T2 — k-schedule under the v15 map

Launch: `launch-many-seat.sh` with `KSCHED='[[1,4,5],[5,12,3],[13,24,1]]'` (T1) / `KSCHED='[[1,8,5],[9,24,3]]'` (T2). Nothing else changes. Names `dsv41-vllm-T1-ksched-5-3-1-EXP`, `dsv41-vllm-T2-ksched-5-3-EXP`. Verify hook lines (`hot=295 cold=89`, `HBM_expert=206.61GiB`), `GPU KV cache size: 2,502,950`, and `Loaded N configs` (hash hit expected; record hit/miss + minutes).

Instruments per window: `knee.sh` ×2; `knee6.sh` once (C3/C4/C6/C12 — the band that moves); `agent_fixture_o.sh` (CAT_ORDER `prose structured code shell_ops tool_json`); `replay_c.py` n=4 ×2; `/metrics` spec accept per position after each.

Win bars (two-window means vs v18ctl): C1 ≥ −1% (k=5 band is unchanged in both, so this is a sanity bar); **C8 ≥ +5%** and **C16 ≥ +5%** on the prose knee OR **replay n=4 ≥ +8%** with C8/C16 ≥ −2%; no fixture class < −3%; tools parsed in replay. Promote the better of T1/T2 if it passes; otherwise both kept as `-EXP-<reason>-FAIL`.

Expectation stated up front: prose knee at C8/C16 may lose (k=3 verify streams 1.5× the tokens of k=1 into a batch that already fills the step) while agent replay wins (85% accept). The verdict must report both; the promotion rule above encodes the lane's traffic (agents), not the knee alone.

## T3 — nightly image

Rationale: the 0909 image predates every merged DSV4.1 kernel. All v18 flag names still exist on main (`--offload-backend uva`, `--cpu-offload-params`, `--engram-config`, `method=dspark`, `num_speculative_tokens_per_batch_size`, `--long-prefill-token-threshold`, `--cudagraph-capture-sizes`). `--engram-config '{"cpu_offload": true}'` becomes redundant (#56512 default) but stays valid. **The one `docker pull` in this plan is pre-authorised for exactly this tag; nothing else.** Pull before T1 so it overlaps.

Step 1 (hook off): v18 flags, no sitecustomize mount, `PIN_MODE` unset. Name `dsv41-vllm-T3-nightly-dee37d89-nohook-EXP`. Autotune will miss; seed_autotune armed; if the seeded 0909 configs are rejected by the new FlashInfer, it is a full 74-min tune — allowed. Record: bind time, `GPU KV cache size`, graph capture GiB, any `Engram` log lines, tool-call parsing on `smoke_vllm.sh` 5/5. Then knee ×2 + fixture. Compare to **v14 88.7 C1** (positional, no hook), not to v18. Bar to continue: C1 ≥ 88.7 × 1.05 (kernels should be net positive with all experts positional) and smoke 5/5.

Step 2 (T3b, hook on): only if `python -c "import vllm...TrtLlmMxfp4ExpertsModular"` resolves in the new image and the class still has `_invoke_kernel` with the same signature (worker checks `inspect.signature` and pastes it into the README before booting). If the class moved/renamed, **stop T3b, report the new path, do not patch the hook overnight.** If it binds: knee ×2, parity 18 vs v18. Bar: C1 ≥ +5% over v18 with parity 18/18.

Known risk to name in the brief: nightly is cu134; the box driver is 595.84/595.91 CUDA 13.0/13.2 in-container. If the container fails at import with a driver/runtime mismatch, record the exact error and try `nightly-dev-*-cu13.0.1-591bb95` (2026-09-15, missing #56266) as the one fallback — it is not pre-pulled, so this fallback needs a second pull; allowed only if the first fails on driver mismatch specifically.

## T4 — observe a little, gate at C8/C16

E4b D1 counter (`COUNTER=d1`, `PIN_ADAPTIVE_FREEZE=0`, EWMA half-life default, `DRAIN_N` default, `SWAPS_PER_DRAIN` default, GO file dropped after bind) on v18 flags. Hash: v18's capture sizes are already tuned under `62426808…`; the hook adds no engine-config change, so hash hit expected. Name `dsv41-vllm-T4-adaptive-d1-unfrozen-C8gate-EXP`.

Windows: cand → v18ctl → cand. Instruments: knee ×2 (**verdict on C8 and C16, C1 reported only**); `e2c_heldout.py` JP/DE + math (the off-profile test — this is where adaptive earns its keep); `agent_fixture_o.sh`; greedy parity 18 vs v18ctl after ≥200 swaps have been applied (read `n_swaps` from the adaptive log). Also record swaps-to-plateau (swaps per drain over time).

Win bars: **C8 ≥ −1.5%, C16 ≥ −1.0%** (the measured tax was −1.26/−0.94; this asks that swaps do not add to it), **JP/DE held-out ≥ +20%** over v18 (E5 got +64% with a rebuild; adaptive should collect a good fraction without one), math ≥ −1.5%, parity 18/18, no fixture class < −3%. Pass = "one lane, no per-domain reboot, ≤1.5% at C8" — the operational claim, not a speed claim. Fail is still a useful number: it closes adaptive on this hook for good.

## Not in this plan, with reasons
- `enable_adaptive_verification`: ADAPT3 (#55095 backport, 2026-09-12) still captured 6.57 GiB and OOM'd KV at off60 on the 0909 image. Only viable on the nightly; if T3b passes, it is the first follow-up.
- `--async-scheduling`: measured wash 2026-09-14. `--prefill-schedule-interval` / `policy=priority`: hash-safe and cheap, but the fund-harness pain was solved by seqs 24; nothing to fix tonight.
- Per-layer slot allocation: the hook hard-codes `N_HOT=295 / N_COLD=89` and raises on any other count; per-layer counts need a hook change plus ~40 new kernel shapes. Not an overnight item; `dsv41-perlayer-slot-alloc-2026-09-18.json` is ready when it is.
- `max_num_seqs`, `max_num_batched_tokens`, capture sizes, offload GiB: each is a 74-min retune and none has a hypothesis tonight.

## Stop conditions
`STOP-CAMPAIGN` file honoured between steps. Any `ValueError|Traceback|RuntimeError:` in a serving log during a window ends that test. Host `MemAvailable` < 40 GiB or a `dsfv-*`/`glm53-*` container appearing ends the night and restores v18.

## Morning deliverables
`results/2026-09-19-overnight-ksched-nightly-adaptive/{T1,T2,T3,T4}/README.md` + jsonl + boot excerpts; `ledger.md` entries per milestone with model+provider; one `VERDICT` line per test; v18 live on `:30006` unless a candidate passed its promotion rule, in which case the candidate is up and v18 is stopped-and-kept — **Milo promotes, workers do not rename REFs.**
