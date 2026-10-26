# v27 live-lane T0 + T2 + T1(C1,C4) + held-out sibling — 2026-10-26

Lane: v27 (DeepSeek-V4.1-Flash-1017, GB300 + RTX PRO 6000 sidecar, block verification), `http://<gb300>:30006/v1`,
model `dsv41-flash-uva`, serving guests via Forge. Client: Mac (M4 Max) over LAN. Nothing started/stopped; no writes on the box.

## Clock (`date`)
- prep start 2026-10-26 17:07:46 CDT; runner START 17:08:15; ALL DONE 17:20:41; RUN.md 17:25 CDT
- per step (campaign.log): T0 17:08:15–17:08:16 · needle 17:08:16–17:09:50 · GSM8K-200 17:09:50–17:10:52 ·
  tool-eval 17:10:52–17:12:37 · T1 17:12:37–17:16:44 · BFCL held-out 17:16:44–17:20:41. Bench wall 12 min 26 s.

## Tools
- llm-inference-bench v0.7.7 @ c71ec1f2b34a4f1c8f702f1750ccd70da0e389e2 (`~/hermes/lil-bench`, its venv, `run.sh` sets
  `LLM_BENCH_NO_UPDATE_CHECK=1 --display-mode plain`). tool-eval-bench 570951a77e88 (bench default pin, cached in ~/.cache).
- bfcl_gate.py sha1 839e69c6 (identical to the box copy `~/pin-hot-experts/scripts/bfcl_gate.py`); BFCL data scp'd
  read-only from the box `~/pin-hot-experts/bfcl_data`, `SHA256SUMS` all OK.
- Commands: `command.txt`; runner `run_all.sh`; T0 `t0.py`.

## Flags (deviations from repo-standard in bold)
- needle: `--test-profile needle-checksum` **`--profile-concurrency 4`** (profile default C8); T=0, max_tokens 4096, 500 reqs.
- GSM8K: `--test-profile gsm8k --profile-runs 200` **`--profile-concurrency 4`** (default C30) `--compare-baseline <below>`;
  T=0, max_tokens 131072, reasoning_effort = template default (bench default; not set).
- tool-eval: `--tool-eval` **`--tool-eval-parallel 4`** (= leaderboard default 4); hardmode, seed 42, T=0; full 92 scenarios
  (T2 row says "optional subset"; the full set was run so it compares to the v21 bundle).
- T1: **`--concurrency 1,4`** (doc: 1,4,8 — C8 dropped for guests) `--contexts 0,32768,131072 --duration 30`.
- BFCL held-out: `SUITE=heldout` **`CONC=4`** (card runners use 8) `MAXTOK=2048 THINKING=0`.

## Baseline
Newest GSM8K json under either recipe's results/: sidecar recipe
`results/2026-10-24-gv-v25m-gates/gsm8k1319-V.json.gz` (Card GV, live v25m lane, 2026-10-24 05:44, full 1,319,
T=0, C30; 1277/1319 = 96.82%). Gunzipped copy: `baseline-gsm8k1319-V-v25m.json`. v25m(+kv-tier) is the lane v27
replaced; v27 = that lane + block verification. Paired on the 200 overlapping item ids.

## Results
| check | result | bar | verdict |
|---|---|---|---|
| T0 /v1/models | `dsv41-flash-uva` | id present | PASS |
| T0 exact-word chat | "PINEAPPLE" | exact | PASS |
| T0 tool_calls | 1 call get_weather {city: Austin, unit: f}, finish=tool_calls | parsed call | PASS |
| needle-checksum | EXACT 500/500 (0 near-miss/wrong/err) | EXACT N/N | PASS |
| GSM8K-200 paired | 192/200 = 96.0% vs baseline 192/200 on same ids; 0/0 flips; McNemar exact p = 1.0; ctok mean 238.6 vs 256.9 (×0.93) | no p<0.05 loss | PASS |
| tool-eval (92 sc.) | 89/100, 163/184 pts, pass 77 / partial 9 / fail 6; fails TC-43, 51, 68, 74, 89, 92; safety gate not passed (also not on v21) | none in doc (informational) | info: v21 bundle 92/100, 169/184, fails 68/74/89/92 — new fails TC-43, TC-51 |
| T1 C1/C4 × 0/32K/128K | see analysis.txt; all 6 cells ABOVE the v21 two-run band (C1 262/254/250 vs ≤214; C4 661/643/637 vs ≤410) | within run-to-run band of last promoted run | no regression; band is cross-lane (only lib T1 on record is v21 GB300-only) |
| BFCL held-out sibling | 1013/1311 = 77.27% (live_simple 78.29, live_multiple 77.02), 0 errors, 238 s | sidecar recipe: ≥ ref −1.0 pt | PASS: vs v25m GV-V 1018/1311 = 77.65 → −0.38 pt; paired 17 vs 12 flips, McNemar p = 0.458 |

Verdict per docs/benchmarking.md promotion bar: **PASS** (no significant T2 accuracy loss, needle EXACT 500/500, T1 not
below band). Held-out sibling run once for v27 (none found for v27 on the box since 2026-10-24 12:00).

## Caveats
- All quality checks are temperature 0. v27's block verification acts only at temperature > 0 (vLLM af7f9488 skips it on
  greedy rows), so needle/GSM8K/BFCL/tool-eval here measure the v25m+kv-tier target path, not the block-verification change.
- Guest traffic (`engine-load.tsv`, /metrics every 5 s, 147 samples): running never exceeded 4 (our own cap), waiting 0
  except one sample =1 at the 17:10:52 GSM8K→tool-eval handoff. No sign of other clients, but /metrics cannot attribute
  requests; vLLM `Running:` log lines were not captured (that needs `docker logs`, not used).
- T1 band reference is v21 (GB300-only, 0909 image) — the only lib T1 on record; no lib T1 exists for the sidecar lane.
- One `docker ps` (read-only) was run on the box during discovery: v27 pair Up ~37-38 min at ~17:05 CDT.
- Concurrency reduced vs repo-standard (see Flags); scores are concurrency-independent at T=0 in principle, latency not.
