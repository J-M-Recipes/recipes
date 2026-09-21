# 2026-09-21 — tool-call correctness gate + #49435 isolation

Worker: grok-4.6 · xai-oauth. Window: James 24 h from 2026-09-20 19:45 CDT.
Box receipts: `/home/milo/dsv41/cardA-2026-09-20/`, `/home/milo/dsv41/cardB-2026-09-20/`.
Copied to `results/2026-09-21-toolcall-gate-49435/{cardA,cardB}/receipts/`.
Runners not rewritten. No `-REF` rename. `:30003` / `dsfv-*` / `glm53-*` untouched. No `docker rm` / `docker pull`.

Promotion rule (James, same window): BFCL exact-match ≥ v18 − 1 pt AND DSpark `tool_json`/`shell_ops` acceptance within 2 pts of v18. Replay tok/s is a speed bar only.

## Card A — BFCL gate, same window, v19-REF vs v18-RETIRED

Card A launched by Milo ~19:55 CDT; END 20:52:46 CDT; v19 restored BOUND after 300 s.

From `cardA/receipts/VERDICT-cardA.txt`:

| metric | v18 | v19 | delta | bar |
|---|---|---|---|---|
| BFCL all exact-match | 0.9333 | 0.9317 | -0.16 pt | >= v18-1pt PASS |
| BFCL simple_python exact-match | 0.94 | 0.935 | -0.5 pt | >= v18-1pt PASS |
| BFCL multiple exact-match | 0.92 | 0.925 | 0.5 pt | >= v18-1pt PASS |
| accept tool_json | 0.878 | 0.838 | -4.0 pt | within 2 pt FAIL |
| accept shell_ops | 0.908 | 0.829 | -7.9 pt | within 2 pt FAIL |
| accept prose | 0.349 | 0.336 | -1.3 pt | (info) |
| accept structured | 0.525 | 0.492 | -3.3 pt | (info) |
| accept code | 0.661 | 0.65 | -1.1 pt | (info) |
| no-call cases | 1 | 2 | | |
| errors | 0 | 0 | | |

BFCL n=600 (simple_python 400 + multiple 200), T=0, thinking off, our AST grader.
- v19: 559/600 = 0.9317, wall 173 s, power-pre 611.44 W / post 498.67 W, 2070 MHz, 0x0
- v18: 560/600 = 0.9333, wall 180 s, power-pre 610.56 W / post 504.78 W, 2070 MHz, 0x0

Acceptance from `agentfix-v{18,19}.json` (same numbers as the table). **Card A FAIL GATE**: BFCL bar holds; both acceptance bars fail.

GPQA-Diamond equal-budget v18 64k (`gpqa-v18-64k.log`): n=198 correct=172 acc=86.9% (95% CI ±4.7) no-parse=7 truncated=7 thinking=True T=0.0 maxtok=65536 reasoning_tok mean=6440 p95=24674 wall=2399s. Prior v19-64k (OPEN-FINDINGS, not this card): 173/198 = 87.4%, truncated=6.

### Failure-kind diff (the finding, not the count)

No `args_json` parse failures, no `extra`, no `missing` on either side.

| why-kind | v18 | v19 |
|---|--:|--:|
| value | 36 | 37 |
| name | 3 | 2 |
| no_call | 1 | 2 |
| args_json / extra / missing | 0 | 0 |

Pairwise (same ids): fail only v18=5, only v19=6, both=35. Both-fail kind pairs: (value,value)=32, (name,name)=2, (no_call,no_call)=1. Kinds do **not** differ; v19 did not introduce a new BFCL failure class. Unique misses are value-normalization / optional-arg / extra-location-string / one extra `no_call` (`multiple_138`).

20 lowest-hanging `ok=false` (sort: no_call, name, then value; `id, why, call`):

v18:
- multiple_76 why=no_call call=None
- multiple_13 why=name:corporate_finance.product_price call={name: corporate_finance.product_price, company=XYZ, product=A}
- multiple_10 why=name:database.create_backup call={name: database.create_backup, db_name=employees, ...}
- multiple_26 why=name:game_missions.list call={name: game_missions.list, game=Fortnite}
- simple_python_137 why=value:annual_growth_rate=6
- simple_python_150 why=value:annual_interest_rate=3
- multiple_133 why=value:annual_yield=5
- simple_python_260 why=value:area={'width': 20, 'height': 12}
- multiple_8 why=value:budget={'min': 300000, 'max': 400000}
- simple_python_337 why=value:cards={Alex/Sam/Robert/Steve hands}
- simple_python_55 why=value:cell_type='human cell'
- multiple_119 why=value:conditions=[age>25, job=engineer]
- simple_python_96 why=value:conditions=[age>25, job=engineer]
- simple_python_89 why=value:conditions={'department': 'Science', ...}
- multiple_54 why=value:frequency='quarterly'
- multiple_33 why=value:func='3*x**2'
- multiple_29 why=value:function1='3x+2'
- multiple_99 why=value:function='2*x^2'
- simple_python_16 why=value:function='2x^2'
- simple_python_14 why=value:function='3x^2 + 2x - 1'

v19:
- multiple_138 why=no_call call=None
- multiple_76 why=no_call call=None
- multiple_13 why=name:corporate_finance.product_price (same call as v18)
- multiple_10 why=name:database.create_backup (same call as v18)
- simple_python_137 why=value:annual_growth_rate=6
- simple_python_150 why=value:annual_interest_rate=3
- simple_python_152 why=value:annual_yield=5
- simple_python_260 why=value:area={'width': 20, 'height': 12}
- multiple_8 why=value:budget={'min': 300000, 'max': 400000}
- simple_python_337 why=value:cards=...
- simple_python_55 why=value:cell_type='human cell'
- multiple_119 why=value:conditions=[...]
- simple_python_96 why=value:conditions=[...]
- simple_python_89 why=value:conditions={...}
- simple_python_202 why=value:energy_type='solar'
- multiple_54 why=value:frequency='quarterly'
- multiple_33 why=value:func='3*x^2'
- multiple_29 why=value:function1='3x+2'
- multiple_99 why=value:function='2*x^2'
- simple_python_16 why=value:function='2x^2'

## Card B — isolation boot nightly-d05da62e (#49435) at v19 flags

Launched 20:53:51 CDT after Card A END. Image present (`docker image inspect` Id `sha256:f29125bcc6d276ed38a67c9dfd4e51ccbaba09ad92847a0913c0df38d9c05c71`, tag `vllm/vllm-openai:nightly-d05da62e9ccdf8e342b15bf6785d83224cc165af`). MemAvailable at launch 126.8 GiB host / 465 GiB after stop+drop_caches. No STOP-CAMPAIGN. Hook fired (not VOID). Candidate stopped-and-kept as `dsv41-vllm-v20cand-nightly-d05da62e-hook-off54-fp8kv-EXP` (Exited 0). v19 restored BOUND 21:28:06 after 320 s.

Live autotune (quoted from `boot-v20cand.txt`):
- `Using FlashInfer autotune cache file: /root/.cache/vllm/flashinfer_autotune_cache/0.6.18.post1/103a/ccaf97c66fb75ac99adc2a262b747fd8348c039a426dd5857ddf41f23ec65d0d/autotune_config`
- `[Autotuner]: Autotuning process starts ...` 2026-09-21 02:01:41,802
- `[Autotuner]: Autotuning process ends` 2026-09-21 02:17:58,513
- `[Autotuner]: Saved 189 configs` (189 new, 0 from previous)
Autotune wall **16 min 17 s** (not the ~75 min budget). New hash dir: `ccaf97c66fb75ac99adc2a262b747fd8348c039a426dd5857ddf41f23ec65d0d`. Bind after 1460 s.

### Boot facts (quoted)

From `boot-v20cand.txt` (must-show lines present except vLLM version):
- `Using DeepSeek's fp8_ds_mla KV cache format.`
- `(EngineCore pid=918) PIN_HOT rehome done layers=40 HBM_expert=206.61GiB pinned_expert=62.33GiB hbm_free=23.12GiB host_avail=19.28GiB (expect ~206.6 / ~62.4)`
- `Using backend AttentionBackendEnum.FLASH_ATTN for vit attention`
- `Using AttentionBackendEnum.FLASH_ATTN for MMEncoderAttention.`
- `GPU KV cache size: 3,179,961 tokens, Maximum concurrency for 1,048,576 tokens per request: 3.03x`
- `Graph capturing finished in 71 secs, took 1.29 GiB`

`boot-v20cand.txt` has **no** `vLLM version` line and **no** decoder-attention backend name. From the same container's docker logs (not in that file):
- `Initializing a V1 LLM engine (v0.29.1rc1.dev422+gd05da62e9) with config: ... kv_cache_dtype=fp8_ds_mla ...`
- `Setting kv cache block size to 128 for FLASHMLA_MEGA_ATTN_DSV41 backend.`

### TF split (discriminator) — `tf-split.txt`

```
v14nohook -> v18  (corpus: 79544 pos, flips 1.06%, mean|Δ| 0.0300)
  agent_tool docs= 8 pos=  1074 mean|Δlp|=0.0107 top1_flips=0.56%
  heldout    docs=10 pos=  2868 mean|Δlp|=0.0333 top1_flips=1.08%
  prose_10k  docs=21 pos= 75602 mean|Δlp|=0.0302 top1_flips=1.06%
v14nohook -> v19  (corpus: 79544 pos, flips 2.18%, mean|Δ| 0.0592)
  agent_tool docs= 8 pos=  1074 mean|Δlp|=0.2570 top1_flips=11.45%
  heldout    docs=10 pos=  2868 mean|Δlp|=0.2044 top1_flips=8.89%
  prose_10k  docs=21 pos= 75602 mean|Δlp|=0.0509 top1_flips=1.80%
v14nohook -> v20cand  (corpus: 79544 pos, flips 2.18%, mean|Δ| 0.0593)
  agent_tool docs= 8 pos=  1074 mean|Δlp|=0.2570 top1_flips=11.45%
  heldout    docs=10 pos=  2868 mean|Δlp|=0.2049 top1_flips=9.10%
  prose_10k  docs=21 pos= 75602 mean|Δlp|=0.0509 top1_flips=1.78%
```

Note: OPEN-FINDINGS used 18 agent/tool docs / 3942 pos (v18 0.94% / v19 9.59%). This card's `tf_split.py` class is 8 `agent_tool` docs / 1074 pos. Cause bar uses this file.

Card B KV-WRITER bar: v20cand agent_tool flips ≤ 2× v18 (1.12%) or ≤ 2.0% abs. Observed 11.45% — no.
IMAGE-KERNELS bar: ≥ 0.8× v19 (9.16%). Observed 11.45% = 1.00× v19 — yes.

### Speed (info only)

knee r2 vs v19's 183/667/995 (`knee-v20cand.txt` / `knee-v20cand-r2.json`):
- C1 193.2 agg tok/s
- C8 710.8 agg tok/s
- C16 1037.3 agg tok/s

### BFCL + acceptance on v20cand

From `bfcl-v20cand-summary.json` / `VERDICT-cardB.txt` / `agentfix-v20cand.json`:
- BFCL all 564/600 = 0.94 | simple 0.9425 | multiple 0.935 | no_call=1 | errors=0 | 166 s
- accept: prose 0.336, structured 0.492, code 0.65, shell_ops 0.829, tool_json 0.806
- fail_kinds all: value 33, name 2, no_call 1 (still no args_json/extra)

tool_json 0.806 is **below** v19's 0.838; shell_ops matches v19 at 0.829.

## VERDICT

VERDICT: v19 FAILS GATE, CAUSE = IMAGE-KERNELS

Card A fails the promotion rule on both acceptance bars (tool_json −4.0 pt, shell_ops −7.9 pt) even though BFCL exact-match holds (−0.16 pt). Card B shows v20cand agent_tool top-1 flips 11.45% identical to v19 and 20× v18's 0.56%; #49435 on nightly `d05da62e` at the same flags did not move the discriminator. James pre-authorised: Milo reverts the release to v18 + corrections. This worker did not execute the revert. `:30006` is still `dsv41-vllm-v19-nightly-hook-off54-fp8kv-ddf01704-BOUND-REF` (BOUND 21:28:06 CDT).
