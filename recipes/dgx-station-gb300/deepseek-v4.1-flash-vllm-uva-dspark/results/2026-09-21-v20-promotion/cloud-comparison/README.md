# Cloud comparison — same 100 held-out tasks, Fable 5.1 vs this recipe (2026-09-21)

The point of the agent claim card is that a cost claim has to name its denominator. This bundle puts a frontier API
model through the **same** `harness/protocol.yaml` (single turn, `tool_choice auto`, max_tokens 2048, same grader,
same BFCL possible-answer files) on a fixed, seeded, stratified 100-task slice of the 1,311-task held-out sibling, and
prices both sides on stated bases.

## Rows

| | solved / 100 | input / output tokens | cost basis | $ per 1,000 solved |
|---|---|---|---|---|
| Claude Fable 5.1 (`claude-fable-5-1`), API | **84** | 142,790 / 14,297 | list $10 / $50 per MTok × metered usage | **$25.5** |
| Claude Fable 5.1, **floor** on all 1,311 | assumed 100% | 1,906,278 counted / 60 per call assumed | list price, no generation run | **≥ $17.5** |
| DSV4.1-Flash v20, this box, C8 | **86** | 100,871 / 9,168 | energy only, 490 W, $0.15/kWh | $0.008 |
| DSV4.1-Flash v20, this box, C8 | 86 | | + Station amortized $3.80/hr (≈$100k over 3 yr, 24×7) | **$0.30** |
| DSV4.1-Flash v20, this box, C1 (estimate) | 86 | | + amortized, single stream (~8× the C8 wall) | ~$2.4 |

Headline, stated conservatively on both sides: **≥ 58× cheaper per solved task at C8, amortized, against the Fable floor**
(the floor assumes Fable solves every task). Measured point estimate is 85× ($25.5 / $0.30). Energy-only the ratio is
~2,000×, and that is not a fair basis — it is on the card so nobody has to ask.

## How the Fable side was metered

- Route: Hermes' own Anthropic adapter on **Claude Max OAuth** (`manual:hermes_pkce`), not a billed API key. Token
  counts are the API's own `usage` fields; **the dollar figure is list price × those tokens, an estimate with no invoice.**
- The OAuth path wraps requests in the Claude Code identity (tool names re-namespaced `mcp__…`, prose aliases), which
  is one reason Fable's input count is ~40% above the local tokenizer's for the same prompts; the other is tokenizer
  difference. Both are real costs a Fable user pays for this suite through this route.
- Slice: `slice-ids.txt`, seed 20260921, stratified by `live_simple : live_multiple` (19 : 81), ids-sha16 `94f042799422e57a`.
  The local rows are the same 100 ids filtered from Card F's full held-out run (`../receipts/`), so both sides are graded
  once by the same code on identical tasks.
- Spend: $2.14 (est.) and 72 s at C4. This is the per-release cost of the comparison from now on.

## Deviations from `harness/protocol.yaml`, recorded

1. `temperature` was **not sent** to Fable 5.1 — the API rejects the parameter for this model ("deprecated"). The
   protocol says T=0; Fable ran at its default.
2. BFCL tool names with dots (`math.hypot`) were renamed on the wire (`_dot_`) to satisfy Anthropic's tool-name
   pattern and reversed before grading.
3. **2 of 100** (10 of 1,311) BFCL function schemas are rejected by Anthropic as invalid JSON Schema draft 2020-12.
   They count as **unsolved** for Fable (the protocol's rule: an API error after one retry is unsolved). The local
   engine accepts them. This is a real difference in what each side can be handed, not a grading artefact — but it is
   worth 2 points on this slice and is stated.
4. Solve rates on 100 are ±5 pt; cost per solved is dominated by tokens, so the ratio moves ~6% per 5 pt of solve rate.

## Assumptions on the local side

- $3.80/hr = $100,000 (Exxact order band for this configuration) / 3 years / 8,760 h. No power, no RTX PRO 4000,
  no NAS, no room, no labour. Saturated at C8 for the whole window; idle time is not charged. A real utilisation
  factor makes the local number worse, linearly.
- Energy number is the two-sample pre/post `nvidia-smi` mean (490 W) × suite wall (392 s), per Card F.

Files: `cloud-fable51-summary.json`, `cloud-fable51.jsonl` (every call, every token count, every graded call),
`cloud-fable51-floor.json`, `slice-ids.txt`. Runner: `../../2026-09-21-toolcall-gate-49435/bfcl_cloud_slice.py`.
