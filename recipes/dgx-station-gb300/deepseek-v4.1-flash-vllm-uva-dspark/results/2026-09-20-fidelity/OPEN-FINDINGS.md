# OPEN FINDINGS — 2026-09-20, end of day (parked; pick up here)

Written by Milo after James's read of a Grok review of the recipe: *"I shouldn't have compromised on JSON tool calling."*
James agreed; so do I, on the evidence below. Development is paused with these open. **Nothing here is resolved.**

## State of the box when parked

- `:30006` = `dsv41-vllm-v19-nightly-hook-off54-fp8kv-ddf01704-BOUND-REF` (promoted 12:11 CDT). James chose to keep using it
  for inference while parked; it was not rolled back.
- Rollback, one line, ~5 min: `docker stop dsv41-vllm-v19-nightly-hook-off54-fp8kv-ddf01704-BOUND-REF && docker start dsv41-vllm-v18-cgsizes-RETIRED-REF`
- v18 (`-RETIRED-REF`), v14 no-hook (`-RETIRED-REF`), v19a/v19b/ASYNC EXP containers all stopped-and-kept.

## Finding 1 — v19's logit drift concentrates on tool-shaped text (the thing the card's average hid)

Teacher-forced Δlogprob (`receipts/tf-compare-*.json`, `per_doc`), split by document class:

| pair | class | positions | mean \|Δlp\| | top-1 flips |
|---|---|--:|--:|--:|
| no-hook → v18 | agent/tool docs (18) | 3,942 | 0.027 | 0.94% |
| no-hook → v18 | 10-K prose (21) | 75,602 | 0.030 | 1.06% |
| no-hook → **v19** | agent/tool docs (18) | 3,942 | **0.219** | **9.59%** |
| no-hook → v19 | 10-K prose (21) | 75,602 | 0.051 | 1.80% |

The v18 hook is uniform noise across both classes. **v19 moves the distribution ~8× more on tool-shaped text than on prose**,
and one in ten argmaxes changes there. The corpus-wide figure on the card (2.2% flips, ppl unchanged) is true and was
dominated by the 95% of positions that are prose. This is the mechanism behind two things that were already on the record
and were explained away as noise:

- DSpark acceptance on every nightly boot: tool_json 0.878 → 0.838, shell 0.908 → 0.829 (stable, every boot, hook on or off).
  Acceptance is a direct read of how often the draft's argmax equals the target's — it was reporting exactly this.
- v19-vs-v18 greedy parity 2/18 while v19a-vs-v19b is 18/18: v19 is self-consistent and emits *different tool calls* than v18.

**What is NOT shown:** a tool-calling correctness regression. Fund harness 64/64 and Hermes 10/10 pass on both. n=64 cannot
detect a few-percent regression. So "compromised" describes the evidence base, not (yet) the behaviour.

**What I got wrong today:** the fixture's tool_json *tok/s* class genuinely cannot resolve 3% (control spans ±7% on identical
boots), and I used that to retire it as a gate and replace it with replay tool-turn *throughput* — which v19 wins. Speed was
never the question; acceptance and argmax agreement were, and both were saying the same thing all along. A gate that measures
the wrong quantity was replaced with a gate that measures a different wrong quantity.

## Finding 2 — the GPQA pair is not a pair

| run | budget | acc | truncated |
|---|--:|--:|--:|
| v18 | 16k | 80.3% (159/198) | 24 |
| v19 | 64k | 87.4% (173/198) | 6 |

v19's number is fine as a sanity anchor (model card 90.9 at max effort, T=1). It says nothing about v19 vs v18 because the
budgets differ; 24 of v18's misses are the cap, not the model. Equal-budget v18 at 64k is ~40 min on a `docker start`.

## Finding 3 — cause not isolated

v19 changes three things at once vs v18: nightly image `dee37d89` (kernels), `--kv-cache-dtype fp8_ds_mla`, `--cpu-offload-gb`
60 → 54. The tool-shaped drift could be the fp8 KV path (attention numerics on short, structured, high-confidence sequences)
or the nightly's MoE/attention kernels. One boot separates them.

## Plan when development resumes (in order; ~2 h box time, all `docker start`s + one live tune)

1. **Tool-call correctness at real n, same window, both containers.** BFCL-v3 `simple` + `multiple` (~400 calls, exact-match
   function + args, our grader) on v19-REF and v18-RETIRED. ~30 min. **This is the gate that answers Grok's question.**
   Promotion rule going forward: BFCL exact-match ≥ v18 − 1 pt, *and* DSpark tool_json/shell acceptance within 2 pts of v18.
   Replay tool-turn tok/s stays as a speed bar, not a correctness bar.
2. **Equal-budget GPQA**: v18 at 64k. ~40 min. Card then shows a real pair.
3. **Isolate**: nightly + hook + off54 with the nightly's *default* KV dtype (nvfp4_ds_mla) — the T3rs container from
   `2026-09-19-overnight-ksched-nightly-adaptive` is that configuration hook-off; hook-on needs one live tune (~75 min) or the
   T3b5 set. Run `tf_logprob.py` on it; if the agent-doc flips follow fp8 KV the fix is a flag, if they follow the image v19 is
   the wrong base and the release reverts to v18 with the 552B/fidelity corrections only.
4. **Card**: keep the split Δlogprob row (already on the card). Add a BFCL row. Never again put a corpus-wide fidelity average
   on the card without the per-class split — `tf_logprob.py compare` should print the split by default (todo in the script).
5. **Blog/X**: the Round 10 callout draft (`~/.hermes/profiles/milo/work/blog/dsv41-round10-callout-DRAFT.html`) claims
   "fidelity measured, not degradation." It is **not deployed** and must not be until (1) lands. The X thread draft likewise.

## Files

- `receipts/tf-compare-v14nohook-vs-{v18,v19}.json` — per_doc split is derivable; the numbers above came from
  `sum over docs` with `parity-*` as the tool class.
- `receipts/gpqa-v19.{jsonl,log}` — 173/198, 64k budget, 6 truncated (idx 79, 81, 88, 99, 101, 147).
- Upstream threads still watched by cron (`5acea53f7356`): FlashInfer #3920, vllm recipes #979.
