# QT1 PRE-REGISTRATION — temperature>0 sampler calibration, DSV4.1-Flash Pg61 (:30006)

Frozen BEFORE any live arm was generated (written 2026-10-31, after design review, before `qt1_gen.py`
touched the lane). `qt1_analyze.py` BARS must equal this file; `VERDICT.txt` prints them side by side.

## Question

The sidecar recipe ships with temperature > 0 sampling and speculative decoding on. Every quality
receipt we have for this lane so far is teacher-forced / greedy (`dlp` floors, top-1 flips), and those
are **blind to the sampler**: they score the model's logits, not the choice of token. QT1 asks the only
question a T>0 recipe actually depends on: **are the emitted tokens exact samples of the target
conditional?** If yes, the sampler (temperature processing, rejection sampling, block verification,
suffix drafting) is distribution-preserving at T=1; if not, the recipe is shipping a different
distribution than the one we validated.

## Instrument (why a PIT and not a likelihood test)

At each verified decode position the vLLM v0.31.0 GPU runner returns top-k logprobs computed from the
**target** logits (rejection sampler, `logprobs_mode=raw_logprobs`), plus the emitted token's own
logprob. At T=1, top_p=1, top_k=-1 the processed distribution equals the raw one, so the returned
bins are the target conditional's top-20, coarsened to 20 singletons + one tail atom.

For each emitted token:

    u = F_{r-1} + V * p_r        (emitted token is rank r in the returned top-20)
    u = F_20    + V * t          (outside the top-20; t = 1 - sum(top-20))
    V ~ U(0,1) drawn fresh per position

Under H0 ("the emitted token is an exact draw from the target conditional") **u ~ U(0,1) exactly**, at
any coarsening, at any temperature, whatever the shape of the distribution. No full-vocab access is
needed and nothing about peakedness has to be assumed.

## Planned N (fixed before the run)

| item | value |
|---|---|
| prompts | `~/dsv41/bench/perclass_prompts.jsonl`, all 200 rows (40 per class, 5 classes) |
| max_tokens / prompt | 128, `ignore_eos` semantics not forced but 128 cap |
| arms | `T1.0a` (seed base 100), `T1.0b` (seed base 900) — the two primary arms |
| positive-control arms | `T0.95` (sampled at 0.95), `T1.05` (sampled at 1.05), both scored at T=1 |
| positions | 25,600 per arm; **primary pooled 51,200**; per class 10,240 |
| concurrency | 4, box loopback only |
| PIT seed | 20261031 |

## Pre-registered bars (alpha BEFORE data)

| id | test | bar |
|---|---|---|
| P1 | pooled PIT KS vs U(0,1), primary arms pooled | p >= 0.01 |
| P2 | per-class KS, Holm across the 5 classes | min Holm p >= 0.05 |
| P3 | dependence: z(lag-1 autocorr of u-0.5) and z(runs) | both \|z\| < 3 |
| P4 | top-1 calibration: fraction(emitted==argmax) vs mean max-prob | \|z\| < 3 |
| P5 | mean(-log p(x)) - mean H_partition | in [-0.02, tail_bound + 0.02] nats |
| P6 | QT1-B decode-vs-prefill: mean\|dlp\| / top-1 agreement | <= 0.10 / >= 0.95 |
| P7 | QT1-C Pg61-vs-h2REF forced-token top-20 (h2-REF, dark window) | mean\|dlp\| <= 0.062 and flips <= 2.40% (h2-REF repeat floor 0.0560 / 2.181% x 1.10) |

P3 exists because a #61161-class bug (one uniform shared by every position in a verified block, or the
residual/shifted uniform reused across a block) leaves the **marginal** of u exactly uniform: P1/P2
cannot see it, only the dependence family can. P4 catches a sampler that always takes the argmax; P5
catches a distribution that is too flat or too peaked for its own reported entropy.

Decision rule: PASS = P1..P6 all pass (P7 measured separately when h2-REF is booted). Any single
failure = FAIL, and the failing family names the suspect subsystem. No bar is adjusted after data.

## Mandatory instrument controls (run with the analysis, not after)

1. **Selftest** (`qt1_analyze.py --selftest`): stats library vs closed forms; PIT exactness on
   synthetic peaked/flat mixtures; T-mismatch must reject; shared-V block must trip P3.
2. **Null simulation**, 200 replicates: draw synthetic tokens exactly from each *recorded* distribution
   (top-20 + tail), score them with the same instrument. Reported false-alarm rate at the registered
   alpha must be ~1 % (accept 0.5 %-2 % at 200 reps).
3. **Power simulations** at the planned N: (a) temperature mismatch (T=0.95/1.05 sampled, scored at 1)
   — also run LIVE as a positive control; (b) draw reuse across verified blocks (k=1,2,3,5);
   (c) acceptance bias toward the drafter (mixture weight epsilon toward the draft's top-5 support).
   Report the smallest detectable effect at 80 % power.

## What this does NOT test

The accept/reject arithmetic itself (that is the G3 GPU unit test), speed, the tail atom's internal
shape beyond its aggregate mass, temperatures not sampled here, or any prompt whose text we did not
send. Prompts are private Hermes transcripts: counts and aggregate statistics only, no prompt text in
any report.
