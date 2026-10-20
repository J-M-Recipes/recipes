# Round 2 — SGLang ladder (Lane B, Sparks 1–4), 2026-10-19 18:57 → 22:52 CDT
Receipts: `spark1:~/bench-r2/<ARM>/` (bench-*.md/json, needles.log, prefill_reps.json, mixed.log, qeval.log, memwatch-sparkN.log, rendered-env.txt). Instrument = Lane C's (v41bench v1 T=0 thinking off, long_check, prefill_reps ×3, mixed_probe, qeval). Same env as the published B except the four host lines; one change per boot vs `.env.tp4.B0`.

| arm | change | C1 agg (run1 / run2) | C4 | C6 | 93K prefill single / 3-rep median | needles 20K / 131K / 300K ×3 (tok/s) | KV tokens | qeval | mixed short TTFT (s) | head MemAvail floor | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 | published v2.4 block, cold JIT | 81.6 / 83.5 | 206.0 | 249.9 | 5,003 / 4,955 | 4,229 / 4,273 / 2,227, 4,099, 4,157 | 5,356,032 | 71/75 | 1.9 / 2.4 | 11.5 GiB | control (cold) |
| B0ctl | warm reboot | 82.2 / 82.9 | 208.2 | 261.6 | **3,145 / 3,304** | 2,934 / 3,835 / 2,525, 2,530, 2,656 | 5,417,216 | 71/75 | 3.2 / 4.5 | 11.8 GiB | control (warm) — **prefill window contaminated, see note 1** |
| B1a | `DSV41_INDEXER_LOGITS_BUDGET_BYTES` 2→1 GiB | 80.0 / 82.4 | 203.8 | 252.8 | 5,847 / **6,187** | 5,283 / 4,427 / 3,484, 4,029, 4,741 | 5,353,728 | 71/75 | **1.7 / 1.8** | **14.9 GiB** | no speed KEEP; best memory floor + best mixed TTFT; needle run-1 still slow |
| B1b | budget 2→4 GiB | 81.9 / 81.2 | 205.0 | 255.2 | 5,906 / 6,080 | 5,358 / 5,028 / 2,653, 4,096, 4,404 | 4,519,680 | 71/75 | 4.8 / 1.7 | 12.9 GiB | REVERT (no gain, smaller pool this boot) |
| B2 | `--swa-prefix-tails` 128→64 | 78.7 / 81.2 | 209.7 | 258.6 | 5,730 / 6,086 | 5,344 / **5,975** / 2,992, 3,990, 4,885 | **5,927,424** | 71/75 | 1.9 / 1.7 | 13.3 GiB | capacity option (+9% pool), speed neutral — offer, not default |
| B3 | `--prefill-decode-interval` 2→4 | 82.2 / 83.7 | 208.8 | 259.4 | 4,361 / 5,290 | 5,312 / 4,784 / 2,973, 3,053, 3,003 | 4,817,664 | 71/75 | 1.7 / 1.9 | 14.2 GiB | REVERT (long prefill −13%, 300K −30%) |

Published B (Oct 18, Sparks 5–8): C1 84.8, C6 262.8, 93K 5,346, 300K 1,791/2,648/3,029. Lane C FINAL (tonight's C0ctl2): C1 104.3 per-stream (92.2 agg R1), C6 288.9, 93K 5,998, 300K 5,462/5,471/5,453.

## Read
1. **Decode has no SGLang lever in this set.** C1 78.7–83.7 and C6 250–262 across six boots is the noise band (±3%); nothing moved it. B stays ~10% behind C at C1 and C6.
2. **B's prefill is environment-bound, not config-bound.** The same config read 3.3K (B0ctl, 20:04–20:21) and 6.1–6.2K (B1a/B1b/B2, 21:25–22:30) at 93K. The only thing that changed between those windows is that the vLLM ladder on Sparks 5–8 was running its full battery (20:14–21:12) through the same CRS804 during B0ctl and was idle (FINAL restored 21:09) afterwards. Hypothesis, untested: concurrent long prefill on the sibling 4-node lane halves B's prefill through the shared switch (or shared power/thermal envelope). Needs a deliberate A/B. Until then, B's honest 93K number when the fabric is quiet is **~6.1K reps median — ahead of the published 5,346 and level with C's 5,998**.
3. **300K needle run 1 is slow on every arm (2.5–3.5K); runs 2–3 reach 4.0–4.9K.** First-touch cost, not config. Report run 1 separately (M3) and do not average it in. The NV_ERR_NO_MEMORY bursts seen in B0 did not recur as a burst in any arm (dmesg counts only rotated down).
4. **B1a (1 GiB indexer budget)** lifts the head's MemAvailable floor 11.8→14.9 GiB and gives the lowest mixed-traffic TTFT, at no measurable decode cost beyond noise. It is the right *robustness* default for a lane that serves 300K+ prompts, but it did not clear a +2% speed bar on n=1, so it is not promoted by the KEEP rule.
5. **KV pool varies by boot** on this lane (4.52M–5.93M) independent of the arm — knapcio's documented fast-loader variance. Only B2's +9% is attributable.

## State at 22:52
B3 (a REVERTED arm) was left serving on spark1 `:8888` by the chain (stop+keep is the operator's call; nothing routes to it — the gateway alias still points at the stopped Sparks-5–8 B). Lane C FINAL serving on spark5 `:8002`. TensorFold stop+keep on 1–4 (restore: `relaunch-rail2.sh`).


---
## Addendum 2026-10-20 (recipe build; not part of the 22:52 write-up above)
- **Read 2 is superseded.** The cross-lane contention A/B (2026-10-20 05:00-05:07 CDT, `raw/contention/RESULTS.md`) was NEGATIVE:
  B 93K 6,219 quiet -> 6,242 with C prefilling continuously; C 5,993 -> 6,009 with B prefilling. The B0ctl 3.3K window
  stays an unexplained outlier, not a fabric finding.
- B1a was re-booted as the serving lane 2026-10-20 04:43:38 -> 04:50:03 CDT (HTTP-ready), KV pool 5,219,072 tokens
  (`raw/chain/serve-b1a.log`, `raw/boot/B1a-serve-boot-20261020T0443.log`).
- `gates.py` exited rc=1 on every arm at its `/metrics` read (HTTP 404: this SGLang launch serves no `/metrics`), before it
  wrote `gates-functional.json`; the models/smoke/tool-call results of those runs were never recorded. The published
  `scripts/gates.py` makes `/metrics` optional.
- B0 and B0ctl ran on spark1's original checkpoint copy (before the 10-19 ~20:27 SKIP_PREPARE incident); B1a-B3 and the
  serving boot ran on the copy restored from spark5 at 21:08-21:21 and checked 48/48 against `raw/weights/sha256-1017.txt`.
