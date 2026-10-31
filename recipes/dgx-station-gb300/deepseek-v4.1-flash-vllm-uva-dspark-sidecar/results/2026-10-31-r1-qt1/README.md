# 2026-10-31 — Card R1 (release dark window) + Card QT1 (temperature > 0 quality)

**Result: every release gate that was open on 2026-10-29 is now closed on Pg61, except the soak and the gate stamps
that follow it.** One dark window fault-injected the shipped watchdog, restored the lane, rolled back to h2-REF and cut
back to Pg61; a read-only live tap measured peer tier v3 numerics per layer for the first time; and a pre-registered
temperature > 0 run says the sampler is calibrated at T = 1.0 on the speculative-decode path. The recipe stays
`experimental` until the soak that started at the end of this window completes.

Read this bundle as five independent receipts, because they answer five different questions:

| question | card | verdict |
|---|---|---|
| Does the watchdog that ships in `scripts/sidecar_watchdog_p3.py` really stop a stalled sidecar? | R1 P2 (run 1) + R1b P2 (re-run) | run 1 **FAIL** on three log-order/cleanup bars, traced to the instrument; re-run **7/7 PASS** |
| Does the rollback script actually return the box to h2-REF, and does the cutover come back? | R1 P4 + P5 | **PASS**: rollback rc 0 in 551 s, cutover rc 0 in 452 s, KV line exact both ways |
| Is peer tier v3 arithmetically right, per layer? | live tap (pre-window) + R1 P3 (bring-up) | **PASS**: 40/40 layers both times, min cos 1.000000, max \|Δ\| 3.906e-3 |
| Do Pg61 and h2-REF put out the same next-token distributions? | QT1 P7 (inside the rollback boot) | **PASS**: 0.0574 mean \|Δlp\| / 1.664 % flips, inside the 0.0684 / 2.066 % bars |
| Does a tool call parse on the live lane (T0)? | smoke, 2026-10-31 11:19:47 CDT | **PASS**: `finish_reason: tool_calls`, `get_weather` with a parsed `city` (`raw/t0/toolcall.json`) |
| Is the sampler calibrated at temperature 1.0? | QT1 (discovery) → QT1-CONFIRM | discovery **FAIL** on a broken statistic; confirmatory run **PASS** on all six bars, both controls detected |

The dark window ran **2026-10-31 12:05:52 → 12:44:08 CDT (38 min 16 s)**, a hard bar of 45 min plus a QT1 allowance
of 75 min. Isolation was an `iptables` REJECT of every non-loopback packet to `:30006` for the dark phases; MAINTENANCE
was held for the whole window. The soak that ends this round started at **12:44:08 CDT** (17:44:08Z) and ends
**Monday 2026-11-02 11:44 AM CST (17:44Z)** — 48 h later, with DST ending on the Sunday between.

## Card R1 — what happened, in order

| time (CDT) | phase | what | receipt |
|---|---|---|---|
| 12:05:52 | P0 preflight | lane bound, KV 11,696,582 tokens, `BUILD-SHA256SUMS -c` OK, p3 file = `ba8373e9…`, rollback `--dry-run` preconditions OK | `raw/r1/P0/` |
| 12:05:52 | P2 start | isolation on, MAINTENANCE set, 4 streams opened, sidecar paused | `raw/r1/runner.log` |
| 12:06:16 | P2 trip | TRIP 2.86 s after the pause (`v3 stalled`), server killed, no stream got a `finish_reason` | `raw/r1/P2-run1/` |
| 12:06:21 | P2 verdict | **FAIL** — B3, B6, B7 (see the instrument note below) | `raw/r1/P2-run1/verdict.json` |
| 12:06:28 | P2 down | containers stopped, shm removed, watchdog killed by R1's own `down` | `raw/r1/P2-run1/down.log` |
| 12:11:30 | P3 restore | `launch-pg61.sh up` rc 0, **bind 302 s**, KV line 11,696,582 exact | `raw/r1/P3/kv.txt` |
| 12:12:02 | P3 tap | bring-up cosine tap rc 0, 40/40 layers | `raw/r1/P3/cos.json` |
| 12:12:02 | P4 rollback | drill to h2-REF starts | `raw/r1/P4/trace.log` |
| 12:14:07 | P4 | h2-REF RTX fresh `serving` | `raw/r1/P4/trace.log` |
| 12:21:04 | P4 | h2-REF **BOUND after 420 s** | `raw/r1/P4/trace.log` |
| 12:21:12 | P4 | v2 watchdog re-armed, pid 828714, PPID 1, status age 0 s | `raw/r1/P4/wd.txt` |
| 12:21:13 | P4 | smoke OK, `soak_check` rc 0 → rollback **rc 0 in 551 s** | `raw/r1/P4/soak.txt` |
| 12:21:45 | P4 | QT1 P7 h2-REF captures, 2 × ~15 s, 39 docs each | `raw/r1/P4/qt1.log` |
| 12:29:47 | P5 cutover | back on Pg61: rc 0 in 452 s, `BUILD=OK`, verify `gumbel 6eac3e13 / rsu edc8f8f4 / PEER3_SPINS=1e12`, KV exact | `raw/r1/P5/` |
| 12:31:35 | P6 | offline fp64 reference over the P3 tap capture, rc 0 | `raw/r1/P3/cosref.log` |

**The P2 run-1 FAIL is an instrument defect, and it stays on file.** `wd3_fault.sh` judged B2–B7 five seconds after the
TRIP, while the watchdog was inside `docker stop -t 5` of the (unpaused) sidecar; R1's own `down` then killed the
watchdog before it logged the sidecar stop. Amendment A4 changes one thing — wait for the watchdog process to exit
(≤ 60 s) before collecting B2–B7 — and no bar. The re-run (R1b, `raw/r1/R1b/`) is what the recipe claims:

| bar | run 1 (R1 P2) | re-run (R1b P2) |
|---|---|---|
| B1 trip latency ≤ 6.0 s | PASS 2.86 s | PASS 3.11 s |
| B2 server killed | PASS (`ds41=exited`) | PASS (`ds41=exited ds41-sidecar=exited`) |
| B3 kill-before-sidecar-stop log order | **FAIL** (verdict taken too early) | PASS (both lines logged, rc 0) |
| B4 no `finish_reason` on 4 streams | PASS (1,497/1,497/1,497/1,496 chunks) | PASS (1,457/1,457/1,455/1,458 chunks) |
| B5 no chunk after the stall (+0.5 s) | PASS 0.004 s | PASS 0.006 s |
| B6 GB300 freed + watchdog exits | **FAIL** (watchdog still `604630 1` at judgement) | PASS (GB300 22 MiB, watchdog after: none) |
| B7 sidecar stopped, not paused | **FAIL** (still running at judgement) | PASS (`stopped, not merely paused`) |
| unit tests before injection | 13/13 `ALL PASS` | 13/13 `ALL PASS` |

The R1b restore that closes the window: down rc 0, `up` rc 0 **bind 318 s**, KV PASS, `verify` rc 0, `BUILD=OK`,
isolation off at 12:44:08 CDT, soak clock started in that same `up` line.

## Peer tier v3 numerics — two taps, both PASS

The tap is a read-only reader over the peer tier v3 segment plus an **offline fp64 reference** (`peer3_cos_ref.py`,
written for this round). There is no v3 "check mode" in Garner's tree to switch on; the tap reads what the serving lane
already computes and never writes to it.

| | live tap, before the window | bring-up tap, R1 P3 |
|---|---|---|
| layers covered | 40/40 | 40/40 |
| cold (token, slot) pairs | 4,299 (2,059 samples attempted, logged `accept_rate` 0.0194) | 4,343 |
| rows compared | 3,449 | 3,475 |
| rows bf16 bit-identical vs fp64 reference | 3,381 (98.0 %) | 3,410 (98.1 %) |
| worst per-layer cosine | 0.99999998 (bar ≥ 0.9990) | 0.99999995 |
| worst per-layer max \|Δ\| | 3.906e-3 (bar ≤ 2e-2) | 3.906e-3 |
| sidecar timeouts during the tap | 0 | 0 |

Gates C1 (coverage), C2 (min cos), C3 (max |Δ|) are recorded as PASS in both `cos.json` files. **C4, C5 and C6** (zero
timeouts, tap overhead ≤ 2 % on p99 TTFT / median inter-token time, accepted-sample rate ≥ 30 %) are **not** recorded as
verdicts in these receipts: the raw per-snapshot timings are in `raw/live-tap2/overhead.csv` and `raw/r1/P3/overhead.csv`
(median snapshot cost ~0.6 ms, first snapshot ~1.4 s), but no gate file computes the bars. Do not quote C5/C6 as passed.

## Card QT1 — temperature > 0 quality

The instrument: private per-class prompts (200 rows, 40 each of prose, structured, code, shell_ops, tool_json) sampled
through `/v1/completions` at T = 1.0, top_p = 1, top_k = −1, `logprobs` = 20, then scored against the reported top-20
distribution with a coarsened randomised PIT. Prompts and per-request rows are **not** in this bundle (see
`rows_withheld` below); every published file here is an aggregate or a verdict.

**Three runs, and all three are on file** because the first two are what makes the third meaningful:

| run | rule set | positions | verdict |
|---|---|---|---|
| `run1/` discovery | registered before data (`PREREG.md`) | 41,705 | **FAIL** — P4 top-1 calibration z 5.07 (bar \|z\| < 3); everything else PASS |
| `run1-A1/` discovery re-scored | amendments A1–A2 applied post-data | 41,705 | **FAIL** — P2 per-class KS Holm min p 0.0341 (`tool_json`); P4 now z 1.15 |
| `run-confirm/` fresh arms | `PREREG-CONFIRM.md` (sha256 `8ae41fd2…`), code hashes frozen and checked | 60,073 | **PASS** — all six primary bars, both controls detected |

**Why the discovery P4 fired, and why it is not a sampler fault** (orchestrator review, `raw/qt1/docs/ORCH-REVIEW.md`):
the server lists the **sampled** token first in its top-logprob dict (6,124/6,124 diffuse positions), and fp32 logprobs
tie at the max on 374 of those 6,124. The registered statistic used first-max-wins credit, so it credited the emitted
token at every tie: tied positions +38 pp (z 15.8), **untied −0.07 pp (z −0.11)**. A no-draft replay showed the same
excess with no drafts in flight (+2.26 pp), and a bit-exact replay of v0.31.0's Gumbel-max sampler from the reported
logprobs, seed and position reproduced **6,124/6,124** emitted tokens. That function is calibrated over 2.45 M
fresh-seed draws (−0.04 pp, z −1.2).

**Confirmatory run, the numbers the recipe quotes** (`raw/qt1/run-confirm/`, arms 12:45–12:48 CDT, 200 records each,
max_tokens 256, 0 errors, watchdog `trip: null` at the end):

| id | test | result | bar | |
|---|---|---|---|---|
| Q1 | mid-PIT mean z | −0.211 | \|z\| < 3 | PASS |
| Q2 | rank-calibration χ², 7 categories | p 0.7355 | p ≥ 0.01 | PASS |
| Q3 | per-class χ², Holm (5 classes) | min p 0.7446 | ≥ 0.05 | PASS |
| Q4 | tie-fair top-1 rate vs mean p_max | z 0.193 (0.8614 vs 0.8612) | \|z\| < 3 | PASS |
| Q5 | dependence: lag-1 autocorr / runs | z −0.009 / −0.171 | both \|z\| < 3 | PASS |
| Q6 | NLL − H_partition | +0.0199 nats (tail bound 0.0514) | in [−0.02, bound+0.02] | PASS |
| S1 | control: K95 sampled, scored at T = 1 | M z −5.31 (R p 2.6e-4) | ≤ −3 | DETECTED |
| S2 | control: K105 sampled, scored at T = 1 | M z +5.33 (R p 3.6e-6) | ≥ +3 | DETECTED |

Supporting, not gating: pooled PIT KS p 0.2097 (D 0.00433), per-class KS Holm min 0.4806, chi² 10-bin p 0.4971.
Arms: C1 29,848 positions, C2 30,225, K95 29,535, K105 30,269.

**The claim that is allowed, verbatim from the pre-registration:** sampler output at T = 1.0, top_p = 1 is calibrated to
the reported target distribution on the speculative-decode path, within the detectable effect. **Not** excluded: argmax
substitution below ~2 % per token, |ΔT| below ~5 %. **Not covered:** other temperatures, top_p/top_k truncation, the
accept/reject arithmetic in isolation (Card G3's GPU unit test), anything outside the reported top-20. A ~5 % error of
the temperature class would pass this gate — the confirmatory controls show exactly that, because 0.95 and 1.05 are the
smallest moves the instrument can see.

## QT1 P7 — forward fidelity, Pg61 vs h2-REF

Captured inside the rollback-drill boot on h2-REF (pre-registered in `raw/qt1/docs/P7-PREREG.md`), 39 documents,
79,505 forced positions, `prompt_logprobs` = 20, T = 0, C = 1.

| compare | mean \|Δlp\| | tie-aware top-1 flips | Jaccard@20 |
|---|--:|--:|--:|
| Pg61 self-repeat (floor) | 0.0547 | 1.532 % | 0.771 |
| h2-REF self-repeat (floor) | 0.0531 | 1.566 % | 0.768 |
| **Pg61 vs h2-REF** | **0.0574** | **1.664 %** | 0.764 |

| bar | rule | measured | |
|---|---|---|---|
| P7a | ≤ 1.25 × max(floors) = 0.0684 | 0.0574 | PASS |
| P7b | ≤ max(floor flips) + 0.50 pp = 2.066 % | 1.664 % | PASS |
| P7c | h2 floor ≤ 0.08 mean \|Δlp\| | 0.0531 | PASS |
| P7d | ≥ 79,000 aligned positions | 79,505 | PASS |

This speaks to the prefill/forced path only. Decode graphs and speculation are QT1-A's subject, above.

**Two older numbers are withdrawn, not deleted:** the cross-lane 0.0785 / 2.93 % reading and the 99.5 % "flip" files
(`raw/qt1/run1/tf-Pg61-vs-*.json`) came from a one-token alignment defect in historical receipts (`prompt_tokens = n+1`
with only the scored positions written). They are kept here so the withdrawal is auditable.

## Soak

| item | value |
|---|---|
| start | 2026-10-31 **12:44:08 CDT** (17:44:08Z, epoch 1793468648), written by `launch-pg61.sh up` |
| end | **Monday 2026-11-02 11:44 AM CST (17:44Z)** — 48 h; DST ends Sunday 2026-11-01 |
| sampler | `scripts/pg61_lane_check.sh`, M4 cron every 10 min, silent when healthy |
| break rule (pre-registered) | any watchdog trip, any restart, or C1 p99 outside the control band |
| status | `experimental` until it completes and the five gates are stamped |

Card R1's own P5 `up` wrote `raw/r1/P5/soak-start.json` at **12:29:46 CDT**; the R1b re-run restarted the lane 14 min
later and wrote **17:44:08Z**, which is the value above (`raw/r1/R1b/up.log`). Both stamps are in the bundle because the
earlier one is what the first half of the window produced.

## What is NOT in this bundle

* **Private prompt text and per-request rows.** `rows_withheld` — the QT1 fixtures are real Hermes transcripts
  (`perclass_prompts.jsonl`, 200 rows) and every `gen-*.jsonl` / `rescore-*.jsonl` / `promptids.jsonl` / `p4ns-*.jsonl`
  carries prompt-derived tokens and output heads. None of them is published, here or anywhere else. The prompt file's
  identity is recorded in `raw/qt1/docs/PREREG.md`; its content is not.
* **The tap's binary payload** (`tap.bin`, `tap.jsonl`) — the per-row cold-expert tensors. Only the aggregate `cos.json`
  and the run logs are published.
* **`shm.tsv` (474 KB) and the two Prometheus dumps (90 KB each)** are published gzipped (`gzip -n`, mtime 0). Nothing
  inside them was edited.
* **The C4/C5/C6 gate verdicts** (above): the raw timings exist, the bars were never computed into a verdict file.

## Reproducing this bundle

`raw/` is a verbatim copy of the box receipts with one transform applied by a script kept outside the repo:
the box's home directory → `~`, an absolute user-home path → `~`, an Authorization header value → `<redacted>`, and LAN
addresses → `HOST`. `scripts/qt1_gen.py` has one further edit: its hard-coded prompt path became
`os.environ.get("QT1_PROMPTS", os.path.expanduser("~/dsv41/bench/perclass_prompts.jsonl"))`, so the published copy runs
from any checkout without naming our home directory. Every other script is byte-identical to what ran on the box.

Instrument inventory: `scripts/r1_runner.sh`, `wd3_fault.sh` (fixed) + `wd3_fault.v1.sh` (as run 1 ran it),
`r1b_fault_rerun.sh`, `go-r1.sh`, `soak_check_local.sh`, `peer3_cos_tap.py`, `peer3_cos_ref.py`; `qt1_gen.py`,
`qt1_rescore.py`, `qt1_tf.py`, `qt1_tf_hist.py`, `qt1_analyze.py`, `qt1_analyze.v1-as-registered.py` (the registered
code, kept so the FAIL can be re-derived), `qt1_exact.py`, `qt1_exact_sim.py`, `qt1_sim.py`, `qt1_p4_nonspec.py`,
`qt1_p4_nonspec_analyze.py`, `qt1_top1_detail.py`, `sampler_replay.py`, `seed_structure.py`, `p4_collision_check.py`,
`qt1_arm.sh`, `qt1c_run.sh`.

## Receipt quirks worth knowing before you quote a file

* `raw/r1/P4/kv.txt` holds `4,400,013` tokens **timestamped 2026-10-28 22:18** — `docker logs | grep -m1` took the
  oldest line of the reused h2-REF container, not the fresh boot. The P4 bar (A4) is BOUND + sidecar running, which the
  trace shows; do not read that KV line as this boot's.
* `raw/r1/P3/kv.txt` and `raw/r1/P5/kv.txt` end in a stray `NO-KV-LINE` sentinel from a second read of the same file;
  the KV line above it is the real one.
* `raw/r1/P2-run1/fault.log` shows the verdict line with `B3/B6/B7` FAIL — that file is run 1, unedited. The PASS
  evidence is under `raw/r1/R1b/P2/`.
* `raw/qt1/docs/DISPOSITION-QT1.md` is the pre-look-ahead disposition (it calls P4 "the one real defect" and QT1-C
  "OPEN"). It is wrong on both counts and kept as the record of what was believed at 11:50 CDT; `ORCH-REVIEW.md` §2 and
  `VERDICT-CONFIRM.md` supersede it.
