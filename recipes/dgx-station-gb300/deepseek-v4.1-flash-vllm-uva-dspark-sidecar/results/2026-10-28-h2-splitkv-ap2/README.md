# 2026-10-28 — Card AP2: split-KV attention on the guarded reference, and the h2-REF promotion

**Result:** `S8=NO S4=PROMOTE-READY` under frozen gates (`ap2_verdict.frozen.py`, sha256 `88cac1b9…`, frozen
15:10:41 CDT before boot A, verdict 17:08:24 CDT). **h2-REF (= arm S4) was promoted at 17:22:58 CDT** and serves `:30006`.

**Arms.** All four boot through the same frozen launcher (`launch/launch_att2d.sh.as-run`, sha1 `88476d94`) from the
same read-only artifact directory. The promotion mounted that directory too.

| arm | what it is | container |
|---|---|---|
| A, A2 | control = **h1-REF**: v27 + hook-h1 L2 key-sharing + the vLLM #61161 overlay, with the **vLLM #60945 guard** (`rejection_sampler_utils.py` 2d43d513), `NOCORE=1` | `dsv41-vllm-1017-AP2-A`, `-A2` |
| S8 | A + split-KV side extension `_flashmla_splitkv_C` (744a9de8) + overlay (9946e6ee / 97c67176), `SPLITKV=8 MAX_TOKENS=8 MAX_REQS=1 SQ=6 TRACE=1 GUARD=0` | `-AP2-S8` |
| S4 | the same at `SPLITKV=4` | `-AP2-S4` |

The image's own `_flashmla_C.abi3.so` (df2cdd1c) is never mounted over. The side `.so` is an extra file under its
own torch namespace.

**Cells per arm.** Each ran under a 2 s fault watcher that stops the cell on a new Xid, a GPU-fault log line or
container death, then skips the arm's remaining cells:

1. warm;
2. BRRR padded codegen 2K at T=0, k=5, 5 reps;
3. greedy 3 per class;
4. llm-inference-bench 0.6.2 as **one process** running C1/C8/C32 × ctx 0/64K (the shape that crashed every
   unguarded split arm);
5. prefill/TTFT at 8K/16K/64K/128K × 2;
6. per-class decode at T=1, natural + r1 + r2, C1 and C8;
7. teacher-forced capture;
8. gauntlet;
9. #62417 probe (thinking on, 40 seeded arithmetic questions);
10. 4 × 16K long-thinking soak;
11. a ~900K needle on A, S8 and S4.

## Gates and readings

See [`VERDICT.txt`](VERDICT.txt) and [`verdict.json`](verdict.json).

| gate | S8 | **S4** |
|---|---|---|
| G0 validity (KV 4,400,013, args = h1-REF, env/binds/ulimits = expected delta, guard sha in-container, 0 foreign POSTs, 0 sidecar timeouts) | PASS | **PASS** |
| G1 controls: A 11.092 / A2 11.106 ms/step, spread 0.13% (bar 1.0%) | PASS | **PASS** |
| engaged: side `.so` mapped; the only decode gate line is `s_q=6 splits=N kind=compressed op=_flashmla_splitkv_C`; TRACE ok=1 only at (6,6,1,0); 0 guard lines | PASS | **PASS** |
| **G_crashfree** (hard): 0 new Xid, 0 engine errors (server + sidecar), 0 fault lines, no fault-stopped cell, alive at END, 0 watchdog trips, no lib 5xx | PASS | **PASS** |
| G_brrr ≥ 1.025 × max(A,A2) | **FAIL** 477.8 = 1.022× | **PASS 484.5 = 1.036×**, −0.429 ms/step |
| G_tf: W ≤ 3.192, confident flips ≤ 4, agent/tool confident 0 | **FAIL** (6 confident flips) | **PASS** W 2.784, 4 flips, 0 agent/tool |
| gauntlet 12/12 · #62417 probe · soak 4/4 0 loops · 1M needle | PASS | **PASS** (P5 39/40; 1M TTFT 31.172 s) |
| HOLD: per-class C1/C8 · lib 6 cells · prefill/TTFT | PASS | **PASS** |

Controls A and A2 were both crash-free. Control A's probe P5 read 37/40, which passes the relative bar
min(0.93, min(ctrl) − 0.05) = 0.875.

Every number is in [`throughput.csv`](throughput.csv) with its arm and raw file. The ratios are summarised in the recipe
README's 2026-10-28 section.

## Promotion

See [`promote/`](promote/).

- **Timeline.** `promote_guard_split.sh` (as run: `launch/promote_guard_split.sh.as-run`) started at 17:12:45 CDT.
  It stopped and kept h1-REF at 17:13:24, then started h2-REF from `launch/start_h2ref.sh.as-run` with `SPLIT=4`.
  The server bound after 537 s.
- **Post-boot proofs.** All of these passed:
  - KV 4,400,013, `l2_patched=2`, autotune `Loaded 117`;
  - `args_eq_h1ref=1`, env, binds and ulimits equal to arm S4's;
  - split enabled for decode only at s_q 6 with 4 splits, and TRACE ok=1 only at (6,6,1,0);
  - 0 guard lines;
  - shipped, side and guard shas as pinned;
  - smokes 17×19 → 323 with thinking off and on;
  - the 1M needle PASS (900,372 tokens, TTFT 31.237 s);
  - 0 engine errors, 0 fault lines and 0 new Xid during the promotion.
- **Watchdog.** The fail-stop watchdog was re-armed as pid 2638240, PPID 1.
- **Live read.** At 22:53:36Z: [`live-check-2026-10-28T2253Z.txt`](live-check-2026-10-28T2253Z.txt).

## Files

| path | what |
|---|---|
| `VERDICT.txt`, `verdict.json`, `ap2_verdict.frozen.py`, `verdict.err` | the frozen gate script and its output |
| `runner.log`, `phases.jsonl`, `instruments.txt`, `pins.txt` | runner timeline; sha1 of every script/fixture as run on the box (published copies differ by path scrub only) |
| `throughput.csv` | every speed cell × arm, built from `verdict.json` |
| `component-shas.txt`, `image.txt`, `model.txt`, `live-check-*.txt` | sha256 of every file h2 adds, image digest, checkpoint revision, live in-container read |
| `raw/<arm>/` | per-arm facts, engagement, faults, Xid base/new, watchdog status, lib / ttft / onem / probe / soak / BRRR JSON, gauntlet summary, spec metrics, per-class tables, gzipped boot log |
| `tf-compare-*.json`, `tfcmp-*.log`, `flipsm-*.txt` | teacher-forced compares vs A (79,544 positions, 39 docs) and margin-weighted flips |
| `errcount-*`, `health-*`, `launch-*`, `settle-pre-*`, `mem.tsv`, `xid-end.txt`, `preflight-host.txt` | per-arm error counts, smokes, launch transcripts, settle readings, host memory, kernel Xid journal at END |
| `args-/env-/binds-/ulimits-*-h1-REF.json`, `restore-*`, `watchdog-*` | the reference before/after the card, the restore of h1-REF at END |
| `promote/` | the h2-REF promotion: log, start log, inspect captures (h1-REF and h2-REF), 1M smoke, watchdog argv, gzipped boot/post logs |
| `guard-w12/` | Card W12 verdict (the #60945 backport verified), Card W11 fault-owner traces for arms S6 (split on) and S7 (split OFF), the desk analysis |
| `h1-lineage/` | Card H1 and Card PR1 verdicts, the h1-REF promotion log, v27 vs h1 args/env, H1 findings |
| `side-extension/` | SOURCES, BUILD-INFO, build/overlay SHA256SUMS, the cmake wrapper and its diff vs the ATT2 build, FlashMLA tree provenance |
| `launch/` | the box launchers as run (paths scrubbed), and `render-vs-live.txt` (the published launcher vs the live container) |

## Withheld or not included

- **Per-class prompt text.** The per-class fixture is private Hermes transcripts (prompt-file sha1 `618898aa`).
  Per-request `rows` are replaced by counts, and greedy `text` / `tool_calls` by sha1 + length. Every aggregate is kept.
- **Teacher-forced captures.** `tf-AP2-*.jsonl` is 2.6 MB per arm, and the TF corpus is not ours to redistribute.
  The compare outputs are included.
- **Large logs.** Full container logs (`full.log`, `postlog-*`, `.scan.log`) and the preflight `docker ps -a` are
  left out. The boot logs are kept, gzipped with `-n` after the path scrub.
- **Path scrub.** Box paths and LAN addresses are rewritten (`/home/<user>` → `~`, LAN IPs → `<station>` /
  `<lan-host>`). GPU UUIDs are kept, as in the earlier bundles.
