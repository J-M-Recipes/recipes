# CARD F3 — one-user decode: Pg67c vs Flare's current one-GB300 core (pre-registered 2026-11-27, before any cell ran)

**Why.** The live recipe post and a drafted X post say our lane is faster for one user than Flare ("372 vs 303").
That pair mixes two harnesses, and the 303 is Flare's Nov 20 core. Their Nov 26 core's card lists 387.5 tok/s at C1
(their box, their measurement). James, 11/27: "if theirs is faster @ c1, no real point of posting this recipe" → "test run".
End state, James: "whichever one wins c1".

**Arms (same box, same night, one after another).**
- P1 = live Pg67c (GB300 + RTX PRO 6000 sidecar), `~/dsv41/sidecar-resid/work-Pg67c`, isolated to loopback.
- F26 = Flare, HF `local-inference-lab/DeepSeek-V4.1-Flash-1017-FLARE` @ `b55be385` (gb300 HEAD), core `d683cc1a…`,
  runtime `NVIDIA/hyperflare-inference` @ `7db06e0c` (the version the card requires) built with the repo Dockerfile,
  their README command unchanged: `--gpus 0 --mtp-draft-tokens 7 --embedding-table-storage ram`, default alpha.
  GB300 only (CDI device), signature and core id verified before the window.
- P3 = Pg67c again after restore (bracket), only when Pg67c is restored.

**Instruments.**
1. Primary: llm-inference-bench 0.6.2 (Local Inference Lab's client; `~/dsv41/m3/p3/lib`), C1 ctx 0, 30 s sustained window,
   `ignore_eos`, max 8,192 tokens, no thinking option sent (each engine's default), aggregate tok/s. Three runs per arm
   (one inside the full C1/C8/C32 × 0/16K table, two C1-only). Engine addressed directly (Flare :30016, Pg :30006).
2. Secondary: our per-class harness (`perclass_decode_f2.py`), C1, T=1, top_p 1, thinking off, exactly 256 tokens,
   40 prompts × 5 classes, seeds 42/43/44, pooled decode tok/s, mean over classes and seeds. Flare goes through the
   key-rename proxy (measured cost on Pg66: −1.1 %); no correction is applied.
3. Recorded, not part of the verdict: natural-length per-class, the C8/C32 and 16K cells of the lib table, and on the
   Flare arm `bench_full` (prefill, GSM8K-200, tool call) and the 12-task agent gauntlet.

**Warm-up.** One discarded natural-length per-class pass per arm before the measured cells (Flare's mover adapts to traffic;
the first rep after a Flare boot ran ~1.3× slow on 11/25).

**Verdict rule (f3_verdict.py).**
- An instrument is a win for a side only when that side leads by ≥ 3 %; a smaller lead is a tie on that instrument.
- Winner = the side with at least one win and no loss. One win each, or two ties = split.
- Fewer than 2 valid runs/seeds on either instrument for either arm = invalid.

**End state.**
- Flare wins → Flare is left serving on the box, loopback only (`127.0.0.1:30016`, key-rename proxy on `127.0.0.1:30006`),
  Pg67c stopped, MAINTENANCE kept with the bring-back command. Pointing LAN clients at it is a separate step for James
  (request fields differ: `chat_template_kwargs.thinking`, `reasoning_effort: none`).
- Pg67c wins, split, invalid, Flare fails to boot, or any abort → Pg67c restored with the production launcher, checked
  (env, ladder with rung 384, smoke, watchdog), then P3.

**Not claimed by this card.** Nothing about quality beyond the recorded smoke cells; nothing about concurrency or long
prefill as a verdict; one boot per arm.

**Receipts.** Box: `~/dsv41/flarecmp/results-F3/` (runner.log, F3-VERDICT.md, F3-verdict.json), prep `~/dsv41/flarecmp/prep-F3/`.
Scripts: `f3_prep.sh`, `f3_img.sh`, `f3_runner.sh`, `f3_verdict.py` (copies in this bundle).

**Disk note (same night, James: "del 1,2,4,5,6").** Removed from `/models` to make room: 8 stray Engram shard files in the
old Flare dir (189 GiB, part of no served core), and four checkpoints with Ark copies, each sha256-equal to the Ark's
SHA256SUMS before deletion: DeepSeek-V4.1-Flash `df42c109` (51/51 files), GLM-5.3-NVFP4 RadixArk `6e389189` (50/50),
GLM-5.3-ABLITERATED-NVFP4 `371bdb98` (284/284), GLM-5.3-Flash FP8 zai `04c4e9e9` (64/64).
