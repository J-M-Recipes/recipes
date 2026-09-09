# C1-only restore-protected quality gate

## Behavior

`./scripts/window_c1_quality_gate.py` is the public C1-only lifecycle entrypoint for `c1-quality-20260909`.

It fails closed unless all of the following hold:

1. The output directory is empty and both the output lock and host operation lock are acquired.
2. Runtime dependencies are copied into an immutable `restore-bundle` and the live execution path is rebound to archived helpers.
3. The candidate name `glm53-big-c1-quality-k1` is exactly absent before any restore timer arm or incumbent stop.
4. The incumbent `glm53-big-sc13g-mtp-ctx512k-keep-pre-dflash2-20260907` is proven running with the expected ID, image, model, `ctx524288`, `max_seq=1`, and MTP K1.
5. The incumbent greedy quality keys `0..19` are captured and validated nonempty before the independent restore timer is armed.
6. The system-scope restore timer is armed and read back before the incumbent is stopped; its `ExecStart` must be the archived `--restore-only` command and must not contain `/Users` paths. The timer-private bundle contains its own archived health checker. If post-window restoration proof fails, the timer remains armed.
7. Exactly one candidate is launched with `STATS_SEC=0`, `MAX_MODEL_LEN=524288`, `MAX_NUM_SEQS=1`, and `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`.
8. Authenticated readiness and `dflash2_acceptance_probe.py --max-tokens 512` must pass.
9. Candidate greedy quality keys `0..19` must be nonempty and bit-identical to the incumbent. The human `greedy_equiv.py --compare` output is recorded, but exit code 0 is not trusted for the gate.
10. Historical speed hygiene is enforced through archived `window_ab_verdict.c1_gate(..., baseline_speed=45.65)`.
11. The candidate is stopped and proven stopped, the incumbent is restored and authenticated, and only then is the independent restore timer cancelled.
12. The final receipt always sets `promotion_authorized=false` and `c2_authorized=false`; a PASS only clears this separate C1 quality gate.

No prose/code benchmark battery is run. The entrypoint contains no C2/K2 constants, names, specs, or paths except the required false authorization receipt field.

## Provenance

Resolved gate constants:

- run id: `c1-quality-20260909`
- candidate: `glm53-big-c1-quality-k1`
- speculative config: `{"method":"mtp","num_speculative_tokens":1}`
- `STATS_SEC`: `0`
- acceptance probe max tokens: `512`
- historical baseline speed: `45.65`
- base published source commit: `f7912f9055d524b387d154db5defe8683a50c9ce`

Stabilized local source hashes:

- `scripts/window_c1_quality_gate.py`: `69359ba38aba6abfdf5a18a9cf35d0b5fa9bf713be13f951fc82d418cb30b571`
- `tests/test_c1_quality_gate.py`: `86d47efb44ec771bcd4c4b1ee522b9098f9777aab89fd894a8cffc763b5ef7e4`

## Local verification evidence

Strict TDD evidence was produced locally:

- RED: `python -m pytest tests/test_c1_quality_gate.py -q` failed with 9 failures while `scripts/window_c1_quality_gate.py` did not exist.
- GREEN: `python -m pytest tests/test_c1_quality_gate.py -q` passed with `12 passed` after adding archive-completeness, failed-restoration failsafe, exact incumbent-model, and exact digest-qualified Docker image regressions.
- REGRESSION: `python -m pytest -q tests` passed with `246 passed in 85.41s` for the exact runner and test hashes above.

Focused coverage includes ordering, systemd `ExecStart`/deadline readback rejection, K1-only launch environment, missing/empty greedy rejection, 19/20 mismatch despite compare rc0, restoration after stop mutation, restore-before-timer-cancel ordering, receipt authorization flags, independent restore-only, candidate inspect ambiguity/redaction, and SIGTERM active-child cleanup via existing `run_logged` process-group behavior.
