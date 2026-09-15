# Round 7: matched replay and bounded tool execution

This directory is an **instrument and release contract**, not a new performance result. v14 remains the reference. No model, offload, context, routing or image change is authorized by running this instrument.

## Questions and populations

1. Compare v13 and v14 on exactly the same private recorded-history workload: twenty distinct context sequences, fifteen turns each, four session-serial workers.
2. Separately compare a twelve-task public-safe, closed-loop gauntlet: four file chains, four code repairs and four structured answers. These are twelve unique tasks repeated across boots, **not seventy-two independent tasks**.

The history fixture is a flattened, truncated proxy, not native Hermes messages or live agent execution. It stays private; its content hash is in `container-contract.json`. The gauntlet executes only bounded local fixture tools. Neither instrument establishes general production capacity or broad model quality.

## Frozen request and run rules

- Preserved v14/v13 containers, immutable image and launch-configuration digests: `container-contract.json`. Only the speculative batch schedule differs. Do not recreate or delete either container.
- Boot order: **v14, v13 / v13, v14 / v14, v13**. Do not add repeats to chase a preferred result.
- Per boot: verify identity/readiness; non-scored C1–C4 warmup; initial-namespace replay; identical-request repeat; closed-loop gauntlet; stop the owned container and verify a dark port.
- Warmup labels denote actual concurrent requests, not names on serial calls. Three input-size bins are **characters**, not tokenizer-measured tokens. Warmup uses a separate cache namespace.
- Replay: temperature 0, seed 42, reasoning `low`, 400-token cap, four workers. No conflicting `thinking=false` flag. This differs from Round 6's reasoning configuration. Fixed seed does not establish bit-identical outputs.
- The same pair namespace is used on both configurations; each next pair gets a fresh namespace. Initial-namespace replay includes within-sequence prefix reuse. The second pass is an **exact request repeat**, not newly continuing sessions or a production steady state.
- Gauntlet: temperature 0, seed 42, reasoning `low`, four workers, twelve model-turn/1,500-output-token-per-turn/180-second task budgets. Failed answers remain in the denominator; no score retries. The requested final format is strict full-string JSON with fixed object keys and decoded JSON types.
- File-chain success requires the actual required reads across at least four distinct model turns. A batch of calls is not four sequential turns. JSON parsing, task-schema conformity, transport/protocol validity and objective correctness are separate fields. A token-limit finish cannot become overall success merely because its partial answer looks correct.
- Tool results feed the next model turn. Container-launch command metadata stays in the receipt but is removed from model-facing tool feedback, so changing paths and sandbox UUIDs do not change semantic prompts.

## Measurement and acceptance

Scored completion requires six boot receipts, twelve complete 300-turn replay captures, six complete twelve-task gauntlet captures and final shutdown proof. Capture completion does not imply successful task answers.

Replay rate is the sum of completion tokens divided by measured run makespan. Busy-worker estimates are diagnostics, not the wall-rate estimator and not its confidence interval. Report the three boot-pair differences and descriptive range separately for the initial and repeated-request phases. Do not claim statistical significance from independently resampling correlated turns. The pre-existing 1.5% wash convention is descriptive, not a substitute for repeat evidence.

The final auditor must independently verify unique request/task coverage, pairwise request hashes and settings, token totals against raw usage and server deltas, finite/monotonic counters, timestamps, clipping/finish reasons, phase identity and absence of unrelated serving traffic. Report differences in generated answers rather than assuming temperature-zero identity. Missing evidence, contamination or mismatched profiles must not be pooled into a winner.

For the gauntlet, report per-task successes/failures, schema/protocol errors, budgets, successful completion times and request TTFT tails. The evaluator checks a finite public test corpus; passing it does not prove generalization or resistance to every deliberate test-gaming strategy. Correct/wrong/early-exit/forged-marker controls are explicit tested cases, not a claim of a universally tamper-proof Python grader.

**No automatic promotion.** A null, mixed or inconclusive outcome is valid. Neither one pilot nor a passing software suite selects a serving configuration.

## Safety and privacy

- Live requests use Station loopback `127.0.0.1:30006`; no new binds, dev endpoints, credentials or reset RPCs.
- Start/stop only the preserved IDs in the contract. Refuse unrelated Docker or CUDA occupancy; never stop a foreign process to make room. Leave both references intact and stopped at the end. Do not restore another incumbent or touch ports 30001/30003.
- Before starting, archive and hash **all** executable source, tests and contract; obtain exact-source SPEC and QUALITY review; publish the source with exact-head CI; extract the release and verify its hashes again. Include the run-specific `CONTROL=RUN` and exact `RELEASE` token in the launch manifest. Refuse reused output directories. Releases under the same deployment parent share one flock.
- The operator must arm and read back an independent system-scope timer with a **four-hour** deadline and an exact `/usr/bin/docker stop --time 30 <v14-id> <v13-id>` command before starting either model. The controller checks the active timer and exact stop command; the operator records the actual timer deadline. Preserve this guard until shutdown has been independently verified.
- Run in a system-scope service as the unprivileged Station owner, with private umask, durable logs and bounded readiness/phase runtimes. Normal failures and signals stop only owned containers; a lost start acknowledgement also takes the stop path. A forced kill can bypass Python cleanup, which is why the independent timer is mandatory.
- Readiness limit 20 minutes, replay limit 20 minutes per pass, gauntlet limit 15 minutes, campaign limit four hours. These are bounds, not estimates of completion time. Keep failed and partial artifacts.
- Generated Python runs only inside the pinned CPU Docker sandbox: explicit `runc`, `NVIDIA_VISIBLE_DEVICES=void`, network none, read-only root, non-root user, dropped capabilities, no-new-privileges, bounded memory/CPU/processes/tmpfs/time. Mount only a fresh task directory and the read-only evaluator. No host home, Docker socket, GPU devices or secrets.
- Prove the real sandbox controls before scoring. Tests mocked at HTTP/process boundaries are labelled synthetic and are not hardware or model results.
- Private request bodies, outputs and lossless SSE evidence are written to private run artifacts. **Never publish the history fixture, raw replay requests/completions, or base64 SSE fields.** Publish only audited allowlisted aggregates and source. The public-safe gauntlet is separate from that private corpus.

## Offline checks

From this directory, using Python 3.12 on Station/CI:

```bash
(cd replay && python3 -m unittest discover -s tests -v)
(cd gauntlet && python3 -m unittest discover -s tests -v)
(cd controller && python3 -m unittest discover -v)
```

The real Docker control test is intentionally skipped unless `R7_SANDBOX_IMAGE` is explicitly set on the qualified Linux host to the immutable image in the contract. It never executes candidate Python on the host.

Inspect the real CLI with:

```bash
python3 controller/controller.py --help
```

A launch requires explicit `--release`, `--contract`, `--fixture`, `--output`, `--run-id` and `--guard-unit`. `--validate-only` checks source/input binding without proving runtime readiness or granting launch permission. Do not use a source example as a substitute for the operator's live identity/guard checks.

## Compatibility pilot provenance

Excluded live pilots established ordered-replay capture, a cache-namespace control and a three-task native tool loop. Their private receipts are retained outside this source package. They are not a matched comparison or a score for the final twelve-task instrument. Subsequent fixes, including explicit seed and stricter grading, make the final instrument a separate version. No scored outcomes are included in this package, and pilot receipts must never be relabelled as scored data.
