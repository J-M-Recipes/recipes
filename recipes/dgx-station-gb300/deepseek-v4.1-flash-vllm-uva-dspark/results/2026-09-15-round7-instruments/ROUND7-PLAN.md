# Round 7 — evidence correction, matched replay, closed-loop tools

Owner: Milo. Authorized by James's `go` following the proposal to correct Round 6, compare v13/v14 on matched recorded sessions, and run a bounded real tool-execution gauntlet. Prepared September 15, 2026. This is preparation, not a live-result receipt.

## Scope and invariants

- Reference stays **v14**. No promotion, model reconfiguration, credential changes, provider routing changes, new image pulls or OFFGB experiment in this release.
- Station was verified idle at 06:34:58 CDT. Recheck before each start; refuse unrelated running containers or compute processes. Never touch ports 30001 or 30003 or their containers.
- Only start/stop the two preserved containers below. Keep them intact. Final state is **both stopped / port 30006 dark**, matching entry state and James's no-restore-on-done rule. Do not resurrect another incumbent.
- Requests stay on Station loopback. Do not enable dev endpoints, change binds, or expose an RPC/reset interface.
- Long execution must survive the chat process: Station-side system-scope service, bounded phase subprocesses, host lock, exact run-id release, independent system-scope shutdown timer, durable append-only receipts. Read back watchdog command/identity/deadline before starting either model.
- If unrelated ownership appears, identity/source mismatch occurs, a hard deadline fires, sandbox controls fail, or capture is incomplete, stop the owned run, retain partial evidence and report the blocker. Do not infer PASS from a vanished process.

## Verified baseline identity

- Image ID for both: `sha256:00d577a6a63281e15336029d5bcee4e9a2cf182214a4f20ba6111b1c8e79893d`.
- v14: `dsv41-vllm-v14-1M-ksched-agent-BOUND-REF`, ID `e58cb5ae5e3790456b9e1ecdb9adef591674a85e4e9184896a21c3b8b487f9f5`.
- v13: `dsv41-vllm-v13-1M-ksched-BOUND-REF`, ID `2e82579972837db31adcda5951198cbd8ef6e8b5173c660a1e10968a6508a6a1`.
- Exact environment hashes matched: `860c45c9ff39513181d284246bc75e8dbc8986d380f8bea4e750b0b1cb8d2bda` (hash of sorted Docker Config.Env JSON).
- Model mount is the same read-only local `df42c109f1defefcbfcedbe7d905718a12266e40` revision; offload60/util0.97/context1048576/max-seqs16/max-batched8192/long-prefill6144 are unchanged. The actual Cmd arrays differ only in `num_speculative_tokens_per_batch_size`: v14 `[[1,4,5],[5,16,1]]`, v13 `[[1,2,5],[3,16,1]]`.
- Existing restart policies are `no`. These are preserved, not changed.

## A. Correct existing evidence, without changing raw captures

Publish explicit corrections in the recipe bundle, YAML, README and blog discovery surfaces. Retain 171.1/174.1 vs150.0/151.7 and232.6/275.2 as observed proxy rates, not a causal cache gain or production capacity. Withdraw original wall-rate CI and exact-floor claims. Note the session-label collision. Verify the recipe PR's exact-head CI, merged content, and each deployed blog file by live bytes/hash after rollback backup.

## B. Freeze and qualify the instruments offline

Two disjoint source modules are being implemented under `replay/` and `gauntlet/`, with vertical TDD. Before live scored use, run focused and full tests, exact-source SPEC and QUALITY/safety review, archive source and tests with hashes, and validate the extracted release bytes. Publication of source is not permission to promote a configuration.

### Recorded-history replay

- Preserve the original PRIVATE ordered proxy's twenty distinct context sequences, fifteen turns each. Replace ambiguous session labels with collision-free entry-based opaque IDs. Never publish the transcript fixture or raw private response text.
- Preserve and disclose proxy limitations: flattened old tool history, shortened system prompt and capped context; no live tool execution or generated-output feedback. This is not a native Hermes capacity test.
- Four workers, session-serial, identical fixture/request bytes on both configurations. Temperature0, max_tokens400, supported reasoning effort `low`, with no conflicting `thinking=false` override. This changes reasoning mode versus Round6; do not treat new/old throughput as a same-condition comparison.
- Use the supported request `cache_salt` field for fresh namespaces, not dev-mode cache reset. First prove live isolation with the same harmless prompt: A-first, A-repeat, B-first; require counter evidence of cache reuse only in A-repeat. Warmup uses a different namespace. Record namespace in every request/hash.
- One initial-namespace pass then one identical-request repeated pass per boot. Use the same pair namespace on the two configurations; use new namespaces for the next pair. Fresh namespaces and boot identity receipts define initial state, not the word cold.
- All300 turn identities must appear exactly once. Missing/duplicate records, incomplete SSE, absent usage, nonfinite counters or decreasing counters invalidate capture. Retain complete private outputs, usage, finish reasons, first-token boundaries, start/end times, request hashes and counter deltas immediately as JSONL.
- Report completion tokens/measured makespan. Busy-worker estimate is a separate diagnostic, never its CI. Three boot repetitions per configuration yield descriptive mean/range; do not manufacture statistical significance from independently resampled correlated turns. Report TTFT quantiles and per-session durations.

### Real tool-execution gauntlet

- Twelve unique public-safe tasks: four sequential file chains (at least four tool turns), four small code repairs with objective tests, four file-grounded structured answers. Freeze task IDs, requests, checks and boundaries before scored responses.
- Native assistant tool_calls and matching tool-role IDs; actual returned answers/tool results feed the next turn. Tools are bounded read_file/write_file/run_tests only. No generic shell, live account, email, production file, secret or network task.
- Generated code runs only in CPU Docker with runtime runc, NVIDIA_VISIBLE_DEVICES=void, network none, read-only root, dropped capabilities, no-new-privileges, process/memory/CPU/time limits. Mount only a fresh task directory and trusted evaluator. No GPU, home directory, docker socket or credentials.
- Exercise correct, wrong and early-exit controls in that real sandbox before model scoring. Synthetic HTTP tests are not proof of sandbox execution.
- Fixed task boundaries:12 turns,1500 max tokens per turn,180 seconds per task; same settings for both configurations. Any justified change after a pilot creates a new explicitly versioned contract before scored use.
- Separate transport/protocol validity, task correctness, tool errors and budget exhaustion. A valid tool-call delta is not a success. Strict full-string JSON where requested; no prefix salvage. Preserve failures, no opportunistic retries.
- Run at four workers, once per boot. Repeats are clustered by the twelve unique tasks, not treated as72 independent tasks. Report per-task correctness, failures and successful completion time, plus TTFT tails. This is a bounded gauntlet, not broad model-quality certification or full Hermes harness latency.

## C. Scored order and decisions

Boot order: **v14, v13 / v13, v14 / v14, v13** (three paired blocks). Each boot: identity/readiness → canary/warmup in non-scored cache namespace → initial replay → identical-request repeat → closed-loop gauntlet → stop exact container and verify dark state. Before starting scored boot1, pilot request compatibility and sandbox controls; pilot data is never pooled with scored data.

Scored completion requires six boot receipts, twelve complete replay captures, six complete gauntlet captures, and final shutdown proof. Complete means coverage and validity, not perfect task correctness. Baseline failures remain visible and can block recommendations without erasing valid measurements.

Decision: retain v14 absent clear contrary matched evidence. No automatic promotion. Describe the observed paired speed differences, TTFT/error tradeoffs and task-level wins/losses; three repeats/twelve tasks do not justify a blanket quality or capacity claim. An inconclusive outcome is a result. Publish new findings only after audit, not merely because tests passed.

## Time bounds and pending release material

Nominal budget after source review: approximately1.5–2hours from historical five-to-eight-minute boots and roughly nine minutes per replay pair, plus bounded tool tests and audit. This is planning, not a clock ETA; update from the first real boot/pass. Hard campaign shutdown:4hours after released start. Readiness bound20minutes, replay pass bound20minutes, gauntlet bound15minutes; kill whole phase process groups on timeout, then stop only owned model IDs.

Pending before release: source/test hashes, full sanitized inspect snapshots, reviewed runner/shutdown helper, sandbox-control receipt, live cache-salt compatibility receipt, independent reviews, source publication and exact archived release readback. None are claimed complete by this plan.
