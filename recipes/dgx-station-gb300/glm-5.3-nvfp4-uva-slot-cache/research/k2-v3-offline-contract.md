# K1/K2 v3 — offline implementation contract

Status: **NOT RELEASED FOR LIVE EXECUTION**. This document qualifies local implementation work, not a Station window or K2 promotion.

## Scope and provenance

- Base source: `2a85ba181369357f1c4d057ff1cdd23d88b162dd`.
- Implementation: Milo on `gpt-6-astra` via `openai-codex`, with bounded implementation/review subagents; their reported model provenance is separate from this parent runtime.
- Work is isolated from main. No Station access, service mutation, credential access, dependency installs, commits, pushes or public-page updates are part of this offline pass.
- Frozen v1 and v2 results remain byte-for-byte unchanged. Existing v2 runner remains unchanged; new code imports tested primitives where appropriate.
- No optional cache policy, coalescing, slot budget or context changes. K1/K2 must differ only in speculative depth and unavoidable run identifiers.

## Scientific contract

1. Instance counts and duration histograms may suggest phase or bytes moved. They cannot certify phase boundaries or runtime mask population.
2. Exact GPU phase analysis takes a GPU trace plus explicit intervals in the same clock domain, with source provenance and integer-nanosecond timestamps. Crossing kernels require rejection or separately reviewed policy; they must not be silently assigned. Missing trustworthy intervals blocks exact decode claims.
3. Existing acceptance-probe `decode_tok_s` is a client metric. Validate all four raw rows, finite positive speeds and durations, metric-counter consistency, and recompute the summary median. Do not derive GPU wall time by dividing total tokens by that median.
4. G1 is a cheap pre-C2 check: validated candidate K1 throughput at least 99% of the historical 45.65 reference and exact nonempty outputs for prompts 0–19. Historical reference is a hygiene threshold only; it is not a matched contemporary control or a statistical confidence interval.
5. G2 requires matched K2 probe speed at least 1.05 times K1. G3 requires the same bounded exact-output regression check against the incumbent. G4 benchmark groups are reported independently, not ambiguously both gating and non-gating.
6. Measurement checks alone never authorize promotion or final PASS. Source/receipt identity, matched configuration, actual capture validity, unconditional restoration and independent review remain separate gates. Missing evidence produces INCONCLUSIVE. Valid negative results may be STOP; promotion requires James's separate decision.

## Telemetry contract

Do not introduce a background CUDA reader guarded only by `is_current_stream_capturing()`. That observation has a check/use race, is stream-local, and does not bracket global engine capture. Keep periodic stats disabled for candidates until a real engine-owned quiescent snapshot boundary is established.

Offline helpers may validate captured counters, exact layer sets, nonnegative integral deltas, and aligned steps. Such tests establish schema/aggregation correctness only. They do not establish CUDA capture safety, synchronization, or real runtime hit rate. An unused validated helper is explicitly an integration prerequisite, not a claim that live telemetry works.

## Lifecycle requirements

- Archive every executed source input and helper, bind digests, reject missing/mutated sources and unsupported configuration. Use exact immutable image identity and verify actual candidate settings.
- Establish incumbent identity and independent release-independent restoration before allowing stop. Set restoration obligation *before* attempting stop, since a command can mutate and then fail.
- Shared host/output locks; fresh output directories; no log credential values; no automatic cleanup of historical receipts/containers.
- K1 failure means no K2. Recheck release between transitions. Reserve restoration budget and enforce in-flight command deadlines; a launch-only deadline check is insufficient.
- Signal/error path terminates child process groups before restoration. Restoration stops and proves both candidate containers inactive before restarting/proving the pinned K1 incumbent.
- Restore-only must not depend on the experimental release, operation lock or a mutable runtime checkout. Repeated signals must not silently abort restoration.
- Restore and prove service before profiler export or expensive analysis. Preserve raw report bytes and attest input/output hashes.
- A fake executable test is behavioral evidence for orchestration only. It is never a real GPU/profile receipt.

## Offline acceptance

- Each new production behavior has observed RED followed by GREEN, with regression checks.
- Existing full test suite, recipe registrations, generated index and diff hygiene checks pass from the isolated worktree.
- Independent review identifies no unresolved blocker within the claimed offline scope.
- Final report explicitly separates implemented/tested behavior from live blockers. No PREP_OK/GO statement until the scope named by that statement is actually proven.

## Live prerequisites still requiring proof

- Engine-owned safe counter snapshot mechanism and real capture/replay validation.
- Trace-aligned phase/step markers and verified warmup exclusion.
- Current immutable release package, reviewed fail-safe command, exact live incumbent identity and available restore budget.
- Explicit live-window authorization and release. Offline tests cannot satisfy these prerequisites.
