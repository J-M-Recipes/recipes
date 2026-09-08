# K1/K2 v3 review disposition

**Offline work only. Not a live release, collection receipt, or promotion decision.**
The implementation contract supersedes the historical plan where they conflict.
A pending repair is not a closed finding; full integration review is required.

## Original plan review

| Finding | Disposition | Remaining proof |
|---|---|---|
| Instance-count filtering misses some prefill and cannot authenticate decode | Replaced in the new helper by external timestamp-interval attribution, with exact input hashes and crossing rejection. Historical retrospective results stay frozen. | Real engine/trace phase markers are still absent; external labels are not authenticated by hashing. |
| Total tokens divided by median throughput is not aggregate decode wall | Prohibited by contract. Verdict uses median only as the declared client speed metric. | Full-precision per-request decode timestamps, summed elapsed time, and weighted throughput are not yet integrated into live evidence. |
| CaptureGate plus side-stream reads can race or remain closed | Rejected. No new background reader; candidate periodic stats remain disabled. | Engine-owned snapshot boundary and CUDA synchronization/capture/replay validation remain blocked. |
| STATS lines and graph launch counts cannot prove G0 | No G0 success is inferred. Pure snapshot helpers validate supplied raw counters only. | Real coherent complete layer snapshots and matching capture evidence are required. |
| G4 is both gating and non-gating in original plan | Contract resolves G4 as independently reported benchmark groups. Speed and exact-output checks are measurement gates, never promotion authority. | Lifecycle and provenance are separate acceptance requirements. |
| G1 must run before C2 | Archived verdict's C1 hygiene/quality check is required before C2. | Runner fault tests and independent lifecycle review must prove it is enforced. |
| Optional copy coalescing confounds K-only comparison | Excluded from v3 scope. | Candidate actual configurations must be verified to differ only in K and unavoidable lane identifiers. |
| Probe naming conflates median speed and accounting | Median is explicitly client-observed speed, not exact GPU/decode wall. | Live raw-duration integration remains prerequisite work, not an inferred result. |

## Integration review findings

- The initial runner's late `production_live_path=BLOCKED` annotation did not prevent live actions. The normal CLI now returns `WINDOW_AB_V3_PREP_BLOCKED` before invoking external commands. There is no production bypass flag.
- The incumbent-stop failure case now establishes restoration obligation before attempting stop. A behavioral test makes the fake stop mutate state and then exit unsuccessfully, and requires incumbent restoration.
- Candidate actual identity, absent profile reports, failed health checks, repeated cancellation during restore, transition release/deadline checks, and restore-only independence were not sufficiently proved by the first implementation. Repair and independent re-review are required; do not treat the earlier passing test count as approval.
- Phase helper review found additional numeric/alias and multi-interval normalization cases. These require behavioral regressions; decode wall/step denominators must never normalize all-phase totals.
- Snapshot delta validation must materialize a one-shot expected-layer iterator before validating both endpoints. Canonical string layer IDs use ASCII decimal spelling; Unicode aliases are rejected rather than normalized.

## Release boundary

The offline artifact can be reviewed and published only with its limitations intact.
It cannot become a live window by deleting a warning or toggling a flag. A future
release needs an explicit reviewed contract, instrumented evidence producers,
source/image binding, independent restoration protection, and live authorization.
No frozen v1/v2 receipt or historical verdict is rewritten by this work.
