# Frozen contract — E2 cache policy/allocation and E3-A trace predictor

Date frozen: 2026-09-07 CDT  
Status at freeze: replay process started; result file did not yet exist  
Execution surface: M4 Max only; the GB300 live container is read-only and remains serving

## Question

Can the existing full-GLM-5.3 router corpus justify either of the two remaining slot-cache speed levers before another live relaunch?

1. **E2:** reduce demand-fill bytes with a better replacement policy or per-layer allocation under the exact same 5,792-slot HBM budget.
2. **E3-A:** predict nonresident experts accurately enough to justify building graph-compatible side-stream prefetch.

This is an offline screen. It cannot prove an end-to-end speedup. A passing candidate earns one live implementation/bench window; it is not promoted by simulation.

## Frozen inputs

Source host path: `/home/milo/big-v1-campaign/trace/r1-base/`

| file | SHA-256 |
|---|---|
| `trace-361.i16` | `8603fa87aa8845bb89b368b45bae4a370277df3170913fab3b241e359dd56480` |
| `trace-361.steps` | `00ce58df2ab763aefdcbc7be298591a7047e0adb52fab2f5b507d05d91b31ef1` |
| `trace-361.slots.i64` | `7f9eee9e46b29352ea5f3ae2f4817aa89b14f5a9753dff14af2d615ba59c286e` |
| `marks.txt` | `eb7346e706e4bedcbb3b2b4e693c0b52feb09e2ce440c99e6193514b2e31a7e2` |
| `slots-5792-ctx512k.json` | `73962276498626af16baaa7c8737384960aff8718d254e66ab92610570c9743c` |

The remote and local hashes for the trace, step, slot sidecar, and two slot maps were compared before the run and matched exactly.

Expected corpus facts from the original analyzer: 79,119 routed tokens; 71,210 decode tokens; 7,909 prefill tokens; 78 traced layers; top-8; 256 experts. Workload markers: `bench_and_gates`, `prose_low`, `code_low`, `tools_low`, `prose_max`, `code_max`.

## Shared replay rules

- Decode is defined exactly as the tracer did: a captured step with `n <= 8`; prefill rows are excluded.
- Each marked workload segment is split chronologically: first 70% training, final 30% held out.
- Nothing from a held-out row may train a policy, allocation, or transition matrix.
- Per-layer cache state is warmed on that segment's training prefix; only held-out accesses count in results.
- Negative expert IDs are sentinels and are ignored.
- Expert accesses remain in recorded order.
- The HBM slot budget is fixed at exactly 5,792 across layers 3–77.
- Allocation candidates use capacities 32–256 in 16-slot increments.
- Allocation curves may use at most 6,000 chronologically sampled training rows; held-out scoring uses every held-out row.
- Deterministic ties prefer the smaller layer/expert ID.

## E2 arms

Each policy is scored both with the frozen 512K allocation and with a reallocation learned from training-only marginal hit-rate curves:

1. LRU (current policy/control)
2. global-count LFU with LRU tie-break
3. static top-frequency
4. 25% static-hot + 75% dynamic LRU
5. 50% static-hot + 50% dynamic LRU
6. 75% static-hot + 25% dynamic LRU

### E2 gate

A candidate passes the offline continue gate only if both are true versus frozen-allocation LRU:

- aggregate held-out hit rate improves by **at least 5.0 percentage points**; and
- held-out `tools_low` hit rate improves by **at least 5.0 percentage points**.

A pass earns implementation plus a matched live C1/C4/C8 bench. Live promotion still requires C1 at least 5% above 55.27 tok/s, matched quality, and a verified restore path. A simulation loss stops the policy branch.

## E3-A predictors

Predictors issue only positive-evidence expert rows not already resident in the frozen LRU cache. Predictions do not mutate cache state during this screen.

1. **Adjacent-layer:** current token's routed expert set at layer L−1 predicts layer L.
2. **Previous-token:** the prior C1 decode token's expert set at the same layer predicts the next token; prefill or batch steps reset the chain so unrelated requests/batch lanes are never paired.

Transition matrices are trained only on the union of segment training prefixes. Prefetch budgets: 1, 2, 4, and 8 expert rows per target layer.

Metrics:

- `issued`: nonresident rows that would be copied;
- `useful`: issued rows actually demanded at the target;
- `precision = useful / issued`;
- `recall = useful / true demand misses`.

### E3-A gate

The best strategy/budget passes the offline continue gate at aggregate held-out precision **>= 0.50**. Recall, issued count, and per-segment precision must be published; a high-precision but negligible-coverage arm is not automatically useful even if it passes the literal gate. A pass earns a graph-compatible side-stream design/prototype, not a live promotion.

## Required receipts

- exact command and harness SHA-256;
- full machine-readable result;
- compact human verdict with per-segment table;
- input manifest and checksums;
- harness tests, repository full tests, recipe validator, syntax checks, and `git diff --check`;
- proof the live 512K/MTP service was never stopped by this offline stage.
