# G3 desk — fail-closed sidecar watchdog for the Garner tree + #61161 on v0.31.0 (2026-10-29, lane untouched)

## 1. #61161 check on v0.31.0 — NOT PRESENT

| file | v0.31.0 sha1 | af7f9488 stock (L3 orig) | production overlay |
|---|---|---|---|
| v1/worker/gpu/sample/gumbel.py | 70ae04f2 | 70ae04f2 | 6eac3e13 (L3) |
| v1/worker/gpu/spec_decode/rejection_sampler_utils.py | fb99972e | fb99972e | edc8f8f4 (L3 + #60945 guard) |
| v1/worker/gpu/spec_decode/dflash2/speculator.py | 7a2333ce | 7a2333ce | 896f9b5c (L3; not on the dspark path) |

v0.31.0 ships the pre-#61161 sampler, and his release overlays (1017, 58132) don't touch it. His fastpath reimplements
the same unmixed draft draw in two fused kernels: `w5_markov.py`, through `gumbel_noised_argmax`, and
`w5_markov_gemv.py`, which calls murmur3 directly. His docstring says so outright: "no `spec_step_seed`". With
block verification and probabilistic drafting (his launch), re-drafted uncommitted positions reuse the draw that was
just rejected. Upstream measured a biased marginal (block: P = 0.417 vs 0.500; probabilistic + block: 0.356 vs 0.300).
**Every T > 0 result from his tree — including his own, since he forces T = 1.0 — is from a biased sampler.** G1f/G2
quality gates are T = 0 or teacher-forced (unaffected); per-class temp-1.0 rows are speed only.

Port (built on the box, py_compile OK, **not yet run on a GPU**):
- `$GR/ovl-61161/` = his overlay-58132 + production's exact `gumbel.py` (6eac3e13) and `rejection_sampler_utils.py`
  (edc8f8f4: L3 + guard). Byte-identical bases, so no rebase. Select with `DS41_OVERLAY_58132_DIR=$GR/ovl-61161`.
- `$GR/fastpath-61161/` = pol fastpath + one line in each fused kernel after the seed load:
  `seed = seed ^ (tl.cast(step, tl.int64) << 32)`; `step` is the q-cache column there, i.e. production's `col`.
  Select with `DS41_FASTPATH_DIR=$GR/fastpath-61161`.
Owed on a GPU (dark card): his `kernels/w6-moe/markov` exactness test (fused == eager bitwise, both patched), the
L3 rejection-sampler tests, a T = 1.0 speed row (expect ≈ 0), and a Pg-61161 rerun of the T = 0 gates (expect bitwise
or noise-floor equal — T = 0 never reads the seed).

## 2. Fail-closed watchdog for his sidecar — BUILT, unit-tested, read-only smoke on the live lane

Finding: **on a peer tier v3 wait timeout (PEER3_SPINS polls, default 20,000,000) all three v3 wait sites skip the
call's cold-expert adds and keep going** — `peer3_gb300.cu finish_kernel` (`if (mine && ok)`), the boundary fold, and
the MegaMoE fork's W6 fold ("2 = spin bound exhausted (the call's cold adds are skipped)"). The timeout count lives
only in GB300 device memory. That is the silent-wrong-tokens case. (His v2 `_wait` is different: it bumps words[2]
and TRAPS, so v2 already fails loud.)

Fix = two pieces:
- `PEER3_SPINS=1000000000000` (plain digits — `fastpath_peer3` parses with `int()`) so a dead/late RTX hangs the GB300
  instead. Forwarded by a sixth no-op-unless-set pol hook, `DS41_PEER3_SPINS` (one line in pol `serve.sh`, backup
  `serve.sh.pre-peer3spins`; `make_pol_tree.py` updated). The fold control word is int64 — fits.
- `sidecar_watchdog_p3.py` (clone of production v2, production untouched): v2 timeouts/stall as before + v3 stall
  (GB300 published tag @rec[0] ≠ RTX done tag @128, both frozen > 2 s) + sidecar container down + segment
  unreadable → `docker kill` server, stop sidecar, fail-stop. **Refuses to arm (exit 3) unless the server env carries
  PEER3_SPINS ≥ 1e12 as digits.**

Verified: 13/13 unit tests (M4 and box). Read-only `--once` against live h2-REF parsed v2 words `[179640, 179640, 0]`,
flagged the absent tier3 segment, refused to arm without PEER3_SPINS (rc 3). Production watchdog pid 3825666
`trip: null` throughout.
Owed on a GPU (dark card): kill `ds41-sidecar` under C8 load on Pg → trip ≤ ~3 s, server killed, clients get errors
not tokens; confirm rec[0].word0 carries every call's tag on a live segment; steady-state speed with 1e12 spins (expect 0).
