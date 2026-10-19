SUFFIX WIN
### Performance
| arm | RT C1 (acc) | C4 | C16 | C32 out | TTFT 8K | cold 64K TTFT | knee r2 C1/C8/C16 | replay r1/r2 | gauntlet | KV tokens | boot | autotune |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CTL1 | 186.5 (2.97) | 318.7 | 522.9 | 504.4 | 0.66 | 2.20 | 192/618/937 | 412/441 | 12/12 | 2504891 | live | hit 62426808 (Loaded 189, saved-lines 6) |
| S0 | 190.2 (3.00) | 325.1 | 523.5 | 506.5 | 0.65 | 3.78 | 192/590/942 | 301/438 | 12/12 | 2504891 | 464s | hit 62426808 (Loaded 189, saved-lines 1) |
| CTL2 | 193.5 (3.02) | 335.9 | 540.2 | 515.8 | 0.63 | 2.18 | 193/619/947 | 314/437 | 12/12 | 2504891 | 316s | hit 62426808 (Loaded 189, saved-lines 1) |

### Host memory (steady state, after all benches)
| arm | pinned /dev/zero maps (n · Size GiB) | cgroup shmem GiB | cgroup memory.current GiB | host Shmem GiB | MemAvailable steady / after drop_caches GiB | Mlocked / Unevictable GiB | hook host_avail L9 / min / rehome-done GiB | pinned_expert GiB |
|---|---|---|---|---|---|---|---|---|
| CTL1 | 192 · 327.96 | 327.96 | 352.61 | 328.03 | 118.38 / 131.44 | 0.00 / 0.01 | 12.10 / 11.52 / 12.10 | 62.33 |
| S0 | 192 · 252.79 | 252.79 | 419.33 | 252.86 | 193.33 / 205.92 | 0.00 / 0.01 | 197.68 / 152.69 / 197.68 | 62.33 |
| CTL2 | 192 · 327.96 | 327.96 | 349.63 | 328.03 | 124.66 / 132.73 | 0.00 / 0.01 | 11.29 / 11.29 / 11.29 | 62.33 |

**Candidate arm:** S0  ·  **Verdict: WIN** (autotune hash hit 62426808)
- WIN bar host pinned down >= 40 GiB: PASS — freed 75.17 GiB (CTL mean 327.96 -> S0 252.79)
- WIN bar RT C1 >= -2%: PASS — 0.1%
- WIN bar replay r2 >= -2%: PASS — -0.2%
- WIN bar gauntlet 12/12: PASS — 12/12
- WIN bar non-tool greedy parity vs both CTL: PASS — S0~CTL1 16/16, S0~CTL2 16/16 (CTL1~CTL2 16/16; all-prompt S0~CTL1 17/18, S0~CTL2 17/18, CTL1~CTL2 17/18)
- WIN bar KV tokens unchanged: PASS — 2504891 vs 2504891
- FAIL bar decode −3% (Δ C1/C4/C16/replay r2 = 0.1, -0.7, -1.5, -0.2 %): clear; non-tool parity mismatch: clear; boot death: clear
- other Δ vs CTL mean: C32 out -0.7%, TTFT 8K 1.0%, cold 64K 72.6%, knee C1 0.0%, knee C8 -4.6%
- host: MemAvailable steady Δ 71.81 GiB; after drop_caches Δ 73.84 GiB; hook guard L9 host_avail Δ 185.99 GiB; cgroup shmem Δ -75.17 GiB
HOST_FREED_GIB 75.17
