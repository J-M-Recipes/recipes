SUFFIX FAIL-PROV
| arm | RT C1 (acc) | C4 | C16 | C32 out | TTFT 8K | cold 64K TTFT | knee r2 C1/C8/C16 | replay r1/r2 | gauntlet | KV tokens | KV mem GiB | graph capture | boot | autotune |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CTL1 | 188.7 (2.97) | 326.8 | 537.4 | 511.2 | 0.64 | 2.17 | 193/585/950 | 421/410 | 12/12 | 2504891 | 5.26 | 1.08 GiB/19 s; pool est 1.74 | live | hit 62426808 (Loaded 189) |
| CG8K | – (–) | – | – | – | – | – | –/–/– | –/– | – | – | – | 8.21 GiB/59 s | – | – (Loaded –) |
| CG2K | – (–) | – | – | – | – | – | –/–/– | –/– | – | 1420753 | 2.98 | 2.46 GiB/16 s; pool est 4.02 | 1380s | – (Loaded –) |
| CTL2 | 185.5 (2.95) | 321.5 | 519.4 | 491.2 | 0.66 | 3.10 | 192/593/940 | 300/423 | 12/12 | 2504891 | 5.26 | 1.08 GiB/18 s; pool est 1.74 | 316s | hit 62426808 (Loaded 189) |

**Candidate arm:** CG2K  ·  **Verdict: FAIL** (PROVISIONAL — new autotune hash)
- WIN bar TTFT 8K <= 0.55 s: MISS — – s (CTL mean 0.65)
- WIN bar KV loss <= 0.3M: MISS — loss 1,084,138
- WIN bar RT C1 >= -2%: MISS — –%
- WIN bar replay r2 >= -2%: MISS — –%
- FAIL bar KV loss > 0.5M: HIT; decode −3% (C1/C4/C16 Δ = –, –, – %): clear
- other Δ vs CTL mean: C32 out –%, TTFT 8K –%, cold 64K –%, C16 TTFT –%, knee C8 –%
- CG8K: KV – (benches skipped or boot failed)
