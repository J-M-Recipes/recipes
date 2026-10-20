# Cross-lane prefill contention A/B — 2026-10-20 05:00–05:07 CDT
Question: does a concurrent 93K prefill on the sibling lane (other 4 Sparks, same CRS804-A switch) slow this lane's prefill?
Context: B0ctl (10-19 20:04–20:21) measured 93K prefill 3,304 tok/s while B0/B1a–B3 measured 5,003–6,207; the slow window overlapped the vLLM ladder on 5–8. Hypothesis: fabric contention.
Setup: B = SGLang B1a lane, spark1 :8888 (serving since 04:50). C = vLLM FINAL lane, spark5 :8002 (up 8 h). Same tool both sides (`prefill_reps.py`, cold 93,335-token prompts, unique nonce, max_tokens 1, T=0). Both lanes otherwise idle (0 running reqs). No hygiene step between phases.
Receipts: this dir (`<phase>-<lane>.log`, timestamped per request); on-box `spark1:~/bench-r2/contention/`, `spark5:~/bench/ladder/contention/`.

| phase | B 93K tok/s (3 reps) | C 93K tok/s (3 reps) |
|---|---|---|
| quiet (lanes alone, sequential) | 5,996 / 6,236 / 6,219 → **6,219** | 5,993 / 6,016 / 5,927 → **5,993** |
| contB: C prefilling back-to-back ×9, B measured inside | 6,254 / 6,242 / 6,210 → **6,242** | (background, 9 reps) 5,970–6,023 |
| contC: B prefilling back-to-back ×9, C measured inside | (background, 9 reps) 6,169–6,250 | 5,999 / 6,015 / 6,009 → **6,009** |

Verdict: **no contention effect.** Both lanes are within +0.4% of their quiet numbers while the sibling lane prefills continuously. Every per-request time is 14.9–15.1 s (B) and 15.5–15.7 s (C) regardless of what the other pod is doing. The switch is not the shared bottleneck for two TP4 pods at this traffic level.
Consequences:
- The B0ctl 3.3K window is NOT explained by cross-lane traffic. Remaining suspects: host state after the B0→B0ctl warm reboot (B0ctl was the only arm whose boot followed a wedged spark2 worker killed by name; compaction/THP state untested), or something transient on 1–4. It stays an unexplained outlier, reported as such — not a fabric finding.
- The Oct-18 B prefill 5,346 was NOT contended-low; B's quiet-fabric 93K number is 6.2K on the 1017 ckpt + 10-19 hygiene (B0 5,003 cold-JIT first boot, B1a–B3 6.1–6.2K).
- Same-day cross-lane benching on one switch is valid at this load (two single-stream 93K prefills). Not tested: both pods under concurrent multi-stream needles/decode at once.
- Post shrinks per the 10-20 04:20 plan: recipe update + corrections (sparknet: pinned 3bc8d10 declares 0.3.0 — the earlier '0.4.0' note was wrong, upstream 0.4.0 is not in the image; CED-skip +1–2% on TP4 note for Owen; first-touch 300K effect; NV_ERR_NO_MEMORY/indexer-budget mitigation). No standalone "contention" post.
