**Verdict: FAIL** — v25m fidelity gates (9/10 PASS). Reference: SC2 C arm (v25 static map). Status stays experimental until the 48 h soak completes.

| gate | result | detail |
|---|---|---|
| G1 TF flips vs v23 CTL1 and vs v25 static | PASS | vs CTL1: GATE PASS hi_agent_tool=0 hi_prose_pct=0.003 W=2.843 W_calib=3.078 skipped_docs=0 | vs v25: GATE PASS hi_agent_tool=0 hi_prose_pct=0.006 W=2.548 W_calib=2.895 skipped_docs=0 |
| G2 GSM8K-1319 >= v25 - 1 pt | PASS | v25m 0.9681576952236542 vs v25 0.9696739954510993 |
| G3 BFCL dev + heldout >= v25 - 1 pt | PASS | dev 0.9266666666666666 vs 0.9266666666666666 · heldout 0.7765064836003052 vs 0.7765064836003052 |
| G4 gauntlet 12/12 | PASS | 12/12 |
| G5 needle 500/500 exact | PASS | exact 500/500 errors 0 |
| G6 reasoning_effort codes == v25 | PASS | v25m {"none": 200, "low": 200, "high": 200, "xhigh": 200, "max": 200, "medium": 400} · v25 {"none": 200, "low": 200, "high": 200, "xhigh": 200, "max": 200, "medium": 400} |
| G7 image request | PASS | http=200 delta=185 image_tokens=None names_content=True answer='The image contains red on the left half and blue on the right half.' |
| G8 per-turn tool emission | PASS | v25m tool [8, 8] (bar >= 4), text [4, 5] (bar <= 5), errors 0 |
| G9 cosine gate (check boot) | FAIL | {'post_startup_checks': 1520, 'uncovered_layers_lt32rows': [], 'min_cos': 0.99811, 'max_rel_diff': 0.062, 'peer_timeouts': 0, 'hook_timeout_lines': 0} |
| G10 0 sidecar timeouts (V, K) | PASS | V shm delta 0 · K peer 0 hook 0 |

Reported (not gated):
-  RT V: c1-r1 agg=269.23 ttft=0.2 acc=2.869 ; c1-r2 agg=278.87 ttft=0.2 acc=2.976 ; c1-r3 agg=271.63 ttft=0.2 acc=2.897
- KNEE V r1: [SC2-V-r1] conc= 1: 217.1 agg tok/s 217.06 per-stream (runs 215.4/218.8);[SC2-V-r1] conc= 8: 892.5 agg tok/s 111.56 per-stream (runs 883.9/901.0);[SC2-V-r1] conc=16: 1429.9 agg tok/s 89.37 per-stream (runs 1404.9/1455.0);
- KNEE V r2: [SC2-V-r2] conc= 1: 219.8 agg tok/s 219.79 per-stream (runs 212.9/226.7);[SC2-V-r2] conc= 8: 881.3 agg tok/s 110.17 per-stream (runs 898.7/864.0);[SC2-V-r2] conc=16: 1392.8 agg tok/s 87.05 per-stream (runs 1409.8/1375.8);
- REPLAY V r2: [SC2-V-r2] REPLAY n=4: agg 513.8 tok/s over 12s, 5974 tokens; accept=56.5% acc/step=2.83; tool turns 24 emitted=14 per-req 141.6 tok/s; text turns 24 per-req 123.0 tok/s
- TIMEOUTS V: {"shm_start": [4963480, 4963480, 0, 389300, 1219281, 7566689664, 1687556], "shm_end": [6047120, 6047120, 0, 32467144, 64285102, 24200902112, 2722098], "shm_end_timeouts": 0, "shm_delta_timeouts": 0, "wait_ms_per_wait": 0.016078817919427148, "waits": 1034542, "peer_log_timeouts": 0, "hook_timeout_lines": 0, "watchdog_timeout_trips": 0, "watchdog_other_trips": 0}
- POWER V: GB300 n=759 p50=600 p99=915 max=961 swcap%=0.26 | RTX n=759 p50=164 p99=222 max=230 swcap%=0.00
- FACTS P: peer2=2 rowmap40=1 replay_line=1 NONDEFAULT  autotune_starts=25 | GPU KV cache size: 4,400,013 tokens, Maximum concurrency for 1,048,576 tokens pe | RTX 66213 MiB
- SOAK clock started (.*): 2026-10-24 06:14:34 CDT -> 2026-10-26 06:14:34 CDT
