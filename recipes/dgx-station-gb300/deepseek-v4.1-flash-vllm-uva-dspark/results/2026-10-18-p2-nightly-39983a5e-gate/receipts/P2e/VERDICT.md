# Card P2e VERDICT (pre-registered bars; CTL = mean(A, D))

| arm | boot s | KV tok | KV fmt | block-size backends | live tune? (starts / window / saved / store / preloaded) | guard host_avail L9 | RT C1 (runs) | TTFT 8K | RT acc_len | prose | structured | code | shell_ops | tool_json |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | live | 2,504,891 | fp8_ds_mla | FLASHMLA_SPARSE_DSV41 | 1 / 18:43:08→18:43:13 / 189 / 210 / — (key —) | 196.82 | 186.8 (190.9/187.9/181.6) | 0.650 | 2.948 | 24.7 | 46.8 | 78.4 | 77.0 | 83.1 |
| E | 526 | 3,877,002 | fp8_ds_mla | FLASHMLA_SPARSE_DSV41/DEEPSEEK_SPARSE_SWA/DEEPSEEK_V41_INDEXER/CompressorBackend | 1 / 22:16:33→22:16:41 / — / 210 / 210/210 (key fd6e12df2b959586) | 197.22 | 202.8 (202.2/207.7/198.5) | 0.460 | 2.953 | 26.2 | 44.3 | 74.7 | 74.7 | 81.5 |
| D | 305 | 2,504,891 | fp8_ds_mla | FLASHMLA_SPARSE_DSV41 | 1 / 22:26:05→22:26:10 / 189 / 210 / — (key —) | 197.55 | 188.6 (187.7/190.8/187.2) | 0.640 | 2.931 | 24.7 | 46.8 | 78.4 | 77.0 | 86.1 |
| CTL | | | | | | | 187.7 | 0.645 | | 24.7 | 46.8 | 78.4 | 77.0 | 84.6 |

| arm | knee r1 C1/C8/C16 | knee r2 C1/C8/C16 | replay r1 (acc) | replay r2 (acc) | cold 64K tok/s (ttft s) | gauntlet | errcount |
|---|---|---|---|---|---|---|---|
| A | 192.0/620.1/928.8 | 192.2/597.4/940.7 | 309.3 (0.599) | 426.6 (0.595) | 23847 (2.24, 2.22) | 12/12 | 0 |
| E | 219.9/634.1/1,069.1 | 220.3/636.4/1,082.1 | 215.5 (0.567) | 470.5 (0.613) | 27469 (1.94, 1.93) | 12/12 | 0 |
| D | 192.4/593.9/943.9 | 192.5/590.2/943.0 | 294.5 (0.592) | 423.4 (0.614) | 24109 (2.23, 2.18) | 12/12 | 0 |

Δ E vs CTL:

| code Δ | structured Δ | prose Δ | shell Δ | tool_json Δ | RT C1 Δ% | replay r2 Δ% | TTFT 8K (vs CTL) | gauntlet | ident-output code | ident-output structured |
|---|---|---|---|---|---|---|---|---|---|---|
| -3.7 | -2.5 | 1.5 | -2.4 | -3.1 | 8.1 | 10.7 | 0.460 (0.645) | 12/12 | n=4: 85.1 vs CTL 86.7 | n=2: 85.7 vs CTL 86.0 |

Bars: RECOVERS = code x, structured x (valid=True, measured=True); candidate-quality adds RT ok, replay r2 ok, TTFT x, gauntlet ok.

Mechanism: A: replay_line=False rehome=True mega_in_facts=False kvfmt=['fp8_ds_mla'] drafter_inherit_line=False; E: replay_line=False rehome=True mega_in_facts=False kvfmt=['fp8_ds_mla'] drafter_inherit_line=False; D: replay_line=False rehome=True mega_in_facts=False kvfmt=['fp8_ds_mla'] drafter_inherit_line=False

Rotation fixture (agent_fixture_o.sh ×3, mean ± spread, %): A: prose 25.2±0.0, structured 48.9±0.0, code 69.6±0.0, shell_ops 80.0±0.0, tool_json 73.3±0.0 ; E: prose 31.8±0.0, structured 36.8±0.0, code 63.1±0.0, shell_ops 85.7±0.0, tool_json 78.4±4.8 ; D: prose 25.2±0.0, structured 48.9±0.0, code 69.6±0.0, shell_ops 80.0±0.0, tool_json 73.3±0.0

**Verdict: E does NOT recover — target MegaAttn is not the cause; config C dead for the agent lane under the 2-pt rule.**
SUFFIX_E FAIL
