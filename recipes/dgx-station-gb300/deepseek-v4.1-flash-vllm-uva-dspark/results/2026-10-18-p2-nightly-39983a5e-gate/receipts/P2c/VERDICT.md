# Card P2c VERDICT (pre-registered bar; CTL = mean(A, D))

| arm | boot s | KV tok | gate | live tune? (starts / store) | RT C1 (runs) | TTFT 8K | RT acc_len | prose | structured | code | shell_ops | tool_json |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | live | 2,504,891 | n/a 0909 | 1 / 210 (—) | 186.7 (191.7/187.7/180.8) | 0.650 | 2.963 | 24.7 | 46.8 | 78.4 | 77.0 | 83.1 |
| C | 334 | 3,878,024 | True | 1 / 210 (210/210) | 200.4 (200.4/199.5/201.3) | 0.430 | 2.996 | 24.9 | 45.1 | 73.6 | 77.4 | 88.0 |
| C1 | 557 | 3,572,679 | False | 1 / 210 (210/210) | 193.0 (199.6/192.7/186.8) | 0.430 | 2.956 | 24.9 | 43.0 | 73.6 | 74.1 | 84.6 |
| C2 | 546 | 3,874,039 | True | 1 / 210 (210/210) | 203.7 (210.5/201.1/199.6) | 0.427 | 2.982 | 24.7 | 46.2 | 74.2 | 73.4 | 87.1 |
| D | 305 | 2,504,891 | n/a 0909 | 1 / 210 (—) | 188.6 (190.7/190.6/184.5) | 0.650 | 2.979 | 24.7 | 46.8 | 78.4 | 77.0 | 86.1 |
| CTL | | | | | 187.7 | | | 24.7 | 46.8 | 78.4 | 77.0 | 84.6 |

Δ vs CTL (pts / %):

| arm | code Δ | structured Δ | tool_json Δ | prose Δ | shell Δ | RT C1 Δ% | TTFT 8K | ident-output code (n, arm vs CTL) | ident-output structured | bar |
|---|---|---|---|---|---|---|---|---|---|---|
| C | -4.7 | -1.7 | 3.4 | 0.2 | 0.3 | 6.8 | 0.430 | n=4: 79.7 vs 83.2 | n=2: 68.6 vs 71.8 | FAIL (code x, struct ok, RT ok, TTFT ok) |
| C1 | -4.8 | -3.8 | -0.0 | 0.3 | -2.9 | 2.9 | 0.430 | n=4: 82.9 vs 86.7 | n=1: 91.5 vs 91.5 | FAIL (code x, struct x, RT ok, TTFT ok) |
| C2 | -4.2 | -0.6 | 2.5 | 0.0 | -3.6 | 8.6 | 0.427 | n=4: 79.7 vs 83.2 | n=2: 69.6 vs 71.8 | FAIL (code x, struct ok, RT ok, TTFT ok) |

Mechanism: A: replay_line=False rehome=True drafter_inherits_target_backend=False gate=None errcount=0; C: replay_line=False rehome=True drafter_inherits_target_backend=True gate=True errcount=0; C1: replay_line=False rehome=True drafter_inherits_target_backend=True gate=False errcount=0; C2: replay_line=False rehome=True drafter_inherits_target_backend=False gate=True errcount=0; D: replay_line=False rehome=True drafter_inherits_target_backend=False gate=None errcount=0

Rotation fixture (agent_fixture_o.sh ×3, mean / spread, %): A: prose 25.2±0.0, structured 48.9±0.0, code 69.6±0.0, shell_ops 80.0±0.0, tool_json 73.3±0.0 ; C: prose 30.6±0.0, structured 42.7±0.0, code 62.7±0.0, shell_ops 80.0±0.0, tool_json 79.9±0.4 ; C1: prose 31.6±0.0, structured 35.5±0.0, code 62.1±0.0, shell_ops 73.8±0.0, tool_json 79.7±5.3 ; C2: prose 28.8±0.0, structured 42.7±0.0, code 63.7±0.0, shell_ops 80.0±0.0, tool_json 77.9±2.6 ; D: prose 25.2±0.0, structured 48.9±0.0, code 69.6±0.0, shell_ops 80.0±0.0, tool_json 73.3±0.0

**Verdict: no override arm recovers** (C1 #56893 gate False, C2 drafter FLASHMLA_SPARSE_DSV41).
SUFFIX_C1 FAIL
SUFFIX_C2 FAIL
