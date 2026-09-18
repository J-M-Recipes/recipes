# v17b-uncaptured (NOT v17 data)

These files are from container `dsv41-vllm-v17b-seqs24-cg-EXP` after `--cudagraph-capture-sizes 1 2 4 8 12 16 24` was passed as TOKEN counts (Milo brief error: written as sequence counts). At DSpark k=1 for >=5 seqs a step is 2x seqs tokens, so >12 seqs ran uncaptured.

Do not mix into the v17 table. Container kept as `dsv41-vllm-v17b-seqs24-cg-EXP-TOKENSIZES-UNCAPTURED-FAIL`.

Real finding kept: trimmed capture sizes cut graph memory 2.33+1.78 GiB -> 0.61+0.46 GiB and KV 1.79M -> 2.96M tokens.
