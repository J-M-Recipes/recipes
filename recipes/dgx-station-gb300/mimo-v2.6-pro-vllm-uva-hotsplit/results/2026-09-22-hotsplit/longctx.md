# Long context on the v23 serving lane (September 22, 2026, 9:17–11:40 PM CDT)

Asked for publicly ("Can you do long context pls?"). Client-only bench, no restart: `scripts/longctx_bench.py`
against `:30007` (v23, `--max-model-len 262144`, KV pool 302,368 tokens).

- Filler: real Python standard-library source, a random offset per prompt, sized with the server's `/tokenize` endpoint.
- Every prompt starts with a unique nonce, so the prefix cache cannot help.
- **Needle:** a random code (`NNN-WORD-NNNN`) is planted at 10%, 50% and 90% depth.
  The model is asked for it; pass = the exact code appears in the reply. Thinking is off, T=0, and max_tokens is 32.
- **Decode:** same filler, no needle, and a 256-token reply with ignore_eos. decode = (tokens−1)/(t_last − t_first).
  Prefill = prompt tokens / TTFT.

## Clean rows (lane otherwise idle)

| prompt tokens | needles 10/50/90% | TTFT | prefill | decode (256 tok) | run |
|---|---|---|---|---|---|
| 61.8K–65.3K | 3/3 | 46.7–48.7 s | 1,326–1,350 tok/s | 34.0 tok/s | v23 |
| 130.7K–130.9K | 3/3 | 99.8 s | 1,310–1,312 tok/s | 34.3 tok/s | v23 |
| 196.3K–196.9K | 3/3 | 152.7–154.4 s | 1,276–1,285 tok/s | 33.3 tok/s | v23b |
| 253.4K–253.7K | 3/3 | 200.9–201.3 s | 1,261–1,262 tok/s | 33.8 tok/s | v23b |

Needles 12/12 on clean rows. Reference at 11.5K on the same lane: decode 36.9–37.1 tok/s, prefill 1,376 tok/s.
At 254K, decode is ~9% lower than at 11.5K, and prefill is ~8% lower.

## Contended rows (kept, not used)

From 9:30 to 9:54 PM CDT, another client's requests shared the lane (vLLM log: `Running: 2` / `Waiting: 1`).
That overlapped the first run's 196K and 254K sizes. In those rows, 196K decode read 21.8 tok/s.
At 254K, the 90% needle's TTFT read 243 s and the decode row's TTFT read 306 s. All six needles still passed.
The two sizes were re-run as `v23b` on an idle lane. Both files are in `receipts/`: `longctx-v23.*` is the first run,
and `longctx-v23b.*` is the re-run. The 16K `longctx-smoke.jsonl` validated the script (3/3).

Total needle recall across all runs: 21/21 (smoke 3, v23 12, v23b 6).

## What this does not show

- A single planted fact is the easiest long-context test. It says the lane attends across 254K tokens.
  It does not measure multi-hop reasoning or summarization quality at that length (RULER-style suites: pending).
- One request at a time. The 302K KV pool fits one 254K request; two long contexts at once need the stock-equal
  budget variant (514K KV).

## v24: 1M-context variant (September 23, 2026, 12:00–12:50 AM CDT)

Same image, patches, ranking file and flags as v23, except for two settings: `--max-model-len 1048576` and
`HOTSPLIT_HOT_GIB=110`. The checkpoint's `max_position_embeddings` is 1,048,576. Only 10 of the 70 layers are
full attention; the other 60 are 128-token sliding window, so KV costs about 55 KB per token. A 1M pool therefore
needs about 43 GiB of HBM taken back from hot experts: 152.8 → 110 GiB, 8,692 → 6,257 hot cells, train-weighted
coverage 51.1%. Boot log: KV pool **1,209,097 tokens** (1.15× one 1M request); host avail after the split 79.0 GiB
(v23: ~123). `receipts/bootlog-v24-1m.txt`.

### Short-context cost (warm, same benches as v23)

| | v23 (262K, 152.8 GiB hot) | v24 (1M, 110 GiB hot) | change |
|---|---|---|---|
| decode after TTFT, 3K / 11.5K / 46K prompt | 37.5 / 37.4 / 36.9 | 33.2 / 33.0 / 32.7 | −12% |
| tool_json / shell / structured | 36.6 / 36.4 / 35.9 | 32.7 / 32.5 / 31.7 | −11% |
| code / prose | 32.6 / 32.7 | 29.5 / 29.0 | −10% / −11% |
| prefill, 11.5K prompt | 1,376 tok/s | 1,235 | −10% |

v22 numbers are without the live counter; v23 with it read 36.9–37.1. v24 runs with the counter.
Receipts: `ttft-v24-1m.json`, `agentfix-v24-1m.txt`, `agentfix-v24-1m-r2.json`.

The published estimate before this boot was −7% (~34.9 tok/s). The estimate was linear in the fraction of traffic
served from HBM, and the measurement came in lower; the residual is not explained.

### Long context on v24

| prompt tokens | needles 10/50/90% | TTFT | prefill | decode (256 tok) |
|---|---|---|---|---|
| 65.0K–65.3K | 3/3 | 53.6–53.9 s | 1,207–1,216 tok/s | 32.0 tok/s |
| 523.5K–524.4K | 3/3 | 489–496 s | 1,057–1,070 tok/s | 28.6 tok/s |
| ~1.04M | **not measured** | | | |

At 500K, decode is 13% below v24's 11.5K number, and the wait before the first token is about 8 minutes 10 seconds.
The 1M run was stopped during the first 1.04M prefill: it was about 17 minutes of full GPU per prompt, for a number
nobody needed tonight. The lane accepts 1,048,576-token requests (served `max_model_len`), but recall and speed at 1M
are unmeasured. Receipts: `longctx-v24.jsonl`, `longctx-v24.log`.
