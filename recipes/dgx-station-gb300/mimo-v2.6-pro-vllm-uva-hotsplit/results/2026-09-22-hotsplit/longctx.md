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
