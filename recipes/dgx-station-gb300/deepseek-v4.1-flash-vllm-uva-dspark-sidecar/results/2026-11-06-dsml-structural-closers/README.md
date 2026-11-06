# DSML closers inside tool arguments: structural parser overlay (vLLM), 2026-11-06

Deployed on this recipe's live lane on 2026-11-06. Live lane 2026-11-06: the file was added to the frozen `$WORK/overlay-vllm/parser/` and to `BUILD-SHA256SUMS` (the build dir is read-only by design: unlock the one dir + sums file, add, re-freeze), then `launch-pg61.sh down` / `up`; window 07:14:35-07:20:31 CST. A first attempt (07:00-07:13) booted stock and rolled back: its copy into the read-only dir failed silently. That was a deploy-script bug, not an overlay fault.

**The bug.** DeepSeek-V4.1 writes tool calls as DSML blocks: `<｜DSML｜ invoke name="…">`, then one
`<｜DSML｜ parameter name="…" string="true">value</｜DSML｜ parameter>` per argument, then `</｜DSML｜ invoke>` and
`</｜DSML｜ calls>`. The stock vLLM and SGLang V4.1 parsers end a value at the first closer they see. When the value
itself contains a closer (an agent writing docs, tests or code *about* DSML, or quoting a transcript), the argument is
cut there. On stock vLLM v0.31.0 a value holding `</｜DSML｜ calls>` or `</｜DSML｜ invoke>` loses the argument and
the text after it, and streaming emits unterminated JSON (`{"path":"dsml.md","content":"`). A value holding
`</｜DSML｜ parameter>` is silently truncated. Stock SGLang `f80c91a4b` does the same: on the live lane, a request
to write a file that names the three closers came back with the `content` argument **missing** (non-streaming) and
**cut at "…each argument ends with"** (streaming).

**The rule** (same idea as TensorFold patch 0052, written independently for these parsers):
`</｜DSML｜ parameter>` ends a value only when what follows is whitespace and then a new `<｜DSML｜ parameter name=`
head, or the block's own closer. `</｜DSML｜ invoke>` / `</｜DSML｜ calls>` end a call only once its arguments are
closed. Anything else is held, and becomes literal text when the next tokens show it was not structure (end of stream
resolves it as structure). A half-written tag at the end of a streamed chunk counts as structure, so speculative
decoding's multi-token chunks behave the same as one token at a time. Prompt rendering is untouched; nothing already
streamed is retracted.

**Intended behaviour changes vs stock.** The model omits `</｜DSML｜ parameter>` before `</｜DSML｜ invoke>`: the
argument is now kept (stock dropped it). A value that ends with a literal closer is kept whole. SGLang: two streamed
cases that stock sent as invalid JSON are now valid; a JSON body holding `</｜DSML｜ invoke>` now parses (stock `{}`).
**Unchanged, out of scope:** a call cut mid-value by `max_tokens` still streams unterminated JSON, stock and patched
alike (vLLM issues #62051 / #62259). On SGLang a call truncated just after a literal closer now returns no call where
stock returned a partial one (`O2`). A value that contains `</｜DSML｜ parameter>` followed by whitespace and a
`<｜DSML｜ parameter name=` head is still read as two arguments; nothing can tell that case apart from real structure.

## Files

| file | sha256 | what |
|---|---|---|
| [`patches/garner-8043a681/overlay-vllm/parser/deepseek_v41.py`](../../patches/garner-8043a681/overlay-vllm/parser/deepseek_v41.py) | `1c309d195e4346969f6b5b511e49f8e6d60e8d1023b1f93619f346cbcb4da3ba` | the overlay, mounted (or `docker cp`'d) over `/usr/local/lib/python3.12/dist-packages/vllm/parser/deepseek_v41.py` |
| [`dsml-structural.patch`](dsml-structural.patch) | | `diff -u` of the image's stock file → overlay, for audit |
| [`tests/`](tests/) | | offline matrix (`matrix.py`, `compare.py`, `run_matrix.sh`) and the live probe (`live_probe.py`) |
| [`raw/`](raw/) | | this lane's stock / patched / kill-switch matrix outputs, `VERDICT.txt`, live-probe results |

Kill switch: `VLLM_DSV41_STRUCTURAL_DSML=0` in the container env reproduces stock byte for byte (gate G5).
The vLLM overlay checks that the engine hooks it replaces exist and stays stock (logging an error) if they do not, so a vLLM bump cannot half-apply it.
An `</｜DSML｜ invoke>` or `</｜DSML｜ calls>` that is kept as text is logged at INFO (prefix `dsv41-structural-dsml-v1`); literal `</｜DSML｜ parameter>` decisions are not logged.

## How it is applied

`launch-pg61.sh build` copies `patches/garner-8043a681/overlay-vllm/` over Garner's overlay into `$WORK/overlay-vllm/`, and Garner's `serve.sh` bind-mounts every file there into the vLLM package, so the parser lands at `vllm/parser/deepseek_v41.py` with no launcher change. `BUILD-SHA256SUMS` (written by `build`, checked by the lane cron) carries its line. The launcher's sha1 gates still cover only the two #61161/#60945 files; the parser is gated by `BUILD-SHA256SUMS` and by the in-container sha256 check below.

## Gates

**Offline, before any lane was touched** (`tests/run_matrix.sh`, the lane's own image `vllm/vllm-openai:v0.31.0`, CPU only,
`--network none`, tokenizer from the checkpoint). 25 cases × {non-streaming, streamed 1-, 3-, 7-token chunks, whole
reply} plus 2 reasoning cases, run three ways (stock, patched, kill switch). `compare.py` gates: G1 every streamed call
is valid JSON, G2 stream equals non-stream, G3 the fixed cases come out exact, G4 every other case equals stock, G5
kill switch equals stock, G6 reasoning output equals stock, plus an instrument check that stock really fails the fixed
cases. Result: **ALL GATES PASS** ([`raw/VERDICT.txt`](raw/VERDICT.txt); the `CHANGE`, `PRE-EXISTING` and
`STOCK-ASYMMETRY` lines are the intended changes and the out-of-scope cases above). `run_matrix.sh` as published was
re-run on 2026-11-06 after the deploy and passed again.

**Live, after the deploy** (`tests/live_probe.py <head>:30006/v1`, temperature 0, thinking off): a control
tool call plus a `write_file` whose requested content names all three closers, non-streaming and streaming. Pass = one
call, valid JSON arguments, the closers kept inside the content, stream equal to non-stream. Result: **ALL_PASS**
([`raw/live-AFTER.jsonl`](raw/live-AFTER.jsonl)). In-container sha256 `1c309d19…`, activation log line ×2, sidecar watchdog ENFORCE with `trip: null`, lane cron silent. A Hermes agent one-shot through the lane made
its terminal tool calls normally.

## Rollback

Remove `patches/garner-8043a681/overlay-vllm/parser/` (or, on a built lane, `$WORK/overlay-vllm/parser/deepseek_v41.py` plus its `BUILD-SHA256SUMS` line), then `launch-pg61.sh down` / `up`. Faster, no file change: set `VLLM_DSV41_STRUCTURAL_DSML=0` in the container env (reproduces stock exactly, gate G5).

## Other lanes

The same file runs on the 4× Spark vLLM recipe (`dgx-spark-4x/deepseek-v4.1-flash-vllm-tp4-sparknet`), and an SGLang port runs on `dgx-spark-4x/deepseek-v4.1-flash-sglang-tp4-roce`. Not applied to the h2-REF rollback lane or to the single-GPU `deepseek-v4.1-flash-vllm-uva-dspark` recipe.

## Credits

Hector Isaza ([@hecisaza](https://github.com/hecisaza)) found and fixed this class of bug in TensorFold's DSML parser:
patch `0052` ("the block's own closer inside a DSML parameter value no longer ends the block"), [MiaAI-Lab
PR #49](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/pull/49), merged 2026-11-04, in [MiaAI-Lab's DSV4.1 4× Spark TensorFold recipe](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold) (read at
`8ffa67d53ef1`), linked to [issue #14](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/issues/14) by ecohash-co (DSML fragments leaking into content in
long agent sessions). This overlay is our own implementation for the vLLM and SGLang parsers; no TensorFold code is
used. The rules differ: 0052 keeps the longer reading when it yields more calls; this overlay decides each closer from what follows it, which also covers a single call whose value holds a closer. A second model (Claude Fable 5.1) reviewed the
design before it was written; the code was not re-reviewed.
