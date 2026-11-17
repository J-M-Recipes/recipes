# overlays — what this recipe mounts over the image, and what it only keeps for the record

The r7 lane mounts **two** files read-only over `spark-ds41f:candidate-60`. Neither needs a rebuild, both are
checksummed in `SHA256SUMS` (54 files: 4 late-admission, 48 sparknet, 1 dsml-structural, 1 so-bitmask, 1 dsml-c60),
and both are mounted on all four ranks or not at all.

The other three directories are **r6x-only** and are kept for rollback and provenance. Their mount targets are
`/opt/spark3/candidate/vllm/...`, a path that does **not exist** in a candidate-60 image (the c60 tree root is
`/opt/spark-ds41f/candidate/vllm`, see `docker/Dockerfile:259` and `ENV PYTHONPATH=` at `:329`). Docker creates a
*directory* at a missing mount target, so mounting any of them on c60 fails **silently**: the lane runs stock c60 and
nobody sees it in the logs. Two of the three would also revert c60 code if they did land — see the notes on each.

## Mounted on r7

### `so-bitmask/vllm/v1/structured_output/__init__.py` — the spark-ds41f#1194 guard

```
host: overlays/so-bitmask/vllm/v1/structured_output/__init__.py
  ->  /opt/spark-ds41f/candidate/vllm/vllm/v1/structured_output/__init__.py:ro
sha256 d286743334c81755c3ee3bc8d33eb3bf85760a9833085289154474ca6fbe165e
```

candidate-60's own file plus four added lines (three of code, one comment) — **Zee-B33's fix for
[spark-ds41f#1194](https://github.com/christopherowen/spark-ds41f/issues/1194), applied verbatim**:

```python
# spark-ds41f#1194: copy drafts verify up to num_verification_tokens rows per request.
_spec = self.vllm_config.speculative_config
if _spec is not None and getattr(_spec, "num_verification_tokens", None):
    max_num_spec_tokens = max(max_num_spec_tokens, _spec.num_verification_tokens)
```

`grammar_bitmask` sized its bitmask from `num_speculative_tokens` (5) only. c60 lets a copy draft verify more rows
than that — the boot log prints `Copy drafts widen the CUDA-graph capture range from 192 to 203 rows` — so a
structured-output request deep enough to be served by copy drafts could index past the 6-rows-per-request bitmask.
The fix widens it to `max(num_spec_tokens, num_verification_tokens)`.

The reproductions in #1194 are from **mjcryptocloud** (4-node, 4 KiB pages) and **hxfhd** (TP3). **We did not
reproduce the crash ourselves**, on r6x or on stock c60; the overlay is a guard, and `scripts/so_probe.py` is the
probe that exercises the shape (32 concurrent `response_format` requests whose answers copy 40 records out of the
prompt, two waves; 64/64 in every arm including the r6x control). It costs nothing when the condition never fires:
`num_verification_tokens` is absent on the non-copy path, and the bitmask then sizes exactly as before.

### `dsml-c60/vllm/parser/deepseek_v41.py` — structural DSML closers, re-ported onto c60

```
host: overlays/dsml-c60/vllm/parser/deepseek_v41.py
  ->  /opt/spark-ds41f/candidate/vllm/vllm/parser/deepseek_v41.py:ro
sha256 8a8b293708eb15b763ed6fd4c6f52e85bff830e254f3029b69a194951d0c35c6   (stock c60 file: 844e4700…)
```

DSML has no escaping, so a tool argument that *contains* a DSML closer (a file about DSML, a note quoting the tags)
used to be cut at that closer: the argument was dropped or truncated and the streaming path emitted unterminated JSON.
The rule: a parameter closer ends a value only when whitespace + a new parameter head, or the end of the call's
arguments, follows; an invoke/calls closer ends a call only when the arguments are closed, and is otherwise held as
text (it still becomes structure if a calls closer, a new invoke head or the end of the stream comes next).

candidate-60 already added a **token-provenance** rule (PR #949): markup spelled in ordinary pieces is text. That
does **not** cover our checkpoint — our live-probe token ids show the quoted closer arriving as the genuine
`｜DSML｜` token (128825), so provenance alone still cuts the value. This overlay keeps c60's provenance path and adds
the lookahead on top of it.

- **Not a mount of the r6x overlay.** `overlays/dsml-structural/` is a whole-file replacement of the r6x parser and
  defines only `deepseek_v41_config` + `DeepSeekV41Parser`. c60's `vllm/parser/parser_manager.py:153` imports
  `DeepSeekV41ServingParser` from that module on every `deepseek_v41` request, so mounting the old file raises
  `ImportError` at request time. This one is a re-port (R4, our own code) onto c60's file.
- **Verified in the image**, CPU only, `--network none`, 25 cases × non-stream + 4 chunkings + 2 reasoning:
  `ALL GATES PASS` — stock fails exactly B/C/D/E/E2/J (6/25), the overlay fixes all six, and it changes only
  N, N2, O2, Q, R. Receipts: `results/2026-11-16-r7-candidate60/image-gate-run.log`, kit in
  `results/2026-11-16-r7-candidate60/tests/`.
- **Live probe** (`tests/live_probe.py`) passes with the overlay *and* on stock c60. The live case is a single quoted
  `</｜DSML｜ parameter>` inside a value — stock c60 keeps it; the overlay additionally covers the **6 synthetic
  genuine-atom cases** the live probe does not construct. Do not read the live probe as evidence the overlay is
  needed; read the image gate as evidence it does something.

Kill switch: `VLLM_DSV41_STRUCTURAL_DSML=0` reproduces stock c60 byte for byte (gate G5). The module also refuses to
half-apply: if c60's private hooks change shape it logs `structural DSML overlay disabled (stock behaviour)` and stays
stock. Startup logs one line per process: `dsv41-structural-dsml-v1: structural DSML closers active`.

## Kept, but r6x-only — do not mount these on c60

| directory | what it was for | why it is r6x-only |
|---|---|---|
| `late-admission/vllm/` (4 files) | Owen's 0222 patch, late admission at layer 20 | **In-tree in c60** (`vllm/v1/worker/gpu/late_admission.py`, patches 0222 + 0226). c60 keeps it as `VLLM_LATE_ADMISSION_LAYER` (default `-1` = off); the lane sets `20`. Mounting these copies would revert c60's `envs.py`, `v1/engine/core.py` and `model_runner.py`. |
| `sparknet-0.5.0/sparknet/` (48 files) | sparknet 0.5.0 over the image's vendored 0.3.0 | **c60 ships a newer 0.5.0** (`1fd6a1fb`, tree `e06f6a06`) that includes the single-slot transfer cap (large prefill chunks stay on PyNCCL). Tag `ed6af327` predates that patch, so mounting it would revert the fix. |
| `dsml-structural/vllm/parser/deepseek_v41.py` | the 2026-11-06 structural-closer overlay | Written against the r6x parser (no `DeepSeekV41ServingParser`); `ImportError` on c60. Superseded by `dsml-c60/`. |

Their provenance is unchanged from r6x: Owen's `0222` late-admission drop (reproduced by `git am` on tree head
`7316c7f9f` → his commit `91eaaa59e1ba`) and `dgx-spark-networking` v0.5.0 tag `ed6af327`. Both are r6x files: on
c60 the equivalent code is in the image, and `sparknet` in the c60 image is newer than `0.5.0` tag `ed6af327`.

## Applying them by hand

Bind mounts are what the launcher does (all four ranks, `:ro`). The alternative that survives `docker stop/start`
but not a recreate is `docker cp` into the container on ranks 3, 2, 1, 0 followed by the ordered stop/start. Either
way, check the file's sha256 inside the container before reading a benchmark — a mount that landed on a directory
looks healthy from `docker ps`.
