# GB300 port: DSML structure only from the real `｜DSML｜` token (vLLM PR #65019), merged with our structural-closers overlay

**Status: draft delivered and proven on CPU off-box; the in-image run is still to be done
(this desk session may not start docker).** 67 CPU-only tests pass locally against the real
ported parser code; the same suite fails 45/67 on stock v0.31.0, which is the point.

| item | value |
|---|---|
| upstream | `vllm-project/vllm` PR **#65019**, head `cad5796ae967dc3a25a7314c1ecfcca22ae4dd43` (OPEN) |
| PR diff saved | `pr/pr65019.diff` (`gh pr diff 65019 -R vllm-project/vllm`), per-file copies in `pr/split/` |
| target tree | stock vLLM **v0.31.0**, `research/dsv41-owen-gb300-20261116/src/vllm-v0.31.0/vllm` |
| our overlay merged in | `src/pg62/overlay-vllm/parser/deepseek_v41.py`, 317 lines, sha256 `1c309d19…` (production) |
| prior merge for reference | `work/spark-recipe-r7-2026-11-16/overlays/dsml-c60/vllm/parser/deepseek_v41.py` (Owen's tree, 708 lines) |
| overlay files to mount | **7** (`overlay/…*.DRAFT`, see `MOUNTS.txt`) |
| new env switches | `VLLM_DSV41_DSML_TOKEN_PROVENANCE` (new), `VLLM_DSV41_KEEP_OPEN_ARGS` (new), `VLLM_DSV41_STRUCTURAL_DSML` (kept) |
| test command | `RUN-TESTS.sh.DRAFT` (one `docker run … -m pytest /port/tests -q`) |

## 1. What the PR changes

The V4.1 tokenizer spells every DSML marker as ordinary text around the single token
`｜DSML｜` (id 128825), so the model can produce byte-identical markup with ordinary
tokens — in a code fence, in a quoted example, in reasoning. Stock v0.31.0 matches
markers as **text**, so a spelled marker is structure; with a tool offered that becomes a
real call, and with no tool offered the content is cut at the fence.

The PR makes structure depend on token provenance:

1. **`DSML_ATOM` becomes a token terminal.** `deepseek_v41_config` adds
   `token_id_terminals["DSML_ATOM"] = "｜DSML｜"` and
   `composite_token_id_terminals = {<each of the six DSML terminals>: "DSML_ATOM"}`
   (`parser_engine_config.py` gains that field).
2. **The engine treats the atom token as text with a trusted span, not as a terminal.**
   `StreamingParserEngine._process_scanner_items` feeds a `PreLexedTerminal` whose
   terminal is a composite atom back into the lexer with
   `token_id_spans=((0, len(text)),)` instead of dispatching it.
3. **Spans are tracked through the lexer.** `LexToken.token_id_spans`,
   `IncrementalLexer(token_id_guards=…)`, `incremental_lexer.require_token_ids`, and
   `_accept_match()` — a literal such as `<｜DSML｜ parameter` is accepted only when the
   atom inside its match sits in a trusted span. `_make_token`/`feed`/`flush` carry the
   spans and rebase them.
4. **A strict scanner mode produces the spans.** `TokenIDScanner(strict_token_spans=True)`
   adds `_scan_token_spans()` (holds ids until their text arrives, byte-level-aware
   `_byte_prefix()`, linear decode work), `has_pending`, and a span-mode `flush_pending()`.
5. **Spans reach the argument converter.** `SemanticEvent` gains a `_TokenOriginEvent`
   subclass with `token_id_spans`; `_on_terminal`/`_emit_for_state`/`_on_content` take and
   forward them; `ToolCallSlot.append_args(value, token_id_spans)` accumulates absolute
   offsets (`arg_token_id_spans`); `_handle_arg_chunk` passes `event.token_id_spans`.
6. **Arguments parse from provenance boundaries.** `deepseek_v41._token_origin_boundaries`
   collects only parameter heads/closers whose atom is trusted; `_token_origin_args` builds
   the JSON from those boundaries only, and `DeepSeekV41Parser._convert_args` uses it when
   `_token_origin` (engine saw token ids **and** the atom is in the vocab). A slot with no
   recorded spans falls back to the text path.
7. **One engine for reasoning + tools.** New `DeepSeekV41ServingParser(DelegatingParser)`
   registered from `parser_manager.get_parser` when both parsers are `deepseek_v41` and
   auto-tools is on. Without it, v0.31.0's `DelegatingParser.parse` hands the tool parser
   *text only*, so non-streaming would never see token ids.
8. **Retention/state fixes that fall out of the same work:** `TOOL_BETWEEN` emits
   `TEXT_CHUNK` under composite ids (text between/after calls is content instead of being
   dropped), `_events_to_delta` defers text so it cannot overtake earlier deferred content,
   `turn_boundary_tokens = {<｜Assistant｜>, <｜User｜>, <｜System｜>}` for prompt reasoning
   state, and no-tools/`tool_choice="none"` requests keep DSML markup as content.

## 2. How it was adapted to v0.31.0 (conflicts, API drift)

Four of the six engine files applied **cleanly** with `patch` (context identical to
v0.31.0): `token_id_scanner.py`, `incremental_lexer.py`, `streaming_parser_engine.py`,
`parser_manager.py`. Two hunks were rejected and were written by hand:

* `ToolCallSlot.__slots__/__init__` — the PR context also lists `extend_open_value` and
  `open_value_streams`; v0.31.0's `ToolCallSlot` has neither. Added only
  `arg_token_id_spans` and `_args_length`.
* `ParserEngineConfig` — the PR inserts the new field before `__post_init__`; v0.31.0 has
  no `__post_init__` (and no `arg_value_extender`). Added the field after
  `validate_tool_names`.

**Dropped, with reason (API drift from upstream's newer tree):**

| PR piece | v0.31.0 situation | consequence |
|---|---|---|
| `deepseek_v4._dsml_open_value_extender` split | v0.31.0 has no `_dsml_arg_value_extender`/`arg_value_extender` at all | `deepseek_v4.py` is **not** overlaid; the file is unchanged |
| `DeepSeekV41Parser._extend_open_value` | depends on the extender hook above | not ported; v0.31.0 re-runs `_convert_args` on the accumulated raw args every tick (`_compute_arg_delta` + `_safe_arg_prefix`), which covers the same growth case only in part |
| `set_prompt_token_ids()` / `adjust_initial_state_from_prompt()` override | `ParserEngine` in v0.31.0 has no `set_prompt_token_ids`; its `adjust_initial_state_from_prompt` is a documented no-op and `DelegatingParser` already marks reasoning ended for a closed prompt | not ported. **Prompt-side reasoning accounting and the PR's two prompt tests are not covered** |
| `docs/features/tool_calling.md` | docs only | not ported |
| PR tests `test_deepseekv41_tool_parser.py` | need the real HF tokenizer (`get_tokenizer(MODEL)`) + `tokenizers`/`transformers` in the image | replaced by a purpose-built fake tokenizer (identical text, distinct ids) — see §4 |

## 3. How it merged with our closer overlay

The two rules are independent and both are needed; they compose in one place.

* **Boundary filter.** `_structural_token_origin_boundaries()` takes the PR's
  provenance boundaries and keeps a closer only when our `_closer_is_structural()` says
  structure follows it (whitespace + a new `<parameter name=`, or end of the arguments).
  A **genuine** closer that is not followed by structure therefore stays literal text —
  the spark-ds41f#948 case — while a spelled closer never becomes a boundary at all.
* **Streaming hold.** `_V41StructuralEngine` (our 317-line overlay's engine class) is kept
  as-is and updated to the PR's signatures: `_on_terminal(_terminal, value, token_count,
  token_id_spans)` and `_emit_for_state(text, token_count, token_id_spans)` forward spans.
  A closer released as literal text is emitted **without** spans on purpose, so the
  token-origin converter cannot read it as a boundary; the synthetic
  `</｜DSML｜ parameter>` the engine adds for a value the model left open is emitted
  **with** the atom's span (`_PARAM_CLOSE_ATOM`) so the converter reads it as structure.
* **Text-path converter.** `_structural_dsml_arg_converter` is our existing rule,
  unchanged, and is still the `arg_converter` (used whenever provenance is unavailable or
  off).
* **New in the merge:** text seen in `TOOL_PREAMBLE` (after a genuine `<calls>` opener and
  before any `<invoke>`) is buffered and, if the stream ends without a call, emitted as
  content. Stock drops it silently (documented in §4 case f). Cleared as soon as a call
  starts.
* **Kill switches** (env-only, no rebuild):
  * `VLLM_DSV41_STRUCTURAL_DSML=0` — rule 2 off; pure upstream-PR behaviour.
  * `VLLM_DSV41_DSML_TOKEN_PROVENANCE=0` — rule 1 off, turn boundaries off, unified
    serving parser not registered (stock composition); our previous parse, plus the
    case-(f) preamble fix.
  * `VLLM_DSV41_KEEP_OPEN_ARGS=0` — an unterminated final parameter is dropped as stock
    and upstream do. Default keeps it (our "closed, not dropped" spirit).

Startup logging: the parser logs once whether provenance is **active** (`atom id N,
special=…`) or **INERT** (`tokenizer has no '｜DSML｜' token → DSML matching is
text-based`). That single line decides whether the port is doing anything on the box.

## 4. Behaviour table

Measured, not asserted: every row below is real output from the local run
(`build/local`, real ported parser files + stubs only for serving-layer leaves):
command in §7. "old" = stock v0.31.0 + our current 317-line overlay.

| # | input | old behaviour (measured on the fixtures) | this port | test |
|---|---|---|---|---|
| a | #876 fence sample, spelled markup, **tools offered** | `CALL_BAD_SHAPE`: a real call `calc{value:42}`, content cut at the fence | **no call**; content = the whole sample incl. the closing fence and the last line | `test_a_*` (4 schedules) |
| b | same, **no tools** | content cut at the fence | **no call**, whole text | `test_b_*`; `tool_choice="none"` variant |
| c | genuine-token call | call parsed | call parsed, streaming + non-streaming, 1- and 2-parameter | `test_c_*` |
| d | genuine call, string arg containing **spelled** `</｜DSML｜ parameter>`, `</｜DSML｜ invoke>`, `</｜DSML｜ calls>` | argument truncated / call broken | `{"text": "a</｜DSML｜ parameter>b</｜DSML｜ invoke>c</｜DSML｜ calls>d"}` | `test_d_spelled_closers_*` |
| d2 | genuine call, string arg containing a **genuine** closer followed by text (our #948 rule) | argument intact (our rule) | intact: `{"text":"a</｜DSML｜ parameter>b"}` | `test_d2_*` |
| d3 | genuine closer followed by a real next head | next parameter parsed | same — the closer rule does not eat real structure | `test_d3_*` |
| e | reasoning quotes a spelled full call, then a genuine call | reasoning text parsed into a bogus call named `...` | reasoning keeps the quote verbatim; **exactly one** call `calc{42}` | `test_e_*`, `test_e2_*` |
| f | genuine `<calls>` opener, no `<invoke>`, spelled parameter | **all text silently dropped** (`TOOL_PREAMBLE` has no content event) | no call; the body is returned as content in one chunk (streaming too) | `test_f_*` |
| g | PR shapes: text call is content / stays reasoning / bare genuine atom / twins around a call / text between calls / emoji then twin / delayed BOS / partial marker | mixed failures (see the stock row below) | all pass on all four schedules | `test_g1_*`…`test_g8_*` |
| h | `VLLM_DSV41_DSML_TOKEN_PROVENANCE=0` | — | old bug returns: `[("calc", {"value": 42})]` again | `test_h_*` (subprocess) |

Kill-switch matrix (case a / c / d2 / f), measured:

| switches | (a) fence | (c) real call | (d2) genuine closer in arg | (f) opener w/o invoke |
|---|---|---|---|---|
| `STRUCTURAL=1, PROVENANCE=1` (delivered) | no call, text kept | parsed | intact | text kept |
| `STRUCTURAL=0, PROVENANCE=1` | no call, text kept | parsed | **truncated to `"a"`** | **text dropped** |
| `STRUCTURAL=1, PROVENANCE=0` | **old bug (call + cut)** | parsed | intact | text kept |

So provenance alone fixes (a)/(b)/(e) but not our closer rule, and our closer rule alone
fixes nothing about spelled markup: both are required.

Discrimination check (same suite, no overlays): stock v0.31.0 passes **22**, fails **45**.
The 22 passes are exactly the shapes stock already handled — genuine calls (c), (d3),
`g3`, `g7`, `g8` and the rollback test (h) — so the suite is not just "green because the
fixtures are weak".

## 5. Risks

1. **Coverage is not the lane.** These tests use a fake tokenizer with the lane's ids and
   a context-free decode. They prove the parser logic; they do not prove the lane's
   tokenizer spells markers the way we assume. The `_has_composite_ids` log line is the
   decisive on-box check (item 2 in §6).
2. **The engine files are global.** `streaming_parser_engine.py`, `incremental_lexer.py`,
   `token_id_scanner.py`, `parser_engine.py` and `parser_engine_config.py` are shared by
   every parser config in the image. The changes are gated (`strict_token_spans` only when
   composite ids resolve; `defer_text`/`TOOL_BETWEEN` only when `_has_composite_ids`), but
   one predicate is **not** gated: the `feed()` fast path now tests `_scanner.has_pending`
   instead of `_scanner._deferred_terminals`, which also counts deferred post-text. That
   can only cost a little speed for non-v41 parsers, but it is a global change.
3. **`turn_boundary_tokens`.** New for v41 (`<｜Assistant｜>`, `<｜User｜>`, `<｜System｜>`).
   If our chat template spells those with different pieces the set resolves to nothing and
   behaviour is as before; if it does resolve, prompt reasoning-state detection changes for
   multi-turn prompts. **Unverified against our template** — item 3 in §6.
4. **Streaming deltas.** Argument JSON is emitted from `_safe_arg_prefix`; when a value
   *grows* past a closer that was previously read as structural, the prefix check
   (`safe_json.startswith(prev)`) drops that tick's delta rather than emitting a wrong
   one, so a client can, in a pathological case, receive a shorter `arguments` prefix than
   the final non-streaming call. The `lag3` schedule covers the ordinary case; an in-image
   test with the real detokenizer is still worth having.
5. **Reasoning-parser interplay.** In auto-tool mode `DeepSeekV41ServingParser.parse_delta`
   bypasses `DelegatingParser.parse_delta`'s phase machine and drives one engine instead.
   That is what gives provenance across the reasoning→tool handoff, but it also means
   `tool_choice="required"`/named choices fall back to the stock JSON path
   (`_uses_json_tool_choice`), and the PR's own note applies: a *forced* call whose
   structural tags are spelled in text stays content. Not reproduced here.
6. **Performance per token.** With composite ids the plain-content fast path is skipped and
   every delta goes through the strict scanner (`_scan_token_spans`, which decodes runs and
   matches prefixes). The PR's own tests assert linear decode work; we did not port them
   (they need a byte-level tokenizer). Expect a small per-token cost on the lane; measure in
   §6 item 6.
7. **Behaviour changes beyond the bug fix**, all intended but worth knowing: text between
   and after tool calls is now content instead of dropped (`TOOL_BETWEEN` → `TEXT_CHUNK`);
   no-tools requests keep markup as content; an unterminated trailing string argument is
   kept rather than dropped (`KEEP_OPEN_ARGS`); text after a bare `<calls>` opener is kept.
8. **Non-streaming without a reasoning parser still has no provenance.** The PR only wires
   the unified engine when *both* parsers are `deepseek_v41`. Our lane passes both
   (`serve.sh: --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41
   --enable-auto-tool-choice`), so this is out of scope here — but a tool-only launch of
   this image would still have the non-streaming hole.
9. **Metrics.** The unified parser records a tool-parser invocation per delta (as upstream
   does), so `vllm:tool_call_parser_invocations_total` for the v41 lane will move.

## 6. What the orchestrator must verify on the box, in order

1. **Diff hygiene.** `DIFFS/*.diff` are the only changes vs stock v0.31.0; `deepseek_v4.py`
   is deliberately absent (§2).
2. **The atom resolves.** After mounting, look for exactly one of these two lines at
   startup:
   `dsv41-token-origin-structural-v2: DSML token provenance active (atom id 128825, special=…)`
   or `… provenance INERT …`. INERT means the vocab spells the atom differently and the
   whole rule is a no-op: stop and report, do not enable the lane.
3. **`turn_boundary_tokens` resolve.** `python3 -c` in the image: is `<｜Assistant｜>` a
   single token in the lane tokenizer? If yes, run one multi-turn prompt that already
   closed reasoning and confirm the new turn is classified normally.
4. **In-image suite.** `bash RUN-TESTS.sh.DRAFT` (mounts all 7 overlays, `--network none`).
   Expect `67 passed`. Then `RUN_SWITCH_MATRIX=1` for the switch matrix; the
   provenance-off run is expected to fail the new cases (that is the rollback).
5. **Replay the measured failures.** Replay `results/probe-gb300.jsonl` case 876 in both
   variants (tools / no tools) through the new parser: expect no call and the full text in
   both, then the sweep's `class=CALL_BAD_SHAPE` rows.
6. **Cost.** Compare output tokens/s and TTFT against the current lane on the same
   requests; the strict scanner is on for every token. If the cost is unacceptable, compare
   `VLLM_DSV41_DSML_TOKEN_PROVENANCE=0` (no guard, old speed) before touching anything else.
7. **Streaming vs non-streaming equality.** Same request, stream and non-stream: the
   concatenated `arguments` must equal the non-streaming `arguments` (risks 4).
8. **Rollback rehearsal.** `VLLM_DSV41_DSML_TOKEN_PROVENANCE=0` restart: the lane must come
   up and reproduce the *old* behaviour (a call from the fence sample) — proving the flip
   is a real rollback, not a broken import.

## 7. Files and how they were checked

```
gb300-port/
  REPORT.md                     this file
  MOUNTS.txt                    host -> container bind mounts (7 files)
  RUN-TESTS.sh.DRAFT            exact docker command, derives the image's vllm dir
  overlay/vllm/parser/deepseek_v41.py.DRAFT
  overlay/vllm/parser/parser_manager.py.DRAFT
  overlay/vllm/parser/engine/{token_id_scanner,incremental_lexer,
                              streaming_parser_engine,parser_engine,
                              parser_engine_config}.py.DRAFT
  DIFFS/*.diff                  unified diff of each DRAFT vs stock v0.31.0
  DIFFS/vllm__parser__deepseek_v41.py.vs-current-overlay.diff
                                vs our production overlay (sha256 1c309d19…)
  tests/conftest.py             shim hook (no-op in-image) + tests dir on sys.path
  tests/dsml_fixtures.py        fake DeepSeek-V4.1 tokenizer, tools, 4 schedules,
                                parse_all() driver
  tests/test_dsml_token_provenance.py   cases (a)–(h), 67 tests
  pr/pr65019.diff, pr/split/    the upstream diff we ported
  build/                        build artifacts: patched tree, split hunks, and the
                                Mac-side harness (build/local: real ported parser +
                                stubs for serving-layer leaves + stock tree for the
                                discrimination run). Not part of the mount set.
```

Local verification actually run (Mac, Python 3.12, `regex`/`pydantic`/
`prometheus_client`/`pytest` in a scratch venv, no torch, no network at test time):

```
# delivered configuration, against the real ported files
env VLLM_SHIM_PATH=build/local/shim python -m pytest tests -q      -> 67 passed
# same suite against stock v0.31.0 (no overlays)
env VLLM_SHIM_PATH=build/local/shim_stock python -m pytest tests -q -> 45 failed, 22 passed
```

The Mac harness replaces only leaf serving modules (`vllm.logger`, entrypoints
protocol dataclasses, `ToolParser`/`ReasoningParser` bases, `xgrammar`, `openai`);
`vllm/parser/**` is the real stock/ported code, including the real
`DelegatingParser.parse_delta`. The in-image run in item 4 remains authoritative.
### Artifact checksums (sha256) at hand-off

```
10e283cf829b7e069c67581bddc018c36d561fcaa02a1581ff155a490487f3f0  overlay/vllm/parser/deepseek_v41.py.DRAFT
c5c96c28b944ebac48f121eb1045d4e06bd736644296d15db50fb3d22da4579f  overlay/vllm/parser/parser_manager.py.DRAFT
4e87c0615f769cefd040cbb58925ee9b72efa95db4fc75cd1931e341447be9ff  overlay/vllm/parser/engine/incremental_lexer.py.DRAFT
1cfb96247a2a045759cd88120ea7c248a52a5cc5bd0bfcb6c52e866dd4749323  overlay/vllm/parser/engine/parser_engine.py.DRAFT
831a6d98cc4d23043a0e95bf902ca57380f072bdd2cd5c42ea2a2c03c7594493  overlay/vllm/parser/engine/parser_engine_config.py.DRAFT
1661766656b4f8af7afdb3ba52a07d62919a3f2f24ad3d02486f96a05a7bd9fb  overlay/vllm/parser/engine/streaming_parser_engine.py.DRAFT
7f9c652a09f0880112cef8b588fbae027842ce2cfda09a0606b41e5c13744bdf  overlay/vllm/parser/engine/token_id_scanner.py.DRAFT
fd3acb5af05ae79c64dd5a0643feaa6d60c38e3288a2fedf1f00a8d2ba6a9eb2  tests/conftest.py
7423d622fa8100851548c0a093a8db7ce6db0ccbc79e5796a0b033e38a5e00ed  tests/dsml_fixtures.py
468d13a5333096706421320f55b96d7c5ec1a2f3007002e36d4b6d95be6e7517  tests/test_dsml_token_provenance.py
```

Fixture assumption worth stating out loud: the fake tokenizer models `｜DSML｜` as an
**added** token (in `get_vocab()`, not in `all_special_tokens`). If the lane's tokenizer
listed it as a *special* token, stock would auto-drop it and no text parser could ever
match a marker — so the lane's working behaviour implies added-not-special. The
`special=…` field of the startup log settles it on the box (item 2).
