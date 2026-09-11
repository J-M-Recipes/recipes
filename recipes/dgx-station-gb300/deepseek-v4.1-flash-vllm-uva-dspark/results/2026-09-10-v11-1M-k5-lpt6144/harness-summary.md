# Hermes-harness tool-call gate — 2026-09-11 06:06 CDT

Why this gate exists: in a bare `replay.py` (2-line system prompt, no harness) V4.1-Flash *narrated* tool use instead of emitting `tool_calls` on 8/12 tool turns. A speed number means nothing if the model won't call tools under the agent loop it will actually serve. So the promotion gate is the real harness, not the raw API.

Command (× 10 prompts, each unanswerable without a real tool):

    hermes chat --provider 'custom[dsv41]' -m dsv41-flash-uva --reasoning low -t hermes-cli -v -q "<prompt>"

Score: `tool_turns=N` from the harness's own `Turn ended` log line, plus a ground-truth check of the answer or the side effect on disk. Script: `scripts/harness_test.sh`.

| # | prompt (trunc) | tool_turns | correct | wall |
|---|---|---|---|---|
| 0 | How many lines are in /tmp/hx/words.txt? | 1 | ✅ `5` | 13.7 s |
| 1 | Read /tmp/hx/secret.txt and tell me exactly what it says. | 1 | ✅ `MAGIC-7731` | 13.3 s |
| 2 | What port is set in /tmp/hx/cfg.json? | 1 | ✅ `30006` | 8.5 s |
| 3 | Run 'uname -s' and tell me the output verbatim. | 1 | ✅ `Darwin` | 8.7 s |
| 4 | What is the third word in /tmp/hx/words.txt? | 1 | ✅ `charlie` | 10.2 s |
| 5 | Create /tmp/hx/out1.txt containing HELLO, then confirm it exists. | 1 | ✅ file present, content HELLO | 9.8 s |
| 6 | How many .txt files are in /tmp/hx? | 1 | ✅ | 9.7 s |
| 7 | Use a shell to compute 1234*5678. | 1 | ✅ `7006652` | 8.7 s |
| 8 | Read two files; reply last word of first + host from second. | 1 | ✅ `echo,gb300` | 8.1 s |
| 9 | Append 'foxtrot' to words.txt, then report the new line count. | 1 | ✅ file has 6 lines, answer `6` | 10.2 s |

**SUMMARY tool_call_turns=10/10 correct=10/10** · `provider=custom[dsv41] reasoning=low` · wall includes Hermes agent bootstrap (~6 s) and one round-trip.

Notes
- `--reasoning low` is required: the V4.1 chat template rejects `reasoning_effort: medium` with HTTP 400 (accepts `low|high|xhigh|max` or int 1–100).
- Server flags that make this work: `--tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 --enable-auto-tool-choice`.
- Raw per-run harness logs retained off-repo (they include the host's memory-plugin debug lines, not recipe-relevant).
