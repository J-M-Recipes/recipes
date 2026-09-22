# Hermes harness tool-call test — v23 lane, 2026-09-22 ~5:51 PM CDT

`scripts/harness_test.sh` (10 prompts that each require a real tool; `hermes chat --provider custom[mimo26] -m mimo26-pro --reasoning low -t hermes-cli`), then `scripts/score_harness.py` on the saved logs.

**Result: tool-call turns 10/10, correct 10/10.**

- harness_test.sh's own scorer printed `correct=1/10`. That was a scorer bug, not the model: its awk extractor matches `⚕ Hermes` (older CLI header) and read an empty answer on the current CLI (`☤ Hermes`). Every answer was present in the logs.
- score_harness.py re-extracts the answer box and applies the same ground truth: 9/10 on answer text. Prompt 5 (create a file) is a side-effect check; harness_test.sh deletes that file at the start of each loop, so it can only be checked live — harness_test.sh's live check scored it ok=1 and the answer reads "contains exactly: HELLO".

```
[0] tools=1 ok=1  answer='5'
[1] tools=1 ok=1  answer='The file contains exactly one line: MAGIC-7731'
[2] tools=1 ok=1  answer='30007'
[3] tools=1 ok=1  answer='Darwin'
[4] tools=1 ok=1  answer='The third word in /tmp/hx/words.txt is charlie (line 3; the file lists'
[5] tools=2 ok=1  answer='Done. /tmp/hx/out1.txt exists (5 bytes) and contains exactly: HELLO.'
[6] tools=1 ok=1  answer='2'
[7] tools=1 ok=1  answer='7006652'
[8] tools=1 ok=1  answer='echo, gb300'
[9] tools=2 ok=1  answer="Done. Appended 'foxtrot' to /tmp/hx/words.txt — it landed as its own l"
SUMMARY tool_call_turns=10/10 correct=10/10
```

Raw logs (paths scrubbed): `harness/run0.log` … `run9.log`. Wall 9.7–34.9 s per prompt.
