# Frozen MTP task-quality gate — campaign v1

Status: acceptance rules selected BEFORE scored primary outputs. Pilot requests are development checks, not qualification evidence. Publication is held until real results and independent audit.

## Question and limits
Does slot-cache SC13G with MTP(1) retain useful task performance while improving warm C1 speed? This is a bounded local regression/recipe-selection test, NOT universal model quality, official benchmark replication, token equivalence, or formal proof of mean-score noninferiority. Public HumanEval/GSM8K tasks may be training-contaminated; novel supplied-data and tool fixtures complement but do not remove that limitation.

## Three lanes
1. Original running V1 keeper: descriptive current-service sanity control.
2. SC13G without MTP: sole primary causal comparator.
3. Same SC13G with MTP num_speculative_tokens=1: candidate.
Use the exact preserved containers and sanitized preflight command/mount/config evidence. SC13G controls match except speculative configuration. Do not infer compilation/sequence/offload settings from nicknames; actual docker inspect is authoritative. No :30003 changes, client routing, model/quant/KV changes, or kernel fixes during this campaign. Preserve all keepers; restore V1 after experiment or error.

## Requests and corpus
- 50 seeded official HumanEval tasks + 50 seeded official GSM8K test tasks; fixed source hashes and selected ids.
- Two repeats per task per lane, stable task/repeat seed across lanes; statistical N=100 unique tasks, NOT 200 independent examples.
- temperature=0, top_p=1, max_tokens=4096, chat_template_kwargs.reasoning_effort=low, stream=false. Omit unsupported enable_thinking control.
- Code scored by official executable tests inside isolated CPU Docker, never host Python. Math scored by exact normalized final numeric answer. No semantic LLM judge or exact natural-language continuation matching.
- Freeze harness, fixture data, request policy, and this contract hashes before scored execution. Never change prompts/extractors to rescue a candidate's outputs. Harness defects invalidate affected runs and require an explicit versioned correction and all affected lane reruns.

## Validity
Require expected model id, authenticated completion, separated reasoning/content, structured tool-call replay, sandbox positive and negative controls, fixture uniqueness/hash, complete 100x2 primary outputs per lane, and no uncontrolled concurrent GPU campaign. No retries on wrong model answers. Transport errors/truncation/empty answers/parser failures are reported separately from task accuracy; any unresolved such event blocks promotion as INCONCLUSIVE (or protocol FAIL for a reproducible candidate defect). No silent row dropping or denominator reduction. A clean rerun after transient infrastructure repair is a new disclosed run, not replacing evidence.

## Practical quality decision
Both controls AND candidate must achieve >=85% raw repeat accuracy overall and >=80% in EACH 50-task category. A weak control makes qualification INCONCLUSIVE, not a free pass.
For each unique primary task, compute correct repeat count 0,1,2. Against SC13G:
- loss: candidate correct-repeat count < control count;
- win: candidate count > control count;
- tie: equal counts.
Use exact one-sided 95% Clopper-Pearson upper bound on the proportion of tasks that are losses; PASS requires upper bound <0.05. Wins are reported but do not cancel this conservative gross-loss bound. This is NOT an interval for mean accuracy difference. There is one inferential comparator, so no claim of simultaneous noninferiority versus V1.
The rule allows a small number of task mistakes; it does not require all continuations or all answers to match. With N=100, zero or one loss can pass this bound; verify numerical boundaries in code.
Additionally candidate raw repeat accuracy may be at most 5 percentage points below V1, overall and in each category. This V1 check is descriptive only, without a confidence claim.
FAIL promotion if candidate raw mean accuracy is >5 points below either control, a category is >5 points below either control, or a reproducible candidate protocol/tool defect occurs. Otherwise an upper bound >=5%, incomplete/invalid data, or competence failure means INCONCLUSIVE, not demonstrated harm. No optional-stopping extension to turn an inconclusive run into a pass; discuss a new prespecified campaign if needed.

## Secondary checks (not folded into N=100)
20 supplied-data/structured-output fixtures, two repeats (40 scored outputs), plus four distinct sequential 20-hop tool-chain episodes. All must pass for promotion. Each tool episode must read all 20 opaque records in order, ignore embedded adversarial instruction data, and return the correct terminal token. This exercises raw OpenAI-compatible tool replay, NOT a claim of a real Hermes CLI canary or arbitrary agent competence. Pilot checks are excluded from these denominators. Report repeated synthetic-template dependence honestly.

## Warm C1 speed gate
Separate speed-only bounded-output probes from quality tasks. Warm each short/long prompt class first; 7 scored repeats per class at C1; same input and output-token cap, actual API usage tokens, not SSE chunks. Forced-length speed outputs are intentionally not quality scored. Report per-run wall, actual prompt/completion tokens, effective completion tokens/wall (includes prefill), medians/ranges and warmup exclusions. Require >=10% median effective-TPS gain on short-prompt C1 vs matched SC13G and no >10% median wall regression on the longer-prompt class. Report these as observed repeated measurements, not a statistical universal speedup. No new C4/C8 or max-context qualification is implied.

## Outcome and publication
Primary PASS + competence + secondary PASS + valid data + speed gate PASS + independent audit permits recommendation of this exact MTP configuration in recipe/blog, with experimental limitations and measured settings retained. Any fail/inconclusive blocks that recommendation. Keep raw outputs, scripts, scores, command snapshots, source hashes, and restoration evidence. No publication or winner claim from a test process exiting 0 alone.
