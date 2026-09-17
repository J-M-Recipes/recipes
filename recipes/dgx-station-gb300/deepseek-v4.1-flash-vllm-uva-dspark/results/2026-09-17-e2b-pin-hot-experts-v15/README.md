# Pin Hot Experts v15 — results bundle (2026-09-17)

v15 is the v14 launch (`OFFGB=60 UTIL=0.97`, DSpark `[[1,4,5],[5,16,1]]`, 1M context, same image and weights) plus a bind-mounted hook that re-homes each MoE layer's experts by measured usage (295 hot rows in HBM, 89 cold rows pinned in Grace) and runs the MoE as two `do_finalize=False` routed-kernel calls plus one fp32-FMA finalize. No vLLM or FlashInfer source change; kernels and cubins unchanged. Same-window pair vs the v14 reference, 2026-09-17 14:51–14:58 CDT, **one pair**.

Design note: https://al-engr.com/vllm-pin-hot-experts.html

## Knee (decision instrument)

Prompt class: **prose**, `knee.sh` ("Write a detailed paragraph about the number N"), temperature 0, `ignore_eos`, 192 output tokens. `--max-num-seqs 16`, so C16 is slot-capped; `max_running_requests` was not set.

| | C1 | C2 | C4 | C8 | C12 | C16 |
|---|--:|--:|--:|--:|--:|--:|
| v14 control | 89.3 | 125.0 | 173.3 | 307.7 | 362.9 | 414.3 |
| v15 candidate | 153.1 | 261.0 | 411.2 | 701.4 | 784.6 | 809.0 |

C1 **+71.4%**. C16 candidate runs 666.8 / 951.2 — spread is real; both still ≫ control 413 / 415.

## Other instruments

| | v14 control | v15 |
|---|--:|--:|
| agent prose tok/s | 98.3 | 169.0 |
| structured | 123.0 | 207.6 |
| code | 153.9 | 213.6 |
| shell_ops | 160.2 | 250.3 |
| tool_json | 180.4 | 182.1 |
| weighted accept | 59.9% | 59.6% |
| T5 prose fixture | 91.4 | 152.5 |
| replay n=4 r1 / r2 | 170.6 / 222.7 | 247.5 / 431.3 |

Accept rates bit-identical on prose / structured / code / shell (29.8 / 49.5 / 63.8 / 90.8). tool_json 89.7→87.8. No category −3%. Replay swing is the known instrument; direction matches the knee. Mean replay 196.6 → 339.4.

## Parity

Greedy T=0, agent-fixture prompts, C1: control vs itself 8/8 content+tool-args exact (tool_call **ids** differ, also on v14 vs itself). Candidate vs control: **8/8** same. Gate PASS. Acceptance-rate identity as above.

## Memory / autotune

HBM_expert **206.61 GiB**, pinned_expert **62.33 GiB** (expect 206.6 / 62.4). KV **4.75 GiB** (v14 4.89, within 0.5). Autotune hash `9ac7b387`: Loaded 210 → Saved 231 (**21 new**, ~12 min — not 150).

## What is not yet done

1. Second same-window pair (candidate→control→candidate; knee ×2 each) before quoting publicly.
2. Parity on a larger set (≥50 prompts, including prose/code/multilingual) and the round-7 gauntlet 36/36.
3. True cross-domain check — the prose fixture was in the E2a profile, so 152.5 tok/s is in-domain.

## Corrections

- **C16 moved ~2×**, contrary to the "C1–C4 lever" expectation. At C16 the positional design still reads most of the offloaded set per step; the usage-aware design reads only the ~1.6% of selections that are cold.
- **Autotune 12 min, not 150.** Only 21 new profiles (the 295/89 split shapes).

## Why this design (spikes, not adopted)

- [`e1b/README.md`](e1b/README.md) — F4 split is bit-exact (two unfinalized calls + one fp32-FMA finalize in k order). Two-finalize split is not.
- [`e1c/README.md`](e1c/README.md) — split (C2) beats staging (C3) beats whole-layer host read (C1) at decode shapes.
- [`track1/O1a-address-path.md`](track1/O1a-address-path.md) — single-VA / cubin change is upstream, not this recipe.
- [`track1/O1b-vmm-rescues.md`](track1/O1b-vmm-rescues.md) — VMM host-page-table rescues dead (best ~91 GB/s).
- [`e2a/README.md`](e2a/README.md) — count-only profile: 34.4M decode selections, own-cold 1.55%; count hook is not free (−3.7%). Rowmap `rowmap-static-v1.json` (not T5). Included npz: `e2a/counts-20260917-183654.npz`; others listed in `e2a/COUNTS-NPZ.txt`.

## Launch

Exact container args: [`v15-container-args.txt`](v15-container-args.txt). Bind-mount `hook/sitecustomize.py` over `/usr/lib/python3.12/sitecustomize.py` and the hook dir at `/w`. Env: `PIN_MODE=split PIN_HOOK=/w/pin_hot_experts_hook.py PIN_ROWMAP=/w/rowmap-static-v1.json`. Image `vllm/vllm-openai:deepseekv41-flash-0909` @ `sha256:00d577a6…`. Model revision `df42c109f1defefcbfcedbe7d905718a12266e40`.

Raw knee/fixture/replay/parity: [`e2b/`](e2b/). Hook sources: [`hook/`](hook/).
