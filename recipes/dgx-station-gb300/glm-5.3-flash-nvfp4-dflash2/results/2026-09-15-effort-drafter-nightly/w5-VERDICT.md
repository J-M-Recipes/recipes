# W5 verdict — `--enable-linear-replayssm-spec` on 8874c51a — CLOSED AT BOOT

Window: 2026-09-16T01:37:50Z → 02:08:36Z. Control booted and instrumented; axis rejected by the engine before pool allocation. Both containers stopped+kept. Nothing restored.

## Rejection (verbatim)

```
File ".../sglang/srt/mem_cache/kv_cache_configurator.py", line 1118, in _build_hybrid_req_pool
    raise ValueError(
ValueError: --enable-linear-replayssm-spec with DSPARK/DFLASH requires a KDA (kimi_linear) model; got a non-KDA model.
```

Source comment at the guard: "DSPARK/DFLASH commit routes through the backend fold (KDA-only); a non-KDA model there would scatter a None intermediate_ssm and crash." The check is `kimi_linear_config(self.model_config) is None` — it keys on the Kimi config shape, and `glm5_next` (`Glm5NextForConditionalGeneration`) does not present as one, even though W3's boot log confirmed `KDA fused chain-verify kernel enabled` on this same model. The kernel path accepts GLM-5.3 as KDA-hybrid; the pool guard does not.

Not a ring-length reject → no retry path. **W5 and W5b closed on this image.** The handoff's W5 precondition list (`attention_hook.py`) missed this guard; that was a code-read gap, not an engine regression.

Upstream ask (not filed): widen the guard from `kimi_linear_config` to the layer-type check the fused-verify path already uses, or document that DFLASH+replayssm-spec is Kimi-only. Receipt: `glmf-w5-replayssm.log` on the Station.

## Salvage — block-7 control `glmf-w5-ctrl` (same-session control, same knobs as W6c-b7)

| Metric | Value | Note |
|---|---|---|
| C1 recipe-method low | 201.0 (199.8/201.0/201.7) | W2 202.7, W6c-b7 202.6 |
| prose / code / shell low | 153.4 / 284.4 / 230.0 | shell 230 vs W6c-b7 162 — check `c1-low.txt` vs `accept-low.txt` class defs before quoting |
| Accept mean low | len 3.48 / rate 0.41 | W6c-b7 3.46 / 0.41 |
| Tools | 10/10 ×2 | |
| Greedy vs w2-a | 20/20 | |
| **C8 clean** | **732.3 agg / 91.5 per-stream** (754/734/709, 6%) | first pass leaked (112%), auto rewarm+rerun. Replaces void W6c-b7 row |
| tf self-score | 2857 tokens, `instrument_repeat_identical: true` | `w5-ctrl/tf-w5-ctrl.json` — reusable reference for W4 |
| Mamba pool @48 | conv 0.24 + ssm 6.70 + **intermediate 9.57** + conv-window 0.34 GB | the buffer W5 was meant to drop |

## Runner bugs fixed mid-window (for the next runner)
- `tf_noninferiority.py` builds its own `/v1` — pass `BASE_URL=http://127.0.0.1:30001` (no `/v1`) or it 404s on `/v1/v1/completions`.
- `accept_probe.py` scrapes `$CONT`; export it globally from the boot function, not `local`.
