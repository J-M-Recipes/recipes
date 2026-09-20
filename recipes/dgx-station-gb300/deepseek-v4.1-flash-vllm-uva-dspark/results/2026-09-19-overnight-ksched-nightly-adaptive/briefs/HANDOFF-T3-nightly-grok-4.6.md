# HANDOFF — T3 / T3b: nightly vLLM image swap (hook off, then on) — for a Grok 4.6 worker

Operator brief from Milo for James Meadlock's pin-hot-experts project. All rules in `HANDOFF-grok-4.6.md` apply (never `:30003`, never `glm53-*`, never `docker rm`, drop caches before a launch, fast-fail grep `ValueError:|Traceback|ERROR|RuntimeError:`, stop means stop / `STOP-CAMPAIGN` file, findings only, no invented numbers). Ledger `results/ledger.md` (`grok-4.6 · xai-oauth`). Box `ssh $BOX_USER@$BOX_HOST`. Mac project `$HOME/hermes/pin-hot-experts/`. Plan of record: `OVERNIGHT-PLAN-2026-09-18.md`. **Precondition: T1/T2 are finished and v18 is back up on `:30006`** (check `results/2026-09-19-overnight/T2/README.md` exists).

**`docker pull` rule for this run:** James pre-approved exactly ONE image, and Milo already pulled it: `vllm/vllm-openai:nightly-dee37d89115db4c94a820a79a78a7828e141c910`. Confirm with `docker image inspect` before anything. **You do not pull.** One exception, stated below, and only on the exact failure named.

## Why

The live image `deepseekv41-flash-0909` was built 2026-09-10 06:47Z from fork commit `e47aa780` — hours before DSV4.1 merged to main and before every kernel PR that followed. Nightly `dee37d89` (2026-09-18) contains #56935 (FlashMLA mega attention + NVFP4 compressed KV), #56962 (Mega-mHC), #56266 (Mega-Gate), #56464 (DeepSelect), #56568/#57204 (MegaMoE shared-expert fusion), #56512 (Engram async prefetch, CPU offload default). All v18 flags still exist under the same names on main (`arg_utils.py`): `--offload-backend uva`, `--cpu-offload-gb`, `--cpu-offload-params`, `--engram-config` (now redundant but valid), `method=dspark`, `num_speculative_tokens_per_batch_size`, `--long-prefill-token-threshold`, `--cudagraph-capture-sizes`. A new image is a new autotune hash: **arm `seed_autotune.sh` before every launch; a full 74-min tune is allowed tonight.** Record hit/miss and minutes.

Two steps. Step 1 is the image on its own (positional offload, no hook). Step 2 adds the v15 hook **only if** its target class still exists in the new image.

## Step 1 — T3, hook off

```
cd $BOX_HOME/pin-hot-experts/scripts   # copy launch-t3-nightly.sh here from the Mac project scripts/
docker image inspect vllm/vllm-openai:nightly-dee37d89115db4c94a820a79a78a7828e141c910 >/dev/null && echo present
```
Stop-and-keep v18. Drop caches. `nohup bash $BOX_HOME/dsv41/seed_autotune.sh > $BOX_HOME/dsv41/seed-T3.log 2>&1 &` then `HOOK=0 bash launch-t3-nightly.sh`.

Record from the log: vLLM version line, `Engram` lines (offload path, prefetch), `Graph capturing finished` GiB, `GPU KV cache size`, autotune cache file hash + `Loaded N configs` **or** tune duration, bind time, any `WARNING` mentioning deprecated flags. **If the container dies at import with a CUDA driver/runtime mismatch (text contains `CUDA driver version is insufficient` or `cudaErrorInsufficientDriver` or a torch `RuntimeError` naming the driver):** that is the one exception — record the exact error, then `docker pull vllm/vllm-openai:nightly-dev-$(...)`: find the newest tag matching `nightly-dev-*-cu13.0.1-*` on Docker Hub (`curl -s "https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags?page_size=100&name=cu13.0.1" | python3 -c 'import json,sys;[print(t["name"],t["last_updated"]) for t in json.load(sys.stdin)["results"]]'`), pull that one tag, relaunch with `IMAGE=` overridden. Any other failure: stop, report, restore v18.

Then, same window: `smoke_vllm.sh` (must be 5/5 incl. parsed `tool_call` and thinking→`36`), `knee.sh` ×2, `CAT_ORDER="prose structured code shell_ops tool_json" agent_fixture_o.sh`, `cold_prefill_probe.py` with `SIZES="8000 32000 128000" N=2 THINKING=0`.

Comparison point for Step 1 is **v14** (positional, no hook, seqs 16): C1 88.7 / C8 305 / C16 386–417; fixture prose 98 / code 154 / shell 160 / tool_json 180; idle 104K prefill 22.0K tok/s. Seqs differ (24 vs 16) so C16 is not slot-capped here; note it. **Continue to Step 2 only if:** smoke 5/5 **and** C1 ≥ 93 (v14 +5%) **and** no fixture class < v14 −3%. Otherwise stop-and-keep as `-EXP-<reason>-FAIL`, restore v18, write up, done.

## Step 2 — T3b, hook on (class check first, no exceptions)

Before booting, run inside the new image (no GPU needed):
```
docker run --rm --entrypoint python3 vllm/vllm-openai:nightly-dee37d89115db4c94a820a79a78a7828e141c910 - <<'EOF'
import importlib, inspect, sys
cands = ["vllm.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe",
         "vllm.model_executor.layers.fused_moe.trtllm_mxfp4_moe",
         "vllm.models.deepseek_v4_1.nvidia.moe", "vllm.model_executor.layers.fused_moe.experts.trtllm_moe"]
found=None
for m in cands:
    try:
        mod=importlib.import_module(m)
    except Exception as e:
        print("no", m, type(e).__name__); continue
    for n in dir(mod):
        if "Mxfp4" in n and "Modular" in n:
            cls=getattr(mod,n); found=(m,n,cls); print("FOUND", m, n)
if not found:
    import pkgutil, vllm; print("grep fallback:"); 
    import subprocess; print(subprocess.run(["grep","-rl","TrtLlmMxfp4ExpertsModular","/usr/local/lib/python3.12/dist-packages/vllm"],capture_output=True,text=True).stdout)
    sys.exit(2)
m,n,cls=found
print("has _invoke_kernel:", hasattr(cls,"_invoke_kernel"))
print(inspect.signature(cls._invoke_kernel) if hasattr(cls,"_invoke_kernel") else "NO _invoke_kernel")
EOF
```
Paste the output into `results/2026-09-19-overnight/T3/README.md`. Compare the signature to the one the hook wraps (`hook/pin_hot_experts_hook.py`, `_invoke_split` / `_orig_invoke` call site). **If the class is missing, renamed, or the signature differs in any positional argument: do not boot T3b, do not edit the hook.** Write the finding and the new path/signature, restore v18, stop. That is a successful outcome for tonight — it tells Milo exactly what to port.

If identical: stop-and-keep T3, arm `seed_autotune.sh` (T3's hash should now hit), `HOOK=1 bash launch-t3-nightly.sh`. Verify hook lines (`hot=295 cold=89`, `HBM_expert=206.61GiB`), KV, graphs. Windows **T3b → v18ctl → T3b**: `knee.sh` ×2, `agent_fixture_o.sh`, `python3 scripts/e2c_parity.py` (18 prompts, T=0, seed 42, 256 tok) vs v18ctl. Bars: C1 ≥ +5% over v18ctl two-window mean, C8/C16 ≥ 0%, parity non-tool exact 12/12 (tool-prompt jitter is known), no fixture class < −3%. Pass → leave T3b UP, v18 stopped-and-kept, Milo promotes. Fail → `-EXP-<reason>-FAIL`, restore v18.

## Outputs

`results/2026-09-19-overnight/T3/{README.md, boot-log-excerpt.txt, class-check.txt, smoke.txt, knee-*.json, agentfix-*.json, coldprefill-*.json[, parity-*.json]}`, ledger entries. Verdicts:
`T3 VERDICT: image dee37d89 boot ok/fail (…) · hash hit yes/no (… min) · KV … · smoke …/5 · knee C1/C8/C16 … vs v14 88.7/305/401 · fixture … · idle 104K … tok/s · step2 eligible yes/no`
`T3b VERDICT: class … present/moved (…) · [knee C1/C8/C16 Δ…/…/…% vs v18 · parity …/12 · fixture min Δ…%] · left up: T3b|v18`
≤ 15-line final summary, then stop.
