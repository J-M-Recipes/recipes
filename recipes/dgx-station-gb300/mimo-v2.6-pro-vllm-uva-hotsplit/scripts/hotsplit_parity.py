"""hotsplit parity: truncated Pro (NL layers), UVA offload, greedy decode on fixed prompts.
Dumps token ids + per-layer last-token residual for the first prompt so stock vs hotsplit can be diffed.
usage: hotsplit_parity.py <model> <NL> <out.pt>   (HOTSPLIT_COUNTS set => hotsplit on)
"""
import os, sys, torch, time
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"; os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
M, NL, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
from vllm import LLM, SamplingParams
llm = LLM(model=M, tensor_parallel_size=1, max_model_len=4096, max_num_seqs=4, gpu_memory_utilization=float(os.environ.get("UTIL", "0.9")),
          hf_overrides={"num_hidden_layers": NL}, enforce_eager=True, trust_remote_code=True, served_model_name="mimo26-pro",
          moe_backend="marlin", offload_backend="uva", cpu_offload_gb=float(os.environ.get("OFFGB", "40")),
          cpu_offload_params=["routed_experts.w13_weight", "routed_experts.w2_weight"])
mr = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.model_runner
lm = getattr(mr.model, "language_model", mr.model)
layers = lm.model.layers
split = sum(1 for l in layers if hasattr(getattr(getattr(l, "mlp", None), "experts", None), "quant_method") and False)
dump = {}
cap = {"on": False}
def mk(i):
    def hook(mod, args, kwargs, out):
        if not cap["on"]: return
        h = (out[0].float() + out[1].float()) if isinstance(out, tuple) and len(out) == 2 and out[1] is not None else (out[0] if isinstance(out, tuple) else out).float()
        dump.setdefault(i, []).append(h[-1].detach().cpu())
    return hook
for i, l in enumerate(layers): l.register_forward_hook(mk(i), with_kwargs=True)
nsplit = 0
for n, m in mr.model.named_modules():
    if getattr(getattr(m, "quant_method", None), "_hotsplit_parts", None) is not None: nsplit += 1
print("hotsplit layers active:", nsplit, "HBM free GiB", round(torch.cuda.mem_get_info()[0] / 2**30, 1), flush=True)
PROMPTS = ["Say hello in five words.", "What is the capital of France? One word.", "Write a haiku about autumn.",
           "List three prime numbers.", "Explain what a mixture-of-experts model is in two sentences.",
           "Write a Python function that reverses a linked list."]
sp = SamplingParams(max_tokens=48, temperature=0.0, logprobs=1)
cap["on"] = True
first = llm.chat([{"role": "user", "content": PROMPTS[0]}], SamplingParams(max_tokens=1, temperature=0.0), chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
cap["on"] = False
t0 = time.time()
outs = llm.chat([[{"role": "user", "content": p}] for p in PROMPTS], sp, chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
dt = time.time() - t0
res = {"layers": {k: torch.stack(v) for k, v in dump.items()}, "tokens": [list(o.outputs[0].token_ids) for o in outs],
       "logprobs": [[list(d.values())[0].logprob for d in o.outputs[0].logprobs] for o in outs],
       "text": [o.outputs[0].text for o in outs], "nsplit": nsplit, "secs": dt}
torch.save(res, OUT)
for p, t in zip(PROMPTS, res["text"]): print(f"  {p[:30]!r:34s} -> {t[:70]!r}", flush=True)
print("saved", OUT, "gen secs", round(dt, 1), flush=True)
