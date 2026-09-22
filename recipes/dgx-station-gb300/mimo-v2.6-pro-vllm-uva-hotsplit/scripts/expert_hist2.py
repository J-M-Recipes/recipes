"""Routed-expert histogram v2: real Hermes turns (private corpus), split train/holdout, prefill vs decode.
Eager mode (no cudagraphs) so the counting hook never runs under capture and padded tokens never pollute counts.
Decode steps are those with num_tokens <= max_num_seqs (mixed steps land in 'prefill' - conservative for decode).
Output: {"train": {"prefill": {L: [384]}, "decode": {...}}, "holdout": {...}, "meta": {...}}
"""
import os, sys, json, torch, time
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"; os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
M, CORPUS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
MAX_SEQS = 16
from vllm import LLM, SamplingParams
from vllm.model_executor.layers.fused_moe.router import fused_moe_router as fmr
N_EXP = 384
_layer_of: dict[int, int] = {}
phase = {"split": None}
counts = {}  # (split, kind, layer) -> tensor
_orig = fmr.FusedMoERouter.select_experts
def patched(self, hidden_states, router_logits, topk_indices_dtype=None, *, input_ids=None):
    w, ids = _orig(self, hidden_states, router_logits, topk_indices_dtype, input_ids=input_ids)
    sp = phase["split"]
    if sp is None or torch.cuda.is_current_stream_capturing():
        return w, ids
    li = _layer_of.get(id(self))
    if li is None:
        return w, ids
    kind = "decode" if ids.shape[0] <= MAX_SEQS else "prefill"
    key = (sp, kind, li)
    c = torch.bincount(ids.flatten().to(torch.int64), minlength=N_EXP)
    counts[key] = counts[key] + c if key in counts else c
    return w, ids
fmr.FusedMoERouter.select_experts = patched
llm = LLM(model=M, tensor_parallel_size=1, max_model_len=16384, max_num_seqs=MAX_SEQS, gpu_memory_utilization=0.96,
          enforce_eager=True, trust_remote_code=True, moe_backend="marlin", served_model_name="mimo26-pro",
          offload_backend="uva", cpu_offload_gb=320.0, cpu_offload_params=["routed_experts.w13_weight", "routed_experts.w2_weight"],
          max_num_batched_tokens=4096, generation_config="auto", enable_prefix_caching=False)
mr = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.model_runner
lm = getattr(mr.model, "language_model", mr.model)
for i, layer in enumerate(lm.model.layers):
    exp = getattr(getattr(layer, "mlp", None), "experts", None)
    r = None
    if exp is not None:
        r = getattr(exp, "router", None) or getattr(getattr(exp, "runner", None), "router", None)
    if r is not None: _layer_of[id(r)] = i
print("routers mapped:", len(_layer_of), flush=True)
convs = json.load(open(CORPUS))
tok = llm.get_tokenizer()
def fits(c):
    try:
        n = len(tok.apply_chat_template(c, tokenize=True, add_generation_prompt=True))
    except Exception:
        return False
    return n <= 16384 - 160
convs = [c for c in convs if fits(c)]
cut = int(len(convs) * 0.7)
splits = {"train": convs[:cut], "holdout": convs[cut:]}
print(f"convs usable {len(convs)}: train {cut} holdout {len(convs)-cut}", flush=True)
sp = SamplingParams(max_tokens=128, temperature=0.6, top_p=0.95, seed=0)
meta = {}
for name, cs in splits.items():
    phase["split"] = name; t0 = time.time()
    outs = llm.chat(cs, sp, chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
    gen = sum(len(o.outputs[0].token_ids) for o in outs); pr = sum(len(o.prompt_token_ids) for o in outs)
    meta[name] = {"convs": len(cs), "prompt_tokens": pr, "gen_tokens": gen, "secs": round(time.time() - t0)}
    print(name, meta[name], flush=True)
    phase["split"] = None
res = {"meta": meta}
for (s, k, li), c in counts.items():
    res.setdefault(s, {}).setdefault(k, {})[str(li)] = c.cpu().tolist()
json.dump(res, open(OUT, "w"))
print("saved", OUT, flush=True)
