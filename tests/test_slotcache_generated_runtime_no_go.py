import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RECIPE = REPO_ROOT / "recipes/dgx-station-gb300/glm-5.3-nvfp4-uva-slot-cache"
PATCHES = RECIPE / "patches"
RUNTIME_GPU_MODEL_RUNNER = Path(
    os.environ.get(
        "K2_V3_RUNTIME_GPU_MODEL_RUNNER",
        REPO_ROOT / "tests/fixtures/k2-v3-runtime-source/vllm/v1/worker/gpu_model_runner.py",
    )
)
PYTHON = sys.executable


def _load_patch_script():
    spec = importlib.util.spec_from_file_location(
        "apply_slot_cache_instrumentation_patch",
        RECIPE / "scripts/apply_slot_cache_instrumentation_patch.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _patched_source():
    return _load_patch_script().patch_source(RUNTIME_GPU_MODEL_RUNNER.read_text())


def _init_guard_source(patched: str) -> tuple[str, str]:
    import_prefix = patched.split("import numpy as np", 1)[0]
    init_line = "if SlotCacheWindowController is None:"
    init_start = patched.index(init_line) - 8
    init_end = patched.index("\n\n        self.check_ep_fault = False", init_start)
    init_src = "def init_seam(self, parallel_config):\n" + "\n".join(
        "    " + line[8:] for line in patched[init_start:init_end].splitlines()
    ) + "\n"
    return import_prefix, init_src


def test_generated_init_guard_rejects_enabled_non_single_gpu_topology_before_launch(tmp_path, monkeypatch):
    patched = _patched_source()
    import_prefix, init_src = _init_guard_source(patched)
    monkeypatch.syspath_prepend(str(PATCHES))
    monkeypatch.setenv("SLOT_CACHE_QUIESCENT_SNAPSHOTS", "1")
    monkeypatch.setenv("SLOT_CACHE_SNAPSHOT_DIR", str(tmp_path))

    ns = {"Any": object}
    exec(import_prefix + "\n" + init_src, ns)

    unsupported = SimpleNamespace(
        pipeline_parallel_size=2,
        tensor_parallel_size=1,
        data_parallel_size=1,
        decode_context_parallel_size=1,
        use_ubatching=False,
    )
    with pytest.raises(RuntimeError, match="only supports single-GPU.*pipeline_parallel_size=2"):
        ns["init_seam"](SimpleNamespace(), unsupported)

    disabled = SimpleNamespace(
        pipeline_parallel_size=2,
        tensor_parallel_size=2,
        data_parallel_size=2,
        decode_context_parallel_size=2,
        use_ubatching=True,
    )
    monkeypatch.setenv("SLOT_CACHE_QUIESCENT_SNAPSHOTS", "0")
    runner = SimpleNamespace()
    ns["init_seam"](runner, disabled)
    assert runner.slot_cache_window_controller is None


def test_generated_methods_finalize_failed_sample_before_next_fallback_step(tmp_path):
    patched = tmp_path / "gpu_model_runner.patched.py"
    patched.write_text(_patched_source())
    probe = tmp_path / "sample_exception_probe.py"
    probe.write_text(
        f"""
from __future__ import annotations
import ast, sys, importlib.util, contextlib, collections
from pathlib import Path
from types import SimpleNamespace
def replace(obj, **kw):
 ns=SimpleNamespace(**obj.__dict__); ns.__dict__.update(kw); return ns
import numpy as np
PATCHED=Path({str(patched)!r})
INST_PATH=Path({str(PATCHES / 'slot_cache_window_instrumentation.py')!r})
spec=importlib.util.spec_from_file_location('slot_cache_window_instrumentation', INST_PATH)
inst=importlib.util.module_from_spec(spec); sys.modules[spec.name]=inst; spec.loader.exec_module(inst)
src=PATCHED.read_text(); tree=ast.parse(src)
cls=next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=='GPUModelRunner')
methods=[next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name==name) for name in ('execute_model','sample_tokens')]
for m in methods: m.decorator_list=[]
mod=ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias('annotations')], level=0), ast.ClassDef(name='ExtractedRunner', bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[]); ast.fix_missing_locations(mod)
calls=[]; pushes=[]
class Ctx:
 def __enter__(self): return self
 def __exit__(self,*exc): return False
def record_function_or_nullcontext(*a,**k): return Ctx()
def nullcontext(*a,**k): return contextlib.nullcontext(*a,**k)
def set_forward_context(*a,**k): return Ctx()
class Logger:
 def debug(self,*a,**k): pass
logger=Logger()
class FakeHidden:
 tensors={{'h':1}}
 def __getitem__(self, idx): return self
 def contiguous(self): return self
class FakeModel:
 def compute_logits(self,h): return 'logits'
class BatchDesc: num_tokens=1; num_reqs=None
class CUDAGraphMode: FULL='FULL'
def maybe_create_ubatch_slices(*a,**k): return (None,None)
class EncoderOnlyAttentionSpec: pass
def has_kv_transfer_group(): return False
def has_ec_transfer(): return False
class PP:
 is_last_rank=True; world_size=1; ranks=[0]; rank=0; last_rank=0; device_group=None
 def send_tensor_dict(self,*a,**k): pass
 def broadcast_tensor_dict(self,d,**k): return d
def get_pp_group(): return PP()
def get_tp_group(): return object()
def is_residual_scattered_for_sp(*a,**k): return False
def apply_grammar_bitmask(*a,**k): pass
ExecuteModelState=collections.namedtuple('ExecuteModelState','scheduler_output logits spec_decode_metadata spec_decode_common_attn_metadata hidden_states sample_hidden_states aux_hidden_states ec_connector_output cudagraph_stats slot_mappings')
class ModelRunnerOutput:
 def __init__(self, **kw): self.__dict__.update(kw)
 @classmethod
 def with_kv_conn_output_only(cls, kv): return cls(kv_connector_output=kv)
EMPTY_MODEL_RUNNER_OUTPUT=ModelRunnerOutput(empty=True)
class AsyncGPUModelRunnerOutput: pass
class IntermediateTensors: pass
class RoutedExpertsLists: pass
class EagleProposer: pass
class DFlashProposer: pass
class DraftModelProposer: pass
class ExtractHiddenStatesProposer: pass
class Gemma4Proposer: pass
class NgramProposerGPU: pass
class Torch:
 class cuda:
  class nvtx:
   @staticmethod
   def range_push(x): pushes.append(x)
   @staticmethod
   def range_pop(): pass
 int32='int32'
 @staticmethod
 def zeros(*a,**k): return 'zeros'
torch=Torch
sys.modules['torch']=Torch
def finalize_slot_cache_snapshot(controller, snapshot, cpu=None): controller.finalize_slot_cache_snapshot(snapshot, cpu)
globals_ns=dict(globals()); exec(compile(mod, str(PATCHED), 'exec'), globals_ns); ExtractedRunner=globals_ns['ExtractedRunner']
class CachedReqs:
 req_ids=['d']
 def is_context_phase(self, req_id): return False
def sched(): return SimpleNamespace(total_num_scheduled_tokens=1,num_scheduled_tokens={{'d':1}},scheduled_cached_reqs=CachedReqs(),scheduled_new_reqs=[],scheduled_spec_decode_tokens={{}},kv_connector_metadata=None,scheduled_encoder_inputs=[],num_common_prefix_blocks=0,finished_req_ids=[],new_block_ids_to_zero=[],kv_cache_block_copies={{}},free_encoder_mm_hashes=[])
class SpecConfig:
 disable_padded_drafter_batch=False
 def use_ngram_gpu(self): return False
 def use_eagle(self): return False
 def uses_draft_model(self): return False
 def uses_extract_hidden_states(self): return False
 def use_dflash(self): return False
class InputBatch:
 num_reqs=1; req_ids=['d']; vocab_size=32000; sampling_metadata='sampling'; prev_sampled_token_ids=None
 num_computed_tokens_cpu=np.array([0]); num_accepted_tokens_cpu=np.array([1])
 def set_async_sampled_token_ids(self,*a,**k): pass
class NumAccepted:
 np=np.array([0])
 def copy_to_gpu(self,n): pass
class Runner(ExtractedRunner):
 def synchronize_input_prep(self): return Ctx()
 def _update_states(self,scheduler_output): return None
 def _prepare_inputs(self,*a): return ('logit_idx',None,1)
 def _determine_batch_execution_and_padding(self,**kw): return (CUDAGraphMode.FULL,BatchDesc(),False,1,SimpleNamespace(mode='FULL'))
 def _allow_microbatching(self,*a): return False
 def _get_slot_mappings(self,**kw): return ({{}},'slot_mappings')
 def _build_attention_metadata(self,**kw): return ('attn_meta','common_attn')
 def _preprocess(self,*a): return ('input_ids',None,'positions',None,{{}},'ec')
 def maybe_get_kv_connector_output(self,*a,**k): return Ctx()
 def _model_forward(self,**kw): return FakeHidden()
 def _sample(self, logits, spec_meta): raise RuntimeError('synthetic sample failure')
 def _update_states_after_model_execute(self,*a): pass
 def _input_fits_in_drafter(self, common): return False
 def _copy_draft_token_ids_to_cpu(self,*a,**k): pass
 def _bookkeeping_sync(self,*a): return (0,None,[],['tok'],{{}},['d'],{{'d':0}},[])
 def finalize_kv_connector(self): pass
 def eplb_step(self): pass
 def _get_or_create_async_output_copy_stream(self): return 'stream'
 def get_routed_experts(self,total): return None
r=Runner(); r.execute_model_state=None; r.speculative_config=SpecConfig(); r.parallel_config=SimpleNamespace(distributed_executor_backend='x',data_parallel_size=1,use_ubatching=False,num_ubatches=1); r.cache_config=SimpleNamespace(kv_sharing_fast_prefill=False,mamba_cache_mode='none'); r.input_batch=InputBatch(); r.cascade_attn_enabled=False; r.kv_cache_config=SimpleNamespace(kv_cache_groups=[]); r.attn_groups={{}}; r.model_config=SimpleNamespace(is_encoder_decoder=False); r.eplb_state=None; r.vllm_config=object(); r.broadcast_pp_output=False; r.is_pooling_model=False; r.use_aux_hidden_state_outputs=False; r.supports_mm_inputs=False; r.kv_connector_output=None; r.num_accepted_tokens=NumAccepted(); r.drafter=None; r.num_spec_tokens=0; r.effective_drafter_max_model_len=999; r.valid_sampled_token_count_event=None; r.device='cuda:0'; r.discard_request_mask=SimpleNamespace(gpu='mask'); r.requests={{}}; r.use_async_scheduling=False; r.routed_experts_initialized=False; r.check_ep_fault=False; r.slot_cache_window_controller=inst.SlotCacheWindowController(enabled=True, snapshot_dir=Path('/private/tmp'), windows=(inst.SlotCacheWindow(1,2),), run_id='realgen', k_mode='K2'); r.model=FakeModel(); r.use_v2_model_runner=False
r.execute_model(sched())
try:
 r.sample_tokens(None)
except RuntimeError as exc:
 assert 'synthetic sample failure' in str(exc)
r.execute_model(sched())
trace_pushes=[p for p in pushes if isinstance(p,str) and p.startswith('slotcache:')]
print({{'local_step': r.slot_cache_window_controller.local_step, 'trace_pushes': trace_pushes}})
assert any(':boundary:sample_exception:' in x and ':step:1:' in x for x in trace_pushes), trace_pushes
assert any(':boundary:target_forward:' in x and ':step:2:' in x for x in trace_pushes), trace_pushes
r_target=Runner(); r_target.__dict__.update(r.__dict__.copy()); r_target.slot_cache_window_controller=inst.SlotCacheWindowController(enabled=True, snapshot_dir=Path('/private/tmp'), windows=(), run_id='realgen-target', k_mode='K2'); r_target.use_v2_model_runner=False; r_target.execute_model_state=None
original_target=ValueError('synthetic target identity')
def fail_target(**kw): raise original_target
r_target._model_forward=fail_target
try:
 r_target.execute_model(sched())
except ValueError as exc:
 assert exc is original_target
else:
 raise AssertionError('target exception must propagate')
r_draft=Runner(); r_draft.__dict__.update(r.__dict__.copy()); r_draft.slot_cache_window_controller=inst.SlotCacheWindowController(enabled=True, snapshot_dir=Path('/private/tmp'), windows=(), run_id='realgen-draft', k_mode='K2'); r_draft.use_v2_model_runner=False; r_draft.execute_model_state=None
class SpecConfigDraft(SpecConfig):
 def use_ngram_gpu(self): return True
r_draft.speculative_config=SpecConfigDraft(); r_draft.drafter=NgramProposerGPU(); r_draft._input_fits_in_drafter=lambda common: True; r_draft._sample=lambda logits, spec_meta: SimpleNamespace(sampled_token_ids='sampled')
original_draft=ValueError('synthetic draft identity')
def fail_draft(*a, **k): raise original_draft
r_draft.propose_draft_token_ids=fail_draft
r_draft.execute_model(sched())
try:
 r_draft.sample_tokens(None)
except ValueError as exc:
 assert exc is original_draft
else:
 raise AssertionError('draft exception must propagate')
r.slot_cache_window_controller=None; r.use_v2_model_runner=True; r.execute_model_state=None
r.execute_model(sched())
assert r.execute_model_state is not None
print('GENERATED_SAMPLE_EXCEPTION_PROBE_OK')
"""
    )
    result = subprocess.run([PYTHON, str(probe)], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
