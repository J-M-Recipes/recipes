"""Desk F DRAFT: build_candidate.py (build-w1-hardened, unchanged otherwise) + extra -D flags from W1_EXTRA_CFLAGS.

CPU-only nvcc build of one isolated W1 variant; never installs into a serving tree. The original build_candidate.py
hard-codes extra_cuda_cflags, so a -D cannot be passed to it without editing it; this copy appends the flags given
in the environment variable W1_EXTRA_CFLAGS (space separated, each must start with -DW1_). Same compiler, same
arch (sm_103a), same base flags, same module name w1_attn_fp8_ext, same -Dw1=w1_fp8_candidate namespace rename.
Use a fresh /work (fresh build/ directory) per variant: torch's load() renames the module to *_v1 if it finds a
previous build with different flags in the same build directory.
"""
import hashlib
import json
import os
from pathlib import Path
from torch.utils.cpp_extension import load

root = Path('/work')
extra = os.environ.get('W1_EXTRA_CFLAGS', '').split()
for f in extra:
    assert f.startswith('-DW1_'), f'refusing non-W1 flag {f!r}'
os.environ['TORCH_CUDA_ARCH_LIST'] = '10.3a'
os.environ['MAX_JOBS'] = '2'
nvcc = Path('/usr/local/lib/python3.12/dist-packages/nvidia/cu13/bin/nvcc')
if nvcc.exists():
    os.environ['PYTORCH_NVCC'] = str(nvcc)
(root / 'build').mkdir(exist_ok=False)
print('W1_EXTRA_CFLAGS', json.dumps(extra), flush=True)
mod = load(name='w1_attn_fp8_ext',
    sources=[str(root / 'source/kernels/w1-attn/csrc/w1_attn.cu')],
    extra_include_paths=[str(root / 'source/kernels/w1-attn/include')],
    extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo',
        '-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK', '-Xptxas=-v',
        '-Dw1=w1_fp8_candidate', '-Xcompiler=-fvisibility=hidden'] + extra,
    extra_cflags=['-fvisibility=hidden'],
    extra_ldflags=['-Wl,-Bsymbolic'],
    build_directory=str(root / 'build'), verbose=True)
p = Path(mod.__file__)
assert p.name == 'w1_attn_fp8_ext.so', f'unexpected module file {p.name} (stale build dir?)'
receipt = {'status': 'COMPILED_AND_CPU_IMPORTED', 'path': str(p), 'extra_cuda_cflags': extra,
    'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
    'gpu_execution': False, 'numerics_verified': False}
(root / 'compile-result.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt), flush=True)
