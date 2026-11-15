"""Desk I DRAFT: run the INCUMBENT's own w1attn.ops.build() unchanged, with -D flags appended.

Runs inside the pinned image (CPU only), cwd /w, where /w has the same layout kernels/w1-attn/build.sh gives its
temp dir: /w/w1attn (the Pg61 fastpath package, ops.py), /w/csrc, /w/include, /w/build. ops.build() hard-codes
extra_cuda_cflags, so a -D cannot reach it without editing it. Instead this wraps torch.utils.cpp_extension.load
(ops.build() imports it at call time) and appends the space-separated flags of W1_EXTRA_CFLAGS (each must start with
-DW1_) to extra_cuda_cflags. Everything else is the incumbent's call: name w1_attn_ext, sources /w/csrc/w1_attn.cu,
include /w/include, flags -O3 -std=c++20 -lineinfo -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK -Xptxas=-v, no host
flags, no linker flags, no namespace rename, build dir /w/build/w1_attn, nvcc 13.4 if present, sm_103a.
Two changes that do not reach the binary: verbose=True (so the log shows the nvcc lines, ptxas -v register/spill
counts and the variants.cuh #pragma messages) and a receipt file.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

EXPECTED_CFLAGS = ["-O3", "-std=c++20", "-lineinfo", "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK", "-Xptxas=-v"]
root = Path(os.environ.get("W1_BUILD_ROOT", "/w"))  # /w in the container; overridable only for the desk test
extra = os.environ.get("W1_EXTRA_CFLAGS", "").split()
for f in extra:
    assert f.startswith("-DW1_"), f"refusing non-W1 flag {f!r}"
build_dir = root / "build" / "w1_attn"
assert not build_dir.exists() or not any(build_dir.iterdir()), f"{build_dir} not empty (stale build -> *_v1 rename)"
print("W1_EXTRA_CFLAGS", json.dumps(extra), flush=True)

import torch  # noqa: E402
import torch.utils.cpp_extension as ce  # noqa: E402

_orig_load = ce.load
seen: dict = {}


def load(*args, **kw):
    assert not args, "incumbent ops.build() passes keywords only"
    seen.update({k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in kw.items()})
    kw["extra_cuda_cflags"] = list(kw.get("extra_cuda_cflags") or []) + extra
    kw["verbose"] = True
    return _orig_load(**kw)


ce.load = load
sys.path.insert(0, str(root))
import w1attn.ops as ops  # noqa: E402  (the Pg61 ops.py copied into /w/w1attn)

mod = ops.build()
# the call must be the incumbent's, unmodified apart from the appended flags
assert seen.get("name") == "w1_attn_ext", seen.get("name")
assert seen.get("extra_cuda_cflags") == EXPECTED_CFLAGS, f"ops.build() flags drifted: {seen.get('extra_cuda_cflags')}"
assert seen.get("sources") == [str(root / "csrc" / "w1_attn.cu")], seen.get("sources")
assert seen.get("extra_include_paths") == [str(root / "include")], seen.get("extra_include_paths")
assert "extra_cflags" not in seen and "extra_ldflags" not in seen, "incumbent passes no host/linker flags"
assert seen.get("build_directory") == str(build_dir), seen.get("build_directory")
p = Path(mod.__file__)
assert p == build_dir / "w1_attn_ext.so", f"unexpected module file {p} (stale build dir?)"
receipt = {
    "status": "COMPILED_AND_CPU_IMPORTED", "path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
    "extra_cuda_cflags_appended": extra, "load_kwargs_from_ops_build": {k: v for k, v in seen.items()},
    "PYTORCH_NVCC": os.environ.get("PYTORCH_NVCC"), "TORCH_CUDA_ARCH_LIST": os.environ.get("TORCH_CUDA_ARCH_LIST"),
    "CUDA_HOME_env": os.environ.get("CUDA_HOME"), "torch": torch.__version__,
    "exports": sorted(n for n in dir(mod) if not n.startswith("_")),
    "w1_attn_decode_doc": (mod.w1_attn_decode.__doc__ or "").strip(),
    "gpu_execution": False, "numerics_verified": False,
}
(root / "compile-result.json").write_text(json.dumps(receipt, indent=2, default=str) + "\n")
print(json.dumps(receipt, default=str), flush=True)
