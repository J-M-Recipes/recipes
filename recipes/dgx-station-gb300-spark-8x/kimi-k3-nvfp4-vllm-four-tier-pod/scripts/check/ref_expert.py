#!/usr/bin/env python3
"""Pure-PyTorch fp32 reference for one Kimi-K3 NVFP4 routed expert.

Math is the checkpoint dequant plus SituAndMul, not the TRT-LLM cubin
layout. See SEMANTICS.md. No activation quantization, no g1_scale.
"""

from __future__ import annotations

import json
import os
import struct
from typing import Mapping

import torch

# E2M1 magnitudes for nibble low-3 bits. Sign is bit 3.
# vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py
E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

HIDDEN = 3584
INTERMEDIATE = 3072
BLOCK = 16
SITU_BETA = 4.0
SITU_LINEAR_BETA = 25.0

_E2M1_TABLE = torch.tensor(E2M1_MAGNITUDES, dtype=torch.float32)


def unpack_fp4_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 packed FP4 e2m1 to fp32.

    Last dimension is K/2. Low nibble is element 2k, high nibble is 2k+1.
    """
    if packed.dtype != torch.uint8:
        raise TypeError(f"packed FP4 must be uint8, got {packed.dtype}")
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    nibbles = torch.stack((low, high), dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    table = _E2M1_TABLE.to(device=nibbles.device)
    mag = table[(nibbles & 0x07).to(torch.long)]
    sign = (nibbles & 0x08) != 0
    return torch.where(sign, -mag, mag)


def dequantize_nvfp4(
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    global_scale: torch.Tensor | float,
    block_size: int = BLOCK,
) -> torch.Tensor:
    """W = e2m1 * e4m3_block_scale * weight_scale_2.

    Block runs along the last dimension (the GEMM K / input axis of the
    stored [out, in] matrix). Scales are linear, not swizzled.
    ``global_scale`` is the checkpoint ``weight_scale_2`` (ModelOpt amax/2688),
    multiplied, not inverted.
    """
    values = unpack_fp4_e2m1(packed)
    if values.shape[-1] % block_size != 0:
        raise ValueError(f"K={values.shape[-1]} is not a multiple of block {block_size}")
    n_blocks = values.shape[-1] // block_size
    if block_scale.shape[-1] != n_blocks:
        raise ValueError(
            f"block scale last dim {block_scale.shape[-1]} != K/{block_size}={n_blocks}"
        )
    if block_scale.dtype == torch.float8_e4m3fn:
        scale = block_scale.to(torch.float32)
    elif block_scale.dtype == torch.uint8:
        scale = block_scale.view(torch.float8_e4m3fn).to(torch.float32)
    else:
        scale = block_scale.to(torch.float32)
    gscale = torch.as_tensor(global_scale, dtype=torch.float32, device=values.device)
    grouped = values.reshape(*values.shape[:-1], n_blocks, block_size)
    # Broadcast a 0-d or 1-element global scale across the matrix.
    while gscale.ndim < grouped.ndim:
        gscale = gscale.unsqueeze(-1)
    out = grouped * scale.unsqueeze(-1) * gscale
    return out.reshape(*values.shape)


def silu_glu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SwiGLU stand-in: silu(gate) * up, computed in fp32.

    silu(g) = g * sigmoid(g). Used only to measure the B12x-vs-reference
    numeric envelope. SiTU remains the default expert activation.
    """
    gate_f = gate if gate.dtype == torch.float32 else gate.float()
    up_f = up if up.dtype == torch.float32 else up.float()
    return torch.nn.functional.silu(gate_f) * up_f


def situ(gate: torch.Tensor, up: torch.Tensor, beta: float = SITU_BETA, linear_beta: float | None = SITU_LINEAR_BETA) -> torch.Tensor:
    """SiTU-GLU. gate/up are the GEMM1 outputs, already in fp32.

    gate_out = beta * tanh(gate / beta) * sigmoid(gate)
    up_out   = linear_beta * tanh(up / linear_beta)   if linear_beta is set
    out      = gate_out * up_out
    """
    gate_f = gate if gate.dtype == torch.float32 else gate.float()
    up_f = up if up.dtype == torch.float32 else up.float()
    if beta <= 0:
        raise ValueError(f"SiTU beta must be > 0, got {beta}")
    gate_out = beta * torch.tanh(gate_f / beta) * torch.sigmoid(gate_f)
    if linear_beta is None:
        up_out = up_f
    else:
        if linear_beta <= 0:
            raise ValueError(f"SiTU linear_beta must be > 0 or None, got {linear_beta}")
        up_out = linear_beta * torch.tanh(up_f / linear_beta)
    return gate_out * up_out


def expert_forward(
    x: torch.Tensor,
    w1: torch.Tensor,
    w3: torch.Tensor,
    w2: torch.Tensor,
    *,
    activation: str = "situ",
    beta: float = SITU_BETA,
    linear_beta: float | None = SITU_LINEAR_BETA,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """y = W2 @ act(W1 @ x, W3 @ x), weights stored [out, in].

    x is [n, 3584] (bf16 or fp32). Compute is fp32. Default return is fp32.
    Pass out_dtype=x.dtype to match vLLM's cast-back (bf16 in production).
    w1 is the gate projection, w3 is the up projection. No router weight.
    activation is "situ" (default) or "silu".
    """
    if x.ndim != 2 or x.shape[-1] != w1.shape[-1]:
        raise ValueError(f"x must be [n, {w1.shape[-1]}], got {tuple(x.shape)}")
    x32 = x.float()
    gate = x32.matmul(w1.float().transpose(0, 1))
    up = x32.matmul(w3.float().transpose(0, 1))
    if activation == "situ":
        hidden = situ(gate, up, beta=beta, linear_beta=linear_beta)
    elif activation == "silu":
        hidden = silu_glu(gate, up)
    else:
        raise ValueError(f"unsupported activation {activation!r}")
    y = hidden.matmul(w2.float().transpose(0, 1))
    if out_dtype is None:
        return y
    return y.to(out_dtype)


class Nvfp4Expert:
    """One dequantized routed expert. Weights are fp32 [out, in]."""

    def __init__(
        self,
        w1: torch.Tensor,
        w1_scale: torch.Tensor,
        w1_scale_2: torch.Tensor,
        w3: torch.Tensor,
        w3_scale: torch.Tensor,
        w3_scale_2: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
        w2_scale_2: torch.Tensor,
        *,
        beta: float = SITU_BETA,
        linear_beta: float | None = SITU_LINEAR_BETA,
        activation: str = "situ",
    ) -> None:
        if activation not in ("situ", "silu"):
            raise ValueError(f"unsupported activation {activation!r}")
        self.activation = activation
        self.beta = beta
        self.linear_beta = linear_beta
        self.w1_scale_2 = w1_scale_2.detach().float().reshape(())
        self.w3_scale_2 = w3_scale_2.detach().float().reshape(())
        self.w2_scale_2 = w2_scale_2.detach().float().reshape(())
        self.w1 = dequantize_nvfp4(w1, w1_scale, self.w1_scale_2)
        self.w3 = dequantize_nvfp4(w3, w3_scale, self.w3_scale_2)
        self.w2 = dequantize_nvfp4(w2, w2_scale, self.w2_scale_2)
        self._check_shapes()

    def _check_shapes(self) -> None:
        if self.w1.shape != (INTERMEDIATE, HIDDEN):
            raise ValueError(f"w1 {tuple(self.w1.shape)} != {(INTERMEDIATE, HIDDEN)}")
        if self.w3.shape != (INTERMEDIATE, HIDDEN):
            raise ValueError(f"w3 {tuple(self.w3.shape)} != {(INTERMEDIATE, HIDDEN)}")
        if self.w2.shape != (HIDDEN, INTERMEDIATE):
            raise ValueError(f"w2 {tuple(self.w2.shape)} != {(HIDDEN, INTERMEDIATE)}")

    def __call__(self, x: torch.Tensor, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        return expert_forward(
            x,
            self.w1,
            self.w3,
            self.w2,
            activation=self.activation,
            beta=self.beta,
            linear_beta=self.linear_beta,
            out_dtype=out_dtype,
        )


def fuse_w13(w1: torch.Tensor, w3: torch.Tensor) -> torch.Tensor:
    """create_weights layout: [2*I, K/2] = cat(w1, w3) on the output axis.

    This is the pre-shuffle ModelOptNvFp4FusedMoE layout. FlashInfer later
    swaps it to [w3, w1]. Do not swap here.
    """
    if w1.shape != w3.shape:
        raise ValueError(f"w1/w3 shape mismatch {tuple(w1.shape)} vs {tuple(w3.shape)}")
    return torch.cat((w1, w3), dim=0)


_SAFETENSORS_DTYPE = {
    "U8": torch.uint8,
    "F32": torch.float32,
    "F8_E4M3": torch.float8_e4m3fn,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "I64": torch.int64,
    "I32": torch.int32,
}


def load_safetensors(path: str) -> dict[str, torch.Tensor]:
    """Minimal safetensors reader. CPU only."""
    with open(path, "rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
        blob = handle.read()
    out: dict[str, torch.Tensor] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        dtype = _SAFETENSORS_DTYPE[info["dtype"]]
        raw = blob[start:end]
        if dtype == torch.float8_e4m3fn:
            tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(dtype)
        else:
            tensor = torch.frombuffer(bytearray(raw), dtype=dtype)
        shape = info["shape"]
        tensor = tensor.reshape(shape).clone() if shape else tensor.reshape(()).clone()
        out[name] = tensor
    return out


def expert_from_checkpoint(
    tensors: Mapping[str, torch.Tensor],
    expert_id: int,
    *,
    prefix: str = "language_model.model.layers.1.block_sparse_moe.experts.",
    activation: str = "situ",
) -> Nvfp4Expert:
    base = f"{prefix}{expert_id}."

    def get(part: str, suffix: str) -> torch.Tensor:
        key = f"{base}{part}.{suffix}"
        if key not in tensors:
            raise KeyError(key)
        return tensors[key]

    return Nvfp4Expert(
        w1=get("w1", "weight"),
        w1_scale=get("w1", "weight_scale"),
        w1_scale_2=get("w1", "weight_scale_2"),
        w3=get("w3", "weight"),
        w3_scale=get("w3", "weight_scale"),
        w3_scale_2=get("w3", "weight_scale_2"),
        w2=get("w2", "weight"),
        w2_scale=get("w2", "weight_scale"),
        w2_scale_2=get("w2", "weight_scale_2"),
        activation=activation,
    )


def default_extract_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "layer01_experts0-7.safetensors")
