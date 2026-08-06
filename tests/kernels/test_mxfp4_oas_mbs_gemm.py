#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Correctness tests for the OAS+MBS MXFP4 GEMM kernel
(kernels/mxfp4_preshuffle_mbs.py), per docs/oas_mbs_gemm/PHASE3_MBS_KERNEL_DESIGN.md.

Mirrors tests/kernels/test_preshuffle_gemm.py's structure. Checks kernel output
against a pure-PyTorch simulation of the exact per-128-K-macro-block
accumulate/scale/merge order the kernel implements (not directly against BF16
truth -- that's a separate accuracy comparison, see
docs/oas_mbs_gemm/verify_mbs_gemm_gpu.py), so a kernel-authoring bug can't hide
behind quantization-approximation error.
"""

import logging
import os
import sys

import pytest
import torch

import flydsl.compiler as flyc

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.mxfp4_preshuffle_mbs import compile_mxfp4_gemm_mbs  # noqa: E402
from tests.kernels.utils import fp4_utils  # noqa: E402
from tests.kernels.utils import oas_mbs_quant  # noqa: E402

logging.basicConfig(level=logging.INFO)

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

def _run_torch_mbs_ref(a_q, b_q, scale_a, scale_b, m8_a, m8_b, dtype, macro_block):
    x = fp4_utils.mxfp4_to_f32(a_q) * fp4_utils.e8m0_to_f32(scale_a[: a_q.shape[0]].repeat_interleave(32, dim=1))
    w = fp4_utils.mxfp4_to_f32(b_q) * fp4_utils.e8m0_to_f32(scale_b[: b_q.shape[0]].repeat_interleave(32, dim=1))
    M, K = x.shape
    N, _ = w.shape
    factor_a = 1.0 + m8_a.float() / 256.0
    factor_b = 1.0 + m8_b.float() / 256.0
    x = x.reshape(M, K // macro_block, macro_block)
    w = w.reshape(N, K // macro_block, macro_block)
    acc = torch.zeros(M, N, device=x.device, dtype=torch.float32)
    for kb in range(K // macro_block):
        local = torch.mm(x[:, kb, :], w[:, kb, :].T)
        sigma = (1.0 / factor_a[:, kb]).unsqueeze(1) * (1.0 / factor_b[:, kb]).unsqueeze(0)
        acc += local * sigma
    return acc.to(dtype)


@pytest.mark.parametrize("out_dtype", ["bf16", "fp16"])
@pytest.mark.parametrize(
    "M, N, K, tile_m, tile_n, tile_k, macro_block",
    [
        (64, 128, 256, 64, 128, 256, 128),  # smallest: K_TILES=1, k_halves=2
        (64, 8192, 8192, 64, 128, 128, 128),  # production scale: k_halves=1
        pytest.param(256, 8192, 8192, 64, 128, 256, 128, marks=pytest.mark.large_shape),  # k_halves=2, multi K-tile
        # macro_block=256 structural variant (docs/oas_mbs_gemm/PHASE4_STRUCTURAL_REWRITE.md):
        # groups 2 native 128-K MFMA calls under one coarser correction. Requires
        # tile_k % macro_block == 0, so tile_k=256 here (n_group=2, n_groups=1/tile).
        (64, 128, 512, 64, 128, 256, 256),  # smallest for macro_block=256: n_groups=1 per K-tile
        pytest.param(256, 8192, 8192, 64, 128, 256, 256, marks=pytest.mark.large_shape),
    ],
)
def test_mxfp4_oas_mbs_gemm(out_dtype, M, N, K, tile_m, tile_n, tile_k, macro_block):
    """OAS+MBS MXFP4 GEMM matches the pure-PyTorch MBS simulation at
    bf16-level tolerance, on gfx950, at both macro_block=128 (default) and
    macro_block=256 (structural rewrite, see PHASE4_STRUCTURAL_REWRITE.md)."""
    if get_rocm_arch() != "gfx950":
        pytest.skip(f"MXFP4 OAS+MBS GEMM requires gfx950, got {get_rocm_arch()}")

    torch.manual_seed(0)
    device = torch.device("cuda")
    M32, N32 = (M + 31) // 32 * 32, (N + 31) // 32 * 32

    a_fp32 = torch.zeros(M32, K, device=device, dtype=torch.float32)
    b_fp32 = torch.zeros(N32, K, device=device, dtype=torch.float32)
    a_fp32[:M] = torch.randn(M, K, device=device)
    b_fp32[:N] = torch.randn(N, K, device=device)

    a_q, scale_a, m8_a, _ = oas_mbs_quant.per_1x32_f4_quant_oas_mbs(a_fp32, macro_block=macro_block)
    a_q = a_q[:M]
    b_q, scale_b, m8_b, _ = oas_mbs_quant.per_1x32_f4_quant_oas_mbs(b_fp32, macro_block=macro_block)
    b_q = b_q[:N]

    c_ref = _run_torch_mbs_ref(a_q, b_q, scale_a, scale_b, m8_a[:M], m8_b[:N], torch.float32, macro_block)

    b_shuffled = fp4_utils.shuffle_weight_w4(b_q, 16, False, False)
    scale_a_shuf = fp4_utils.shuffle_scale_w4(scale_a, 1, False)
    scale_b_shuf = fp4_utils.shuffle_scale_w4(scale_b, 1, False)
    mbs_a_k = oas_mbs_quant.shuffle_mbs_scale_w4(m8_a, M32).to(device)
    mbs_b_k = oas_mbs_quant.shuffle_mbs_scale_w4(m8_b, N32).to(device)

    torch_out_dtype = torch.bfloat16 if out_dtype == "bf16" else torch.float16
    c_out = torch.zeros((M, N), dtype=torch_out_dtype, device=device)
    dummy_bias = torch.empty(0, dtype=torch_out_dtype, device=device)

    def _to_bytes(t):
        return t if t.dtype in (torch.uint8, torch.int8) else t.view(torch.uint8)

    def _args(c, a, b, sa, sb, ma, mb):
        return (
            c.contiguous().view(-1),
            _to_bytes(a).contiguous().view(-1),
            _to_bytes(b).contiguous().view(-1),
            _to_bytes(sa).contiguous().view(-1),
            _to_bytes(sb).contiguous().view(-1),
            _to_bytes(ma).contiguous().view(-1),
            _to_bytes(mb).contiguous().view(-1),
            dummy_bias,
            M,
            N,
            torch.cuda.current_stream(),
        )

    launch_fn = compile_mxfp4_gemm_mbs(
        N=N, K=K, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, out_dtype=out_dtype, macro_block=macro_block
    )
    args = _args(c_out, a_q, b_shuffled, scale_a_shuf, scale_b_shuf, mbs_a_k, mbs_b_k)
    compiled_fn = flyc.compile(launch_fn, *args)
    compiled_fn(*args)
    torch.cuda.synchronize()

    c_out_f32 = c_out.to(torch.float32)
    mean_abs_err = (c_out_f32 - c_ref).abs().mean().item()
    ref_scale = c_ref.abs().mean().item()
    assert mean_abs_err < 0.02 * ref_scale, (
        f"MBS kernel output diverges from torch MBS reference: "
        f"mean_abs_err={mean_abs_err:.4f} ({100 * mean_abs_err / ref_scale:.2f}% of ref mean abs)"
    )
