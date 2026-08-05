# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Overflow-Aware Scaling (OAS) and Macro Block Scaling (MBS) quantization,
per arXiv:2603.08713 Sec 4.2/4.3, adapted to FlyDSL's existing per-1x32 E2M1
MXFP4 quantization path in ``fp4_utils.per_1x32_f4_quant``.

Adaptation note (see docs/oas_mbs_gemm/PHASE3_NOTES.md for full derivation):
FlyDSL's existing per-32 quantizer (matching the OCP MX spec + this hardware's
native scaled-MFMA) picks the E8M0 scale as the *round-to-nearest* power of two
of ``max_abs / 4.0`` (``f32_to_e8m0``). This is a different rounding convention
than the paper's own "standard" derivation (``SF = 6/absmax`` floored to a power
of two, mapping absmax into (3,6]) -- both are valid ways to pick a power-of-two
scale, they just round differently. Rather than force-fit the paper's exact
(3, 3.5] boundary condition (which was derived for their specific floor-based
convention), OAS here is implemented as a direct 3-way candidate search: try the
nominal round-to-nearest exponent and its two neighbors, keep whichever
minimizes the block's actual quantization SSE. This is the same idea (spend a
little more scale-selection effort to reduce error versus the naive per-block
rule) applied honestly to the convention this kernel actually uses.
"""

import torch

from . import fp4_utils

fp4x2 = fp4_utils.fp4x2
fp8_e8m0 = fp4_utils.fp8_e8m0

F4E2M1_MAX = 6.0
_BLOCK = 32
_MACRO_BLOCK = 128  # matches this kernel's native 16x16x128 scaled-MFMA K-granularity


def _dequant_mxfp4_error_sse(x_block: torch.Tensor, scale_f32: torch.Tensor) -> torch.Tensor:
    """SSE of round-trip MXFP4 quant/dequant of x_block (..., 32) at the given
    per-block scale_f32 (...,)."""
    y = x_block.float() / scale_f32.unsqueeze(-1)
    y_fp4 = fp4_utils.f32_to_mxfp4(y)
    y_deq = fp4_utils.mxfp4_to_f32(y_fp4).view(y.shape) * scale_f32.unsqueeze(-1)
    return ((y_deq - x_block.float()) ** 2).sum(dim=-1)


def per_1x32_f4_quant_oas(x: torch.Tensor):
    """Drop-in replacement for ``fp4_utils.per_1x32_f4_quant`` (block=32, no
    shuffle) that applies Overflow-Aware Scaling: a 3-candidate (e-1, e, e+1)
    search over the E8M0 exponent, picking the one with lowest per-block SSE.

    Returns (y_fp4, scale_e8m0_uint8, y_dequant_preview) matching the original
    function's return signature so it's a drop-in for existing test/kernel
    plumbing (shuffle_scale_w4 etc. operate on the returned scale tensor).
    """
    shape_original = x.shape
    x2 = x.reshape(-1, shape_original[-1])
    m, n = x2.shape
    xb = x2.reshape(-1, _BLOCK)
    max_abs = torch.amax(torch.abs(xb.float()), 1)

    nominal_e8m0 = fp4_utils.f32_to_e8m0(max_abs / (2.0 ** int(torch.log2(torch.tensor(F4E2M1_MAX)).item())))
    # NOTE: e8m0 is a "minifloat" dtype whose *value* when cast to float32 is
    # 2^(biased_exp-127) -- `.to(torch.int32)` would truncate that value (~0),
    # not extract the bits. Must reinterpret via `.view(torch.uint8)` to get
    # the raw biased exponent as an integer we can add/subtract/clamp.
    nominal_exp = nominal_e8m0.view(torch.uint8).to(torch.int32)

    best_sse = None
    best_exp = nominal_exp.clone()
    for delta in (-1, 0, 1):
        cand_exp = torch.clamp(nominal_exp + delta, 0, 254).to(torch.uint8)
        cand_scale_f32 = fp4_utils.e8m0_to_f32(cand_exp)
        sse = _dequant_mxfp4_error_sse(xb, cand_scale_f32)
        if best_sse is None:
            best_sse = sse
            best_exp = cand_exp
        else:
            improve = sse < best_sse
            best_sse = torch.where(improve, sse, best_sse)
            best_exp = torch.where(improve, cand_exp, best_exp)

    scale_e8m0_biased = best_exp.to(torch.uint8)
    scale_f32 = fp4_utils.e8m0_to_f32(scale_e8m0_biased)
    y = xb.float() / scale_f32.view(-1, 1)
    y_fp4 = fp4_utils.f32_to_mxfp4(y)
    y_fp4 = y_fp4.view(*shape_original[:-1], -1)
    scale = scale_e8m0_biased.view(m, -1).view(torch.uint8)
    return y_fp4, scale.view(fp8_e8m0), y


def mbs_factor_static_e0m8(macro_absmax: torch.Tensor) -> torch.Tensor:
    """Paper Eq. 3: 8 MSBs of the mantissa of (6/macro_absmax) as an E0M8
    "MBS Factor" mantissa byte. Returns (factor, mantissa_byte) where
    factor = 1 + mantissa/256 in [1, 2)."""
    macro_absmax = torch.clamp(macro_absmax, min=1e-12)
    target = (F4E2M1_MAX / macro_absmax).float()
    bits = target.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    m8 = ((bits & 0x007F8000) >> 15).to(torch.uint8)
    factor = 1.0 + m8.float() / 256.0
    return factor, m8


def per_1x32_f4_quant_oas_mbs(x: torch.Tensor, macro_block: int = _MACRO_BLOCK):
    """OAS + MBS-Static: pre-scale each 128-wide macro block by its MBS factor
    (Eq. 3) before running the OAS-enhanced per-32 quantizer. Returns
    (y_fp4, scale_e8m0_uint8, mbs_mantissa_uint8) -- the extra mbs_mantissa
    tensor (shape [rows, K/macro_block]) is the new per-macro-block correction
    the GEMM epilogue must apply (see PHASE3_NOTES.md for the required kernel
    change; not yet wired into the production kernel -- this is the host-side
    quantization half of MBS, independently testable against a reference
    dequant/GEMM in torch before kernel work continues).
    """
    shape_original = x.shape
    x2 = x.reshape(-1, shape_original[-1])
    m, n = x2.shape
    assert n % macro_block == 0
    xm = x2.reshape(m, n // macro_block, macro_block)
    macro_absmax = torch.amax(torch.abs(xm.float()), dim=-1)
    factor, m8 = mbs_factor_static_e0m8(macro_absmax)
    x_scaled = (xm.float() * factor.unsqueeze(-1)).reshape(m, n)

    y_fp4, scale_e8m0, y = per_1x32_f4_quant_oas(x_scaled)
    return y_fp4, scale_e8m0, m8, factor


def shuffle_mbs_scale_w4(m8: torch.Tensor, rows_padded: int) -> torch.Tensor:
    """Transpose+pad the per-row MBS mantissa tensor from ``[rows, K/128]`` to
    ``[K/128, rows_padded]`` so that, for a fixed macro-block index, the 4
    consecutive rows one lane needs (see PHASE3_MBS_KERNEL_DESIGN.md) are
    contiguous in memory -- one dword (4-byte) buffer load per lane per macro
    block instead of 4 separate strided byte loads. Padding rows beyond
    ``rows`` are zero, which decodes to an MBS factor of 1.0 (no-op) for the
    ragged tail past M or N.
    """
    rows, k_macro = m8.shape
    out = torch.zeros(k_macro, rows_padded, dtype=torch.uint8, device=m8.device)
    out[:, :rows] = m8.T
    return out.contiguous()
