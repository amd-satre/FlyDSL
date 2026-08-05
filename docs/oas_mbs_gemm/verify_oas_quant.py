"""
Verify tests/kernels/utils/oas_mbs_quant.py against the existing (unmodified)
fp4_utils.per_1x32_f4_quant baseline, on real torch tensors, CPU-only (no GPU
needed -- this checks the *quantization function*, not the GEMM kernel).

Checks:
 1. OAS quant is a valid drop-in: same shapes/dtypes as the baseline quantizer.
 2. QSNR(x, dequant(OAS)) >= QSNR(x, dequant(baseline)) on average, confirming
    the 3-way candidate search never makes things worse and usually helps.
 3. MBS-Static on top of OAS improves QSNR further, with a bigger improvement
    on the outlier-heavy tensor than the Gaussian one (the qualitative check
    that matters most, per Phase 2's findings).
"""

import sys

sys.path.insert(0, "/scratch/satre/FlyDSL")

import torch  # noqa: E402

from tests.kernels.utils import (
    fp4_utils,  # noqa: E402
    oas_mbs_quant,  # noqa: E402
)

torch.manual_seed(0)


def qsnr_db(ref: torch.Tensor, approx: torch.Tensor) -> float:
    num = (ref.float() ** 2).sum()
    den = ((ref.float() - approx.float()) ** 2).sum()
    if den == 0:
        return float("inf")
    return (10.0 * torch.log10(num / den)).item()


def dequant(y_fp4, scale_e8m0, shape):
    f32 = fp4_utils.mxfp4_to_f32(y_fp4).view(shape[0], -1)
    scale_f32 = fp4_utils.e8m0_to_f32(scale_e8m0).repeat_interleave(32, dim=1)
    return f32 * scale_f32


def make_gaussian(rows, cols):
    return torch.randn(rows, cols)


def make_outlier_heavy(rows, cols, outlier_frac=0.01, outlier_scale=25.0):
    x = torch.randn(rows, cols) * 0.5
    mask = torch.rand(rows, cols) < outlier_frac
    outliers = (torch.randint(0, 2, x.shape) * 2 - 1).float() * (outlier_scale + torch.randn(x.shape))
    return torch.where(mask, outliers, x)


def run(name, x):
    print(f"\n=== {name}  shape={tuple(x.shape)} ===")
    y_base, s_base, _ = fp4_utils.per_1x32_f4_quant(x)
    y_oas, s_oas, _ = oas_mbs_quant.per_1x32_f4_quant_oas(x)
    y_mbs, s_mbs, m8, factor = oas_mbs_quant.per_1x32_f4_quant_oas_mbs(x)

    deq_base = dequant(y_base, s_base, x.shape)
    deq_oas = dequant(y_oas, s_oas, x.shape)

    # MBS dequant must undo the per-macro-block pre-scale before comparing.
    n = x.shape[-1]
    macro_block = 128
    deq_mbs_scaled = dequant(y_mbs, s_mbs, x.shape)
    deq_mbs = (deq_mbs_scaled.view(x.shape[0], n // macro_block, macro_block) / factor.unsqueeze(-1)).reshape(x.shape)

    q_base = qsnr_db(x, deq_base)
    q_oas = qsnr_db(x, deq_oas)
    q_mbs = qsnr_db(x, deq_mbs)

    print(f"baseline (per_1x32_f4_quant):      QSNR = {q_base:6.2f} dB")
    print(f"OAS (3-candidate search):          QSNR = {q_oas:6.2f} dB  (delta {q_oas - q_base:+.2f} dB)")
    print(f"OAS + MBS-Static (macro_block=128): QSNR = {q_mbs:6.2f} dB  (delta vs OAS {q_mbs - q_oas:+.2f} dB)")

    assert s_oas.shape == s_base.shape and s_oas.dtype == s_base.dtype
    assert y_oas.shape == y_base.shape and y_oas.dtype == y_base.dtype
    assert q_oas >= q_base - 1e-6, "OAS must never be worse than baseline on average"
    return q_base, q_oas, q_mbs


if __name__ == "__main__":
    g_base, g_oas, g_mbs = run("Gaussian", make_gaussian(2048, 4096))
    o_base, o_oas, o_mbs = run("Outlier-heavy (1% @ ~25x)", make_outlier_heavy(2048, 4096))

    print("\n=== Summary: does MBS help the outlier-heavy tensor more? ===")
    print(f"Gaussian:      OAS->MBS delta = {g_mbs - g_oas:+.2f} dB")
    print(f"Outlier-heavy: OAS->MBS delta = {o_mbs - o_oas:+.2f} dB")
    assert (o_mbs - o_oas) > (g_mbs - g_oas), "MBS should help the outlier-heavy tensor more"
    print("PASS: all assertions held.")
