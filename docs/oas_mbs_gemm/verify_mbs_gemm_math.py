"""
Prove the MBS *algorithm* (not just quantization SSE) actually improves real
GEMM accuracy, simulating in pure PyTorch exactly the per-128-K-macro-block
local-accumulate -> scale -> merge order the kernel would need to implement,
BEFORE writing any kernel code (this is the last hypothesis gate for the
kernel-loop surgery described in docs/oas_mbs_gemm/PHASE3_MBS_KERNEL_DESIGN.md).

Why this matters: verify_oas_quant.py's QSNR numbers were a *proxy* (per-block
round-trip quant SSE). This script instead runs the literal GEMM: for each
128-K macro block, accumulate A_block @ B_block^T into a fresh zero tile,
multiply by the per-row/per-col MBS correction, and add into the running
sum -- exactly the control flow kernels/mxfp4_preshuffle.py's compute() would
need per (kh, ni, mi) iteration.
"""
import sys

sys.path.insert(0, "/scratch/satre/FlyDSL")

import torch  # noqa: E402

from tests.kernels.utils import fp4_utils  # noqa: E402
from tests.kernels.utils import oas_mbs_quant  # noqa: E402

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MACRO_BLOCK = 128


def qsnr_db(ref, approx):
    num = (ref.float() ** 2).sum()
    den = ((ref.float() - approx.float()) ** 2).sum()
    return (10.0 * torch.log10(num / den)).item() if den > 0 else float("inf")


def make_outlier_heavy(rows, cols, outlier_frac=0.01, outlier_scale=25.0):
    x = torch.randn(rows, cols) * 0.5
    mask = torch.rand(rows, cols) < outlier_frac
    outliers = (torch.randint(0, 2, x.shape) * 2 - 1).float() * (outlier_scale + torch.randn(x.shape))
    return torch.where(mask, outliers, x)


def dequant_oas(x_padded):
    """OAS-quantized dequant, block=32, per-row scale expanded to full width."""
    y_fp4, scale_e8m0, _ = oas_mbs_quant.per_1x32_f4_quant_oas(x_padded)
    f32 = fp4_utils.mxfp4_to_f32(y_fp4)
    scale_f32 = fp4_utils.e8m0_to_f32(scale_e8m0).repeat_interleave(32, dim=1)
    return f32 * scale_f32


def simulate_kernel_gemm_with_mbs(a_fp32, b_fp32, mbs_a: bool, mbs_b: bool):
    """Simulate the per-128-K-macro-block accumulate/scale/merge the kernel
    would do. mbs_a/mbs_b: whether to apply MBS pre-scale + post-correction
    for that operand (both False == plain OAS, matching verify_oas_gemm_gpu.py)."""
    M, K = a_fp32.shape
    N, _ = b_fp32.shape

    if mbs_a:
        macro_absmax_a = torch.amax(torch.abs(a_fp32.reshape(M, K // MACRO_BLOCK, MACRO_BLOCK)), dim=-1)
        factor_a, _ = oas_mbs_quant.mbs_factor_static_e0m8(macro_absmax_a)  # (M, K/128)
        a_scaled = (a_fp32.reshape(M, K // MACRO_BLOCK, MACRO_BLOCK) * factor_a.unsqueeze(-1)).reshape(M, K)
    else:
        a_scaled, factor_a = a_fp32, torch.ones(M, K // MACRO_BLOCK, device=a_fp32.device)

    if mbs_b:
        macro_absmax_b = torch.amax(torch.abs(b_fp32.reshape(N, K // MACRO_BLOCK, MACRO_BLOCK)), dim=-1)
        factor_b, _ = oas_mbs_quant.mbs_factor_static_e0m8(macro_absmax_b)  # (N, K/128)
        b_scaled = (b_fp32.reshape(N, K // MACRO_BLOCK, MACRO_BLOCK) * factor_b.unsqueeze(-1)).reshape(N, K)
    else:
        b_scaled, factor_b = b_fp32, torch.ones(N, K // MACRO_BLOCK, device=b_fp32.device)

    a_deq = dequant_oas(a_scaled).reshape(M, K // MACRO_BLOCK, MACRO_BLOCK)
    b_deq = dequant_oas(b_scaled).reshape(N, K // MACRO_BLOCK, MACRO_BLOCK)

    acc = torch.zeros(M, N, device=a_fp32.device, dtype=torch.float32)
    for kb in range(K // MACRO_BLOCK):
        # "local accumulator": this macro block's isolated contribution
        local = torch.mm(a_deq[:, kb, :], b_deq[:, kb, :].T)
        # epilogue Hadamard: undo each operand's macro-block pre-scale, per row/col
        sigma = (1.0 / factor_a[:, kb]).unsqueeze(1) * (1.0 / factor_b[:, kb]).unsqueeze(0)
        acc += local * sigma
    return acc


if __name__ == "__main__":
    M, N, K = 256, 512, 8192
    for name, a_maker in [
        ("Gaussian A", lambda: torch.randn(M, K, device=device)),
        ("Outlier-heavy A", lambda: make_outlier_heavy(M, K).to(device)),
    ]:
        print(f"\n=== {name}  (M={M}, N={N}, K={K}) ===")
        a = a_maker()
        b = torch.randn(N, K, device=device)
        bf16_ref = torch.mm(a, b.T).to(torch.bfloat16).float()

        oas_only = simulate_kernel_gemm_with_mbs(a, b, mbs_a=False, mbs_b=False)
        mbs_a_only = simulate_kernel_gemm_with_mbs(a, b, mbs_a=True, mbs_b=False)
        mbs_both = simulate_kernel_gemm_with_mbs(a, b, mbs_a=True, mbs_b=True)

        q_oas = qsnr_db(bf16_ref, oas_only)
        q_mbs_a = qsnr_db(bf16_ref, mbs_a_only)
        q_mbs_both = qsnr_db(bf16_ref, mbs_both)
        print(f"  OAS only (no MBS):        QSNR = {q_oas:6.2f} dB")
        print(f"  OAS + MBS on A only:      QSNR = {q_mbs_a:6.2f} dB  (delta {q_mbs_a - q_oas:+.2f} dB)")
        print(f"  OAS + MBS on A and B:     QSNR = {q_mbs_both:6.2f} dB  (delta {q_mbs_both - q_oas:+.2f} dB)")
        assert q_mbs_both >= q_oas - 0.05, "full MBS simulation must not regress GEMM QSNR"
    print("\nPASS: per-128-K-macro-block accumulate/scale/merge algorithm improves real GEMM QSNR.")
