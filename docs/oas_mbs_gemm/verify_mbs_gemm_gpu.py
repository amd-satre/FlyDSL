"""
Full-scale GPU verification of the MBS kernel (kernels/mxfp4_preshuffle_mbs.py):
 1. Correctness at production shape (multi K-tile, k_halves=2 case) against the
    pure-PyTorch MBS simulation.
 2. GEMM-output QSNR vs true BF16 matmul: baseline MXFP4 vs OAS-only (existing
    unmodified kernel) vs OAS+MBS (new kernel), on Gaussian and outlier-heavy A.
"""
import sys

sys.path.insert(0, "/scratch/satre/FlyDSL")

import torch  # noqa: E402

import flydsl.compiler as flyc  # noqa: E402
from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.mxfp4_preshuffle import compile_mxfp4_gemm  # noqa: E402
from kernels.mxfp4_preshuffle_mbs import compile_mxfp4_gemm_mbs  # noqa: E402
from tests.kernels.utils import fp4_utils  # noqa: E402
from tests.kernels.utils import oas_mbs_quant  # noqa: E402

torch.manual_seed(0)
device = torch.device("cuda")
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


def shuffle_mbs_scale(m8: torch.Tensor, rows_padded: int) -> torch.Tensor:
    rows, k_macro = m8.shape
    out = torch.zeros(k_macro, rows_padded, dtype=torch.uint8, device=m8.device)
    out[:, :rows] = m8.T
    return out.contiguous()


def run_torch_mbs_ref(a_q, b_q, scale_a, scale_b, m8_a, m8_b, dtype):
    x = fp4_utils.mxfp4_to_f32(a_q) * fp4_utils.e8m0_to_f32(scale_a[: a_q.shape[0]].repeat_interleave(32, dim=1))
    w = fp4_utils.mxfp4_to_f32(b_q) * fp4_utils.e8m0_to_f32(scale_b[: b_q.shape[0]].repeat_interleave(32, dim=1))
    M, K = x.shape
    N, _ = w.shape
    factor_a = 1.0 + m8_a.float() / 256.0
    factor_b = 1.0 + m8_b.float() / 256.0
    x = x.reshape(M, K // MACRO_BLOCK, MACRO_BLOCK)
    w = w.reshape(N, K // MACRO_BLOCK, MACRO_BLOCK)
    acc = torch.zeros(M, N, device=x.device, dtype=torch.float32)
    for kb in range(K // MACRO_BLOCK):
        local = torch.mm(x[:, kb, :], w[:, kb, :].T)
        sigma = (1.0 / factor_a[:, kb]).unsqueeze(1) * (1.0 / factor_b[:, kb]).unsqueeze(0)
        acc += local * sigma
    return acc.to(dtype)


def run_baseline_or_oas_kernel(a_fp32, b_fp32, M, N, K, quant_fn):
    M32, N32 = (M + 31) // 32 * 32, (N + 31) // 32 * 32
    a_pad = torch.zeros(M32, K, device=device); a_pad[:M] = a_fp32
    b_pad = torch.zeros(N32, K, device=device); b_pad[:N] = b_fp32
    a_q, scale_a, _ = quant_fn(a_pad); a_q = a_q[:M]
    b_q, scale_b, _ = quant_fn(b_pad); b_q = b_q[:N]
    b_shuf = fp4_utils.shuffle_weight_w4(b_q, 16, False, False)
    scale_a_shuf = fp4_utils.shuffle_scale_w4(scale_a, 1, False)
    scale_b_shuf = fp4_utils.shuffle_scale_w4(scale_b, 1, False)

    c_out = torch.zeros((M, N), dtype=torch.bfloat16, device=device)
    dummy_bias = torch.empty(0, dtype=torch.bfloat16, device=device)
    _tb = lambda t: t if t.dtype in (torch.uint8, torch.int8) else t.view(torch.uint8)
    args = (c_out.view(-1), _tb(a_q).contiguous().view(-1), _tb(b_shuf).contiguous().view(-1),
            _tb(scale_a_shuf).contiguous().view(-1), _tb(scale_b_shuf).contiguous().view(-1),
            dummy_bias, M, N, torch.cuda.current_stream())
    launch_fn = compile_mxfp4_gemm(N=N, K=K, tile_m=64, tile_n=128, tile_k=128, out_dtype="bf16")
    compiled = flyc.compile(launch_fn, *args)
    compiled(*args)
    torch.cuda.synchronize()
    return c_out.to(torch.float32)


def run_mbs_kernel(a_fp32, b_fp32, M, N, K, tile_m=64, tile_n=128, tile_k=128):
    M32, N32 = (M + 31) // 32 * 32, (N + 31) // 32 * 32
    a_pad = torch.zeros(M32, K, device=device); a_pad[:M] = a_fp32
    b_pad = torch.zeros(N32, K, device=device); b_pad[:N] = b_fp32

    a_q, scale_a, m8_a, _ = oas_mbs_quant.per_1x32_f4_quant_oas_mbs(a_pad, macro_block=MACRO_BLOCK)
    a_q = a_q[:M]
    b_q, scale_b, m8_b, _ = oas_mbs_quant.per_1x32_f4_quant_oas_mbs(b_pad, macro_block=MACRO_BLOCK)
    b_q = b_q[:N]

    c_ref = run_torch_mbs_ref(a_q, b_q, scale_a, scale_b, m8_a[:M], m8_b[:N], torch.float32)

    b_shuf = fp4_utils.shuffle_weight_w4(b_q, 16, False, False)
    scale_a_shuf = fp4_utils.shuffle_scale_w4(scale_a, 1, False)
    scale_b_shuf = fp4_utils.shuffle_scale_w4(scale_b, 1, False)
    mbs_a_k = shuffle_mbs_scale(m8_a, M32).to(device)
    mbs_b_k = shuffle_mbs_scale(m8_b, N32).to(device)

    c_out = torch.zeros((M, N), dtype=torch.bfloat16, device=device)
    dummy_bias = torch.empty(0, dtype=torch.bfloat16, device=device)
    _tb = lambda t: t if t.dtype in (torch.uint8, torch.int8) else t.view(torch.uint8)
    args = (c_out.view(-1), _tb(a_q).contiguous().view(-1), _tb(b_shuf).contiguous().view(-1),
            _tb(scale_a_shuf).contiguous().view(-1), _tb(scale_b_shuf).contiguous().view(-1),
            _tb(mbs_a_k).contiguous().view(-1), _tb(mbs_b_k).contiguous().view(-1),
            dummy_bias, M, N, torch.cuda.current_stream())
    launch_fn = compile_mxfp4_gemm_mbs(N=N, K=K, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, out_dtype="bf16")
    compiled = flyc.compile(launch_fn, *args)
    compiled(*args)
    torch.cuda.synchronize()
    c_out_f32 = c_out.to(torch.float32)

    mean_abs_err = (c_out_f32 - c_ref).abs().mean().item()
    ref_scale = c_ref.abs().mean().item()
    print(f"    [correctness] mean_abs_err={mean_abs_err:.4f} ({100*mean_abs_err/ref_scale:.2f}% of ref mean abs)")
    assert mean_abs_err < 0.02 * ref_scale, "MBS kernel should match torch MBS reference at bf16-level tolerance"
    return c_out_f32


if __name__ == "__main__":
    assert get_rocm_arch() == "gfx950", f"requires gfx950, got {get_rocm_arch()}"
    M, N, K = 256, 8192, 8192  # same scale as verify_oas_gemm_gpu.py; K_TILES=64 (tile_k=128), k_halves=1

    for name, a_maker in [
        ("Gaussian A", lambda: torch.randn(M, K, device=device, dtype=torch.float32)),
        ("Outlier-heavy A", lambda: make_outlier_heavy(M, K).to(device)),
    ]:
        print(f"\n=== {name}  (M={M}, N={N}, K={K}) ===")
        a_fp32 = a_maker()
        b_fp32 = torch.randn(N, K, device=device, dtype=torch.float32)
        bf16_ref = torch.mm(a_fp32, b_fp32.T).to(torch.bfloat16).float()

        c_base = run_baseline_or_oas_kernel(a_fp32, b_fp32, M, N, K, fp4_utils.per_1x32_f4_quant)
        c_oas = run_baseline_or_oas_kernel(a_fp32, b_fp32, M, N, K, oas_mbs_quant.per_1x32_f4_quant_oas)
        c_mbs = run_mbs_kernel(a_fp32, b_fp32, M, N, K)

        q_base, q_oas, q_mbs = qsnr_db(bf16_ref, c_base), qsnr_db(bf16_ref, c_oas), qsnr_db(bf16_ref, c_mbs)
        print(f"  QSNR vs BF16 truth: baseline={q_base:.2f} dB  OAS={q_oas:.2f} dB  OAS+MBS={q_mbs:.2f} dB")
        print(f"  deltas: OAS-baseline={q_oas-q_base:+.2f} dB   MBS-OAS={q_mbs-q_oas:+.2f} dB   "
              f"MBS-baseline={q_mbs-q_base:+.2f} dB")

    print("\nDONE.")
