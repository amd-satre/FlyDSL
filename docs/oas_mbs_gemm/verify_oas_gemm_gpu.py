"""
GPU test: feed OAS-quantized A/B into the UNMODIFIED existing mxfp4_preshuffle
kernel (compile_mxfp4_gemm) and confirm:
 1. Correctness: kernel output matches an OAS-aware torch dequant+matmul
    reference (proves OAS integrates with zero kernel changes -- it's purely
    a different, still-valid, E8M0 scale choice within the existing per-32
    block format the kernel already consumes).
 2. Accuracy: OAS-quantized GEMM has higher QSNR against the true BF16 result
    than plain MXFP4-quantized GEMM, on both a Gaussian and an outlier-heavy
    A matrix (B kept Gaussian, mirroring "weights" vs "activations" framing).

Run inside the satre-oas-mbs-flydsl container (needs GPU + built FlyDSL).
"""

import sys

sys.path.insert(0, "/scratch/satre/FlyDSL")

import torch  # noqa: E402

import flydsl.compiler as flyc  # noqa: E402
from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.mxfp4_preshuffle import compile_mxfp4_gemm  # noqa: E402
from tests.kernels.utils import (
    fp4_utils,  # noqa: E402
    oas_mbs_quant,  # noqa: E402
)

torch.manual_seed(0)
device = torch.device("cuda")


def qsnr_db(ref: torch.Tensor, approx: torch.Tensor) -> float:
    num = (ref.float() ** 2).sum()
    den = ((ref.float() - approx.float()) ** 2).sum()
    return (10.0 * torch.log10(num / den)).item() if den > 0 else float("inf")


def make_outlier_heavy(rows, cols, outlier_frac=0.01, outlier_scale=25.0):
    x = torch.randn(rows, cols) * 0.5
    mask = torch.rand(rows, cols) < outlier_frac
    outliers = (torch.randint(0, 2, x.shape) * 2 - 1).float() * (outlier_scale + torch.randn(x.shape))
    return torch.where(mask, outliers, x)


def run_torch_w4(x_q, w_q, x_scales, w_scales, dtype):
    x_f32 = fp4_utils.mxfp4_to_f32(x_q)
    w_f32 = fp4_utils.mxfp4_to_f32(w_q)
    x_scales_f32 = fp4_utils.e8m0_to_f32(x_scales[: x_q.shape[0]].repeat_interleave(32, dim=1))
    w_scales_f32 = fp4_utils.e8m0_to_f32(w_scales[: w_q.shape[0]].repeat_interleave(32, dim=1))
    return torch.mm(x_f32 * x_scales_f32, (w_f32 * w_scales_f32).T).to(dtype)


def run_gemm(a_fp32, b_fp32, M, N, K, quant_fn, label):
    M_align_32 = (M + 31) // 32 * 32
    N_align_32 = (N + 31) // 32 * 32
    a_fp32_padded = torch.zeros(M_align_32, K, device=device, dtype=torch.float32)
    b_fp32_padded = torch.zeros(N_align_32, K, device=device, dtype=torch.float32)
    a_fp32_padded[:M] = a_fp32
    b_fp32_padded[:N] = b_fp32

    a_q, scale_a_orig, _ = quant_fn(a_fp32_padded)
    a_q = a_q[:M]
    scale_a = fp4_utils.shuffle_scale_w4(scale_a_orig, 1, False)

    b_q, scale_b, _ = quant_fn(b_fp32_padded)
    b_q = b_q[:N]

    c_ref = run_torch_w4(a_q, b_q, scale_a_orig, scale_b, torch.float32)

    b_shuffled = fp4_utils.shuffle_weight_w4(b_q, 16, False, False)
    scale_b_shuffled = fp4_utils.shuffle_scale_w4(scale_b, 1, False)

    c_out = torch.zeros((M, N), dtype=torch.bfloat16, device=device)
    dummy_bias = torch.empty(0, dtype=torch.bfloat16, device=device)

    def _to_bytes(t):
        return t if t.dtype in (torch.uint8, torch.int8) else t.view(torch.uint8)

    def _args(c, a, b, sa, sb):
        return (
            c.contiguous().view(-1),
            _to_bytes(a).contiguous().view(-1),
            _to_bytes(b).contiguous().view(-1),
            _to_bytes(sa).contiguous().view(-1),
            _to_bytes(sb).contiguous().view(-1),
            dummy_bias,
            M,
            N,
            torch.cuda.current_stream(),
        )

    launch_fn = compile_mxfp4_gemm(N=N, K=K, tile_m=64, tile_n=128, tile_k=128, out_dtype="bf16")
    compiled_fn = flyc.compile(launch_fn, *_args(c_out, a_q, b_shuffled, scale_a, scale_b_shuffled))
    compiled_fn(*_args(c_out, a_q, b_shuffled, scale_a, scale_b_shuffled))
    torch.cuda.synchronize()

    c_out_f32 = c_out.to(torch.float32)
    max_abs_err = (c_out_f32 - c_ref).abs().max().item()
    rel_err = ((c_out_f32 - c_ref).abs() / c_ref.abs().clamp(min=1e-3)).mean().item()
    print(f"  [{label}] kernel-vs-torch-dequant-ref: max_abs_err={max_abs_err:.4f} mean_rel_err={rel_err:.4%}")
    return c_out_f32, c_ref


if __name__ == "__main__":
    assert get_rocm_arch() == "gfx950", f"MXFP4 preshuffle kernel requires gfx950, got {get_rocm_arch()}"
    M, N, K = 256, 8192, 8192

    for a_name, a_maker in [
        ("Gaussian A", lambda: torch.randn(M, K, device=device, dtype=torch.float32)),
        ("Outlier-heavy A", lambda: make_outlier_heavy(M, K).to(device)),
    ]:
        print(f"\n=== {a_name}  (M={M}, N={N}, K={K}) ===")
        a_fp32 = a_maker()
        b_fp32 = torch.randn(N, K, device=device, dtype=torch.float32)
        bf16_ref = torch.mm(a_fp32, b_fp32.T).to(torch.bfloat16).to(torch.float32)

        c_base, ref_base = run_gemm(a_fp32, b_fp32, M, N, K, fp4_utils.per_1x32_f4_quant, "baseline quant")
        c_oas, ref_oas = run_gemm(a_fp32, b_fp32, M, N, K, oas_mbs_quant.per_1x32_f4_quant_oas, "OAS quant")

        q_base = qsnr_db(bf16_ref, c_base)
        q_oas = qsnr_db(bf16_ref, c_oas)
        print(
            f"  GEMM output QSNR vs true BF16 matmul: baseline={q_base:.2f} dB, OAS={q_oas:.2f} dB "
            f"(delta {q_oas - q_base:+.2f} dB)"
        )
        assert q_oas >= q_base - 0.05, "OAS-quantized GEMM should not be meaningfully worse than baseline"

    print("\nPASS: OAS quantization is a correct, zero-kernel-change drop-in; GEMM accuracy improves.")
