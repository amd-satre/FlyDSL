"""
QSNR accuracy comparison across formats (vs true BF16 matmul), on Gaussian
and outlier-heavy inputs, M=256/N=8192/K=8192, gfx950.
"""
import sys

sys.path.insert(0, ".")

import torch  # noqa: E402

import flydsl.compiler as flyc  # noqa: E402
from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.mxfp4_preshuffle import compile_mxfp4_gemm, compile_mxfp6_gemm  # noqa: E402
from kernels.mxfp4_preshuffle_mbs import compile_mxfp4_gemm_mbs  # noqa: E402
from kernels.preshuffle_gemm import compile_preshuffle_gemm  # noqa: E402
from tests.kernels.utils import fp4_utils, oas_mbs_quant  # noqa: E402
from tests.utils import pertoken_quant, shuffle_weight  # noqa: E402

device = torch.device("cuda")
ARCH = str(get_rocm_arch())
DTYPE_FP8 = torch.float8_e4m3fn if "gfx95" in ARCH else torch.float8_e4m3fnuz


def qsnr_db(ref, approx):
    num = (ref.float() ** 2).sum()
    den = ((ref.float() - approx.float()) ** 2).sum()
    return (10.0 * torch.log10(num / den)).item() if den > 0 else float("inf")


def make_outlier_heavy(rows, cols, outlier_frac=0.01, outlier_scale=25.0):
    x = torch.randn(rows, cols) * 0.5
    mask = torch.rand(rows, cols) < outlier_frac
    outliers = (torch.randint(0, 2, x.shape) * 2 - 1).float() * (outlier_scale + torch.randn(x.shape))
    return torch.where(mask, outliers, x)


def _tb(t):
    if "float8" in str(t.dtype):
        return t.view(torch.int8)
    return t if t.dtype in (torch.uint8, torch.int8) else t.view(torch.uint8)


def run_preshuffle(a_fp32, b_fp32, M, N, K, in_dtype, tile=(64, 256, 256)):
    tm, tn, tk = tile
    if in_dtype in ("fp16", "bf16"):
        torch_dtype = torch.float16 if in_dtype == "fp16" else torch.bfloat16
        a_q, b_q = a_fp32.to(torch_dtype), b_fp32.to(torch_dtype)
        sa_flat = torch.empty((0,), device=device, dtype=torch.float32)
        sb_flat = torch.empty((0,), device=device, dtype=torch.float32)
    else:
        quant_dtype = torch.int8 if in_dtype == "int8" else DTYPE_FP8
        a_q, scale_a = pertoken_quant(a_fp32, quant_dtype=quant_dtype)
        b_q, scale_b = pertoken_quant(b_fp32, quant_dtype=quant_dtype)
        sa_flat = scale_a.contiguous().view(-1)
        sb_flat = scale_b.contiguous().view(-1)
    a_q, b_q = a_q.contiguous(), b_q.contiguous()
    b_shuf = shuffle_weight(b_q, layout=(16, 16))
    c = torch.zeros((M, N), dtype=torch.bfloat16, device=device)
    bias = torch.empty(0, dtype=torch.bfloat16, device=device)
    args = (c.view(-1), _tb(a_q).contiguous().view(-1), _tb(b_shuf).contiguous().view(-1),
            sa_flat, sb_flat, bias, M, N, torch.cuda.current_stream())
    launch = compile_preshuffle_gemm(N=N, K=K, tile_m=tm, tile_n=tn, tile_k=tk, in_dtype=in_dtype, out_dtype="bf16")
    compiled = flyc.compile(launch, *args)
    compiled(*args)
    torch.cuda.synchronize()
    return c.float()


def run_w4a4(a_fp32, b_fp32, M, N, K, tile=(64, 128, 256)):
    tm, tn, tk = tile
    M32, N32 = (M + 31) // 32 * 32, (N + 31) // 32 * 32
    a_pad = torch.zeros(M32, K, device=device)
    a_pad[:M] = a_fp32
    b_pad = torch.zeros(N32, K, device=device)
    b_pad[:N] = b_fp32
    a_q, sa, _ = fp4_utils.per_1x32_f4_quant(a_pad)
    a_q = a_q[:M]
    b_q, sb, _ = fp4_utils.per_1x32_f4_quant(b_pad)
    b_q = b_q[:N]
    b_shuf = fp4_utils.shuffle_weight_w4(b_q, 16, False, False)
    sa_shuf = fp4_utils.shuffle_scale_w4(sa, 1, False)
    sb_shuf = fp4_utils.shuffle_scale_w4(sb, 1, False)
    c = torch.zeros((M, N), dtype=torch.bfloat16, device=device)
    bias = torch.empty(0, dtype=torch.bfloat16, device=device)
    args = (c.view(-1), _tb(a_q).contiguous().view(-1), _tb(b_shuf).contiguous().view(-1),
            _tb(sa_shuf).contiguous().view(-1), _tb(sb_shuf).contiguous().view(-1),
            bias, M, N, torch.cuda.current_stream())
    launch = compile_mxfp4_gemm(N=N, K=K, tile_m=tm, tile_n=tn, tile_k=tk, out_dtype="bf16")
    compiled = flyc.compile(launch, *args)
    compiled(*args)
    torch.cuda.synchronize()
    return c.float()


def run_w4a6(a_fp32, b_fp32, M, N, K):
    M32, N32 = (M + 31) // 32 * 32, (N + 31) // 32 * 32
    a_pad = torch.zeros(M32, K, device=device)
    a_pad[:M] = a_fp32
    b_pad = torch.zeros(N32, K, device=device)
    b_pad[:N] = b_fp32
    a_fp6, sa, _ = fp4_utils.per_1x32_f6_quant(a_pad)
    a_fp6 = a_fp6[:M]
    b_q, sb, _ = fp4_utils.per_1x32_f4_quant(b_pad)
    b_q = b_q[:N]
    b_shuf = fp4_utils.shuffle_weight_w4(b_q, 16, False, False)
    sa_shuf = fp4_utils.shuffle_scale_w4(sa, 1, False)
    sb_shuf = fp4_utils.shuffle_scale_w4(sb, 1, False)
    c = torch.zeros((M, N), dtype=torch.bfloat16, device=device)
    bias = torch.empty(0, dtype=torch.bfloat16, device=device)
    args = (c.view(-1), _tb(a_fp6).contiguous().view(-1), _tb(b_shuf).contiguous().view(-1),
            _tb(sa_shuf).contiguous().view(-1), _tb(sb_shuf).contiguous().view(-1),
            bias, M, N, torch.cuda.current_stream())
    launch = compile_mxfp6_gemm(N=N, K=K, M_hint=M, out_dtype="bf16")
    compiled = flyc.compile(launch, *args)
    compiled(*args)
    torch.cuda.synchronize()
    return c.float()


def run_mbs(a_fp32, b_fp32, M, N, K, macro_block, mbs_on_a, mbs_on_b, tile=(64, 128, 256)):
    tm, tn, tk = tile
    M32, N32 = (M + 31) // 32 * 32, (N + 31) // 32 * 32
    a_pad = torch.zeros(M32, K, device=device)
    a_pad[:M] = a_fp32
    b_pad = torch.zeros(N32, K, device=device)
    b_pad[:N] = b_fp32

    def quant(x, use_mbs):
        if use_mbs:
            q, sc, m8, _ = oas_mbs_quant.per_1x32_f4_quant_oas_mbs(x, macro_block=macro_block)
            return q, sc, m8
        q, sc, _ = oas_mbs_quant.per_1x32_f4_quant_oas(x)
        m8 = torch.zeros(x.shape[0], x.shape[1] // macro_block, dtype=torch.uint8, device=x.device)
        return q, sc, m8

    a_q, sa, m8a = quant(a_pad, mbs_on_a)
    a_q = a_q[:M]
    b_q, sb, m8b = quant(b_pad, mbs_on_b)
    b_q = b_q[:N]
    b_shuf = fp4_utils.shuffle_weight_w4(b_q, 16, False, False)
    sa_shuf = fp4_utils.shuffle_scale_w4(sa, 1, False)
    sb_shuf = fp4_utils.shuffle_scale_w4(sb, 1, False)
    mbsa = oas_mbs_quant.shuffle_mbs_scale_w4(m8a, M32).to(device)
    mbsb = oas_mbs_quant.shuffle_mbs_scale_w4(m8b, N32).to(device)
    c = torch.zeros((M, N), dtype=torch.bfloat16, device=device)
    bias = torch.empty(0, dtype=torch.bfloat16, device=device)
    args = (c.view(-1), _tb(a_q).contiguous().view(-1), _tb(b_shuf).contiguous().view(-1),
            _tb(sa_shuf).contiguous().view(-1), _tb(sb_shuf).contiguous().view(-1),
            _tb(mbsa).contiguous().view(-1), _tb(mbsb).contiguous().view(-1),
            bias, M, N, torch.cuda.current_stream())
    launch = compile_mxfp4_gemm_mbs(N=N, K=K, tile_m=tm, tile_n=tn, tile_k=tk, out_dtype="bf16",
                                     macro_block=macro_block, mbs_on_a=mbs_on_a, mbs_on_b=mbs_on_b)
    compiled = flyc.compile(launch, *args)
    compiled(*args)
    torch.cuda.synchronize()
    return c.float()


if __name__ == "__main__":
    assert get_rocm_arch() == "gfx950"
    torch.manual_seed(0)
    M, N, K = 256, 8192, 8192

    for name, a_maker in [("Gaussian A", lambda: torch.randn(M, K, device=device)),
                           ("Outlier-heavy A", lambda: make_outlier_heavy(M, K).to(device))]:
        print(f"\n=== {name} (M={M}, N={N}, K={K}) ===")
        a_fp32 = a_maker()
        b_fp32 = torch.randn(N, K, device=device)
        bf16_ref = torch.mm(a_fp32, b_fp32.T).to(torch.bfloat16).float()

        formats = {
            "fp8 (W8A8)": lambda: run_preshuffle(a_fp32, b_fp32, M, N, K, "fp8"),
            "int8 (W8A8)": lambda: run_preshuffle(a_fp32, b_fp32, M, N, K, "int8"),
            "w4a4 (MXFP4)": lambda: run_w4a4(a_fp32, b_fp32, M, N, K),
            "w4a6 (MXFP6xMXFP4)": lambda: run_w4a6(a_fp32, b_fp32, M, N, K),
            "mbs_both_mb128": lambda: run_mbs(a_fp32, b_fp32, M, N, K, 128, True, True),
            "mbs_bonly_mb128": lambda: run_mbs(a_fp32, b_fp32, M, N, K, 128, False, True),
            "mbs_bonly_mb256": lambda: run_mbs(a_fp32, b_fp32, M, N, K, 256, False, True),
        }
        for label, fn in formats.items():
            try:
                out = fn()
                q = qsnr_db(bf16_ref, out)
                print(f"  {label:22s}: QSNR = {q:6.2f} dB")
            except Exception as e:  # noqa: BLE001
                print(f"  {label:22s}: ERROR {str(e)[:80]}")

    print("\nDONE.")
