"""
Smallest-shape correctness test for kernels/mxfp4_preshuffle_mbs.py, per
docs/oas_mbs_gemm/PHASE3_MBS_KERNEL_DESIGN.md's verification plan step 1:
compare kernel output to the pure-PyTorch MBS simulation (already validated
against BF16 truth in verify_mbs_gemm_math.py), NOT directly to BF16 truth,
to isolate kernel-authoring bugs from quantization-approximation error.
"""
import sys

sys.path.insert(0, "/scratch/satre/FlyDSL")

import torch  # noqa: E402

import flydsl.compiler as flyc  # noqa: E402
from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.mxfp4_preshuffle_mbs import compile_mxfp4_gemm_mbs  # noqa: E402
from tests.kernels.utils import fp4_utils  # noqa: E402
from tests.kernels.utils import oas_mbs_quant  # noqa: E402

torch.manual_seed(0)
device = torch.device("cuda")
MACRO_BLOCK = 128


def shuffle_mbs_scale(m8: torch.Tensor, rows_padded: int) -> torch.Tensor:
    """m8: [rows, K/128] uint8 -> transposed+padded [K/128, rows_padded] flat
    layout matching the kernel's expected addressing (see PHASE3_MBS_KERNEL_DESIGN.md)."""
    rows, k_macro = m8.shape
    out = torch.zeros(k_macro, rows_padded, dtype=torch.uint8, device=m8.device)
    out[:, :rows] = m8.T
    return out.contiguous()


def run_torch_ref(a_q, b_q, scale_a, scale_b, m8_a, m8_b, dtype):
    x_f32 = fp4_utils.mxfp4_to_f32(a_q)
    w_f32 = fp4_utils.mxfp4_to_f32(b_q)
    x_scales_f32 = fp4_utils.e8m0_to_f32(scale_a[: a_q.shape[0]].repeat_interleave(32, dim=1))
    w_scales_f32 = fp4_utils.e8m0_to_f32(scale_b[: b_q.shape[0]].repeat_interleave(32, dim=1))
    x = x_f32 * x_scales_f32
    w = w_f32 * w_scales_f32
    M, K = x.shape
    N, _ = w.shape
    factor_a = 1.0 + m8_a.float() / 256.0  # (M, K/128)
    factor_b = 1.0 + m8_b.float() / 256.0  # (N, K/128)
    x = x.reshape(M, K // MACRO_BLOCK, MACRO_BLOCK)
    w = w.reshape(N, K // MACRO_BLOCK, MACRO_BLOCK)
    acc = torch.zeros(M, N, device=x.device, dtype=torch.float32)
    for kb in range(K // MACRO_BLOCK):
        local = torch.mm(x[:, kb, :], w[:, kb, :].T)
        sigma = (1.0 / factor_a[:, kb]).unsqueeze(1) * (1.0 / factor_b[:, kb]).unsqueeze(0)
        acc += local * sigma
    return acc.to(dtype)


if __name__ == "__main__":
    assert get_rocm_arch() == "gfx950", f"requires gfx950, got {get_rocm_arch()}"
    M, N, K = 64, 128, 256  # smallest shape: K_TILES=1 (tile_k=256) or 2 (tile_k=128)
    tile_m, tile_n, tile_k = 64, 128, 256

    a_fp32 = torch.randn(M, K, device=device, dtype=torch.float32)
    b_fp32 = torch.randn(N, K, device=device, dtype=torch.float32)

    # Host-side OAS+MBS quantization (pre-scale by MBS factor, then OAS-quantize).
    a_q, scale_a, m8_a, factor_a = oas_mbs_quant.per_1x32_f4_quant_oas_mbs(a_fp32, macro_block=MACRO_BLOCK)
    b_q, scale_b, m8_b, factor_b = oas_mbs_quant.per_1x32_f4_quant_oas_mbs(b_fp32, macro_block=MACRO_BLOCK)

    c_ref = run_torch_ref(a_q, b_q, scale_a, scale_b, m8_a, m8_b, torch.float32)

    scale_a_shuf = fp4_utils.shuffle_scale_w4(scale_a, 1, False)
    b_shuffled = fp4_utils.shuffle_weight_w4(b_q, 16, False, False)
    scale_b_shuf = fp4_utils.shuffle_scale_w4(scale_b, 1, False)

    M_pad32 = (M + 31) // 32 * 32
    N_pad32 = (N + 31) // 32 * 32
    mbs_a_kernel = shuffle_mbs_scale(m8_a, M_pad32).to(device)
    mbs_b_kernel = shuffle_mbs_scale(m8_b, N_pad32).to(device)

    c_out = torch.zeros((M, N), dtype=torch.bfloat16, device=device)
    dummy_bias = torch.empty(0, dtype=torch.bfloat16, device=device)

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

    launch_fn = compile_mxfp4_gemm_mbs(N=N, K=K, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, out_dtype="bf16")
    args = _args(c_out, a_q, b_shuffled, scale_a_shuf, scale_b_shuf, mbs_a_kernel, mbs_b_kernel)
    compiled_fn = flyc.compile(launch_fn, *args)
    compiled_fn(*args)
    torch.cuda.synchronize()

    c_out_f32 = c_out.to(torch.float32)
    max_abs_err = (c_out_f32 - c_ref).abs().max().item()
    mean_abs_err = (c_out_f32 - c_ref).abs().mean().item()
    ref_scale = c_ref.abs().mean().item()
    print(f"max_abs_err={max_abs_err:.4f}  mean_abs_err={mean_abs_err:.4f}  ref_mean_abs={ref_scale:.4f}")
    print(f"c_out[:2,:4]=\n{c_out_f32[:2,:4]}")
    print(f"c_ref[:2,:4]=\n{c_ref[:2,:4]}")
    assert mean_abs_err < 0.15 * ref_scale, "kernel output should match the MBS torch reference within bf16 tolerance"
    print("PASS: kernel matches pure-PyTorch MBS simulation.")
