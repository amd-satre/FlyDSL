"""
Correctness test for the macro_block=256 structural variant: compare kernel
output to a pure-PyTorch MBS simulation using macro_block=256 (not 128),
at a shape where tile_k=256 (so BK % macro_block == 0 as required).
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
MACRO_BLOCK = 256


def run_torch_ref(a_q, b_q, scale_a, scale_b, m8_a, m8_b, dtype):
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


if __name__ == "__main__":
    assert get_rocm_arch() == "gfx950"
    M, N, K = 64, 128, 512  # tile_k=256 required (BK % macro_block == 0)
    tile_m, tile_n, tile_k = 64, 128, 256

    a_fp32 = torch.randn(M, K, device=device, dtype=torch.float32)
    b_fp32 = torch.randn(N, K, device=device, dtype=torch.float32)

    a_q, scale_a, m8_a, _ = oas_mbs_quant.per_1x32_f4_quant_oas_mbs(a_fp32, macro_block=MACRO_BLOCK)
    b_q, scale_b, m8_b, _ = oas_mbs_quant.per_1x32_f4_quant_oas_mbs(b_fp32, macro_block=MACRO_BLOCK)

    c_ref = run_torch_ref(a_q, b_q, scale_a, scale_b, m8_a, m8_b, torch.float32)

    scale_a_shuf = fp4_utils.shuffle_scale_w4(scale_a, 1, False)
    b_shuffled = fp4_utils.shuffle_weight_w4(b_q, 16, False, False)
    scale_b_shuf = fp4_utils.shuffle_scale_w4(scale_b, 1, False)

    M_pad32 = (M + 31) // 32 * 32
    N_pad32 = (N + 31) // 32 * 32
    mbs_a_kernel = oas_mbs_quant.shuffle_mbs_scale_w4(m8_a, M_pad32).to(device)
    mbs_b_kernel = oas_mbs_quant.shuffle_mbs_scale_w4(m8_b, N_pad32).to(device)

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

    launch_fn = compile_mxfp4_gemm_mbs(
        N=N, K=K, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, out_dtype="bf16", macro_block=MACRO_BLOCK
    )
    args = _args(c_out, a_q, b_shuffled, scale_a_shuf, scale_b_shuf, mbs_a_kernel, mbs_b_kernel)
    compiled_fn = flyc.compile(launch_fn, *args)
    compiled_fn(*args)
    torch.cuda.synchronize()

    c_out_f32 = c_out.to(torch.float32)
    mean_abs_err = (c_out_f32 - c_ref).abs().mean().item()
    ref_scale = c_ref.abs().mean().item()
    print(f"macro_block=256: mean_abs_err={mean_abs_err:.4f}  ({100 * mean_abs_err / ref_scale:.2f}% of ref mean abs)")
    assert mean_abs_err < 0.02 * ref_scale, "macro_block=256 kernel should match its own torch reference"
    print("PASS")
