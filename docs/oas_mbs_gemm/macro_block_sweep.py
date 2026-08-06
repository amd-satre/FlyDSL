import sys
import time
import statistics

sys.path.insert(0, ".")

import torch

import flydsl.compiler as flyc
from kernels.mxfp4_preshuffle import compile_mxfp4_gemm
from kernels.mxfp4_preshuffle_mbs import compile_mxfp4_gemm_mbs
from tests.kernels.utils import fp4_utils, oas_mbs_quant

device = torch.device("cuda")


def bench_median(fn, warmup=10, iters=50, repeats=7):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    medians = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        medians.append((time.perf_counter() - t0) / iters * 1e3)
    return statistics.median(medians)


def run_baseline(a, b, M, N, K, tm, tn, tk):
    M32, N32 = (M + 31) // 32 * 32, (N + 31) // 32 * 32
    a_pad = torch.zeros(M32, K, device=device)
    a_pad[:M] = a
    b_pad = torch.zeros(N32, K, device=device)
    b_pad[:N] = b
    a_q, sa, _ = fp4_utils.per_1x32_f4_quant(a_pad)
    a_q = a_q[:M]
    b_q, sb, _ = fp4_utils.per_1x32_f4_quant(b_pad)
    b_q = b_q[:N]
    b_shuf = fp4_utils.shuffle_weight_w4(b_q, 16, False, False)
    sa_shuf = fp4_utils.shuffle_scale_w4(sa, 1, False)
    sb_shuf = fp4_utils.shuffle_scale_w4(sb, 1, False)
    c = torch.zeros((M, N), dtype=torch.bfloat16, device=device)
    bias = torch.empty(0, dtype=torch.bfloat16, device=device)

    def _tb(t):
        return t if t.dtype in (torch.uint8, torch.int8) else t.view(torch.uint8)

    args = (c.view(-1), _tb(a_q).contiguous().view(-1), _tb(b_shuf).contiguous().view(-1),
            _tb(sa_shuf).contiguous().view(-1), _tb(sb_shuf).contiguous().view(-1),
            bias, M, N, torch.cuda.current_stream())
    launch = compile_mxfp4_gemm(N=N, K=K, tile_m=tm, tile_n=tn, tile_k=tk, out_dtype="bf16")
    compiled = flyc.compile(launch, *args)
    return bench_median(lambda: compiled(*args))


def run_mbs(a, b, M, N, K, tm, tn, tk, macro_block, mbs_on_a, mbs_on_b):
    M32, N32 = (M + 31) // 32 * 32, (N + 31) // 32 * 32
    a_pad = torch.zeros(M32, K, device=device)
    a_pad[:M] = a
    b_pad = torch.zeros(N32, K, device=device)
    b_pad[:N] = b

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

    def _tb(t):
        return t if t.dtype in (torch.uint8, torch.int8) else t.view(torch.uint8)

    args = (c.view(-1), _tb(a_q).contiguous().view(-1), _tb(b_shuf).contiguous().view(-1),
            _tb(sa_shuf).contiguous().view(-1), _tb(sb_shuf).contiguous().view(-1),
            _tb(mbsa).contiguous().view(-1), _tb(mbsb).contiguous().view(-1),
            bias, M, N, torch.cuda.current_stream())
    launch = compile_mxfp4_gemm_mbs(N=N, K=K, tile_m=tm, tile_n=tn, tile_k=tk, out_dtype="bf16",
                                     macro_block=macro_block, mbs_on_a=mbs_on_a, mbs_on_b=mbs_on_b)
    compiled = flyc.compile(launch, *args)
    return bench_median(lambda: compiled(*args))


if __name__ == "__main__":
    torch.manual_seed(0)
    tile = (64, 128, 256)
    shapes = [(7168, 7168, 1024), (36864, 7168, 1024), (7168, 2048, 1024)]
    macro_blocks = [32, 64, 128, 256, 512]

    for label, mbs_on_a, mbs_on_b in [("BOTH operands MBS", True, True), ("B-only MBS", False, True)]:
        print(f"=== macro_block sweep: {label} ===")
        for N, K, M in shapes:
            a = torch.randn(M, K, device=device)
            b = torch.randn(N, K, device=device)
            base = run_baseline(a, b, M, N, K, *tile)
            row = f"N={N} K={K} M={M}: baseline={base:.4f}ms  "
            for mb in macro_blocks:
                # tile_k must be a multiple of macro_block (kernel constraint);
                # use tile=(64,128,256) for mb<=256, tile_k=mb itself for mb=512
                # (the smallest valid tile_k for that macro_block).
                this_tile = tile if mb <= 256 else (tile[0], tile[1], mb)
                if K % mb != 0 or K % this_tile[2] != 0:
                    row += f"mb={mb}:N/A(shape)  "
                    continue
                try:
                    base_this = base if this_tile == tile else run_baseline(a, b, M, N, K, *this_tile)
                    t = run_mbs(a, b, M, N, K, *this_tile, mb, mbs_on_a, mbs_on_b)
                    row += f"mb={mb}:{100 * (t / base_this - 1):+.1f}%  "
                except Exception as e:  # noqa: BLE001
                    row += f"mb={mb}:ERR({str(e)[:40]})  "
            print(row)
        print()
    print("DONE.")
