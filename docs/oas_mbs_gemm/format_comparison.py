"""
Multi-format dense GEMM comparison for DeepSeek-R1 shapes: BF16, W8A8 (fp8),
W8A8 (int8), W4A4 (MXFP4 baseline), W4A6 (MXFP6xMXFP4), and our OAS+MBS
variants (both-operand/B-only x macro_block 128/256). Each format is
individually tuned (tile search or built-in auto-tune) before benchmarking.
W4A8 (MXFP4 weight x FP8 activation) is a documented gap: no gfx950 kernel
exists in this repo for it (tests/kernels/test_preshuffle_gemm.py explicitly
skips it: "fp8-A not yet supported with MXFP4 preshuffle kernel").
"""
import sys
import time
import statistics

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

M_VALUES = [64, 1024, 3000]
NK_SHAPES = [(7168, 7168), (36864, 7168), (7168, 2048)]

MXFP4_TILE_CANDIDATES = [
    (16, 128, 256), (32, 128, 256), (64, 128, 128), (64, 128, 256),
    (64, 256, 128), (128, 128, 128), (128, 256, 128),
]
PRESHUFFLE_TILE_CANDIDATES = [
    (32, 128, 256), (48, 128, 256), (64, 128, 256), (64, 256, 256),
    (96, 256, 256), (128, 256, 256), (16, 128, 256),
]


def bench_median(fn, warmup=8, iters=30, repeats=5):
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


def _tb(t):
    if "float8" in str(t.dtype):
        return t.view(torch.int8)
    return t if t.dtype in (torch.uint8, torch.int8) else t.view(torch.uint8)


# ---- Format runners: each returns a zero-arg callable(fn) given (a,b,M,N,K,tile) ----

def make_preshuffle_runner(a_fp32, b_fp32, M, N, K, tile, in_dtype):
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
    return lambda: compiled(*args)


def make_mxfp4_runner(a_fp32, b_fp32, M, N, K, tile):
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
    return lambda: compiled(*args)


def make_mxfp6_runner(a_fp32, b_fp32, M, N, K):
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
    return lambda: compiled(*args)


def make_mbs_runner(a_fp32, b_fp32, M, N, K, tile, macro_block, mbs_on_a, mbs_on_b):
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
    return lambda: compiled(*args)


def best_over_tiles(maker, tiles, *maker_args):
    best = None
    for tile in tiles:
        try:
            fn = maker(*maker_args, tile)
            ms = bench_median(fn)
        except Exception:  # noqa: BLE001
            continue
        if best is None or ms < best[0]:
            best = (ms, tile)
    return best


if __name__ == "__main__":
    assert get_rocm_arch() == "gfx950"
    torch.manual_seed(0)

    results = []
    for N, K in NK_SHAPES:
        for M in M_VALUES:
            a = torch.randn(M, K, device=device)
            b = torch.randn(N, K, device=device)
            row = {"N": N, "K": K, "M": M}

            bf16 = best_over_tiles(
                lambda a_, b_, M_, N_, K_, t: make_preshuffle_runner(a_, b_, M_, N_, K_, t, "bf16"),
                PRESHUFFLE_TILE_CANDIDATES, a, b, M, N, K)
            fp8 = best_over_tiles(
                lambda a_, b_, M_, N_, K_, t: make_preshuffle_runner(a_, b_, M_, N_, K_, t, "fp8"),
                PRESHUFFLE_TILE_CANDIDATES, a, b, M, N, K)
            int8 = best_over_tiles(
                lambda a_, b_, M_, N_, K_, t: make_preshuffle_runner(a_, b_, M_, N_, K_, t, "int8"),
                PRESHUFFLE_TILE_CANDIDATES, a, b, M, N, K)
            w4a4 = best_over_tiles(make_mxfp4_runner, MXFP4_TILE_CANDIDATES, a, b, M, N, K)
            try:
                w4a6_fn = make_mxfp6_runner(a, b, M, N, K)
                w4a6 = (bench_median(w4a6_fn), "auto")
            except Exception as e:  # noqa: BLE001
                w4a6 = (None, f"ERR:{str(e)[:60]}")
            mbs_both_128 = best_over_tiles(
                lambda a_, b_, M_, N_, K_, t: make_mbs_runner(a_, b_, M_, N_, K_, t, 128, True, True),
                MXFP4_TILE_CANDIDATES, a, b, M, N, K)
            mbs_bonly_128 = best_over_tiles(
                lambda a_, b_, M_, N_, K_, t: make_mbs_runner(a_, b_, M_, N_, K_, t, 128, False, True),
                MXFP4_TILE_CANDIDATES, a, b, M, N, K)
            mbs_bonly_256_tiles = [t for t in MXFP4_TILE_CANDIDATES if t[2] % 256 == 0]
            mbs_bonly_256 = best_over_tiles(
                lambda a_, b_, M_, N_, K_, t: make_mbs_runner(a_, b_, M_, N_, K_, t, 256, False, True),
                mbs_bonly_256_tiles, a, b, M, N, K)

            row["bf16_ms"], row["bf16_tile"] = bf16 if bf16 else (None, None)
            row["fp8_ms"], row["fp8_tile"] = fp8 if fp8 else (None, None)
            row["int8_ms"], row["int8_tile"] = int8 if int8 else (None, None)
            row["w4a4_ms"], row["w4a4_tile"] = w4a4 if w4a4 else (None, None)
            row["w4a6_ms"], row["w4a6_tile"] = w4a6
            row["mbs_both128_ms"], row["mbs_both128_tile"] = mbs_both_128 if mbs_both_128 else (None, None)
            row["mbs_bonly128_ms"], row["mbs_bonly128_tile"] = mbs_bonly_128 if mbs_bonly_128 else (None, None)
            row["mbs_bonly256_ms"], row["mbs_bonly256_tile"] = mbs_bonly_256 if mbs_bonly_256 else (None, None)
            results.append(row)

            def fmt(ms):
                return f"{ms:.4f}" if isinstance(ms, float) else str(ms)

            print(f"N={N} K={K} M={M}: "
                  f"bf16={fmt(row['bf16_ms'])} fp8={fmt(row['fp8_ms'])} int8={fmt(row['int8_ms'])} "
                  f"w4a4={fmt(row['w4a4_ms'])} w4a6={fmt(row['w4a6_ms'])} "
                  f"mbs_both128={fmt(row['mbs_both128_ms'])} mbs_bonly128={fmt(row['mbs_bonly128_ms'])} "
                  f"mbs_bonly256={fmt(row['mbs_bonly256_ms'])}")

    import json
    with open("format_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nDONE. Results saved to format_comparison_results.json")
