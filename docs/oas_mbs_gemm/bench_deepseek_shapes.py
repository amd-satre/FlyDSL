"""
Microbenchmark: MXFP4 baseline (production, tuned) vs OAS+MBS (correctness-
first, untuned) across DeepSeek-R1-shaped GEMMs.

Shapes (per user request): M sweeps 64->3000 (decode-to-prefill batch range),
N,K in {(7168,7168), (36864,7168), (7168,2048)}.

This is Phase 4 step 1: establish where we stand before tuning. The MBS
kernel currently has no scheduler/async-copy (see
docs/oas_mbs_gemm/PHASE3_MBS_KERNEL_DESIGN.md's "Status" section) so this is
NOT yet an apples-to-apples comparison -- it tells us the current gap to close.
"""
import itertools
import sys
import time

sys.path.insert(0, "/scratch/satre/FlyDSL")

import torch  # noqa: E402

import flydsl.compiler as flyc  # noqa: E402
from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.mxfp4_preshuffle import compile_mxfp4_gemm  # noqa: E402
from kernels.mxfp4_preshuffle_mbs import compile_mxfp4_gemm_mbs  # noqa: E402
from tests.kernels.utils import fp4_utils  # noqa: E402
from tests.kernels.utils import oas_mbs_quant  # noqa: E402

device = torch.device("cuda")
torch.manual_seed(0)

M_VALUES = [64, 128, 256, 512, 1024, 1536, 2048, 3000]
NK_SHAPES = [(7168, 7168), (36864, 7168), (7168, 2048)]
# Candidate tile configs to search per shape (BM, BN, BK). BK in {128,256}.
TILE_CANDIDATES = [
    (16, 128, 256),
    (32, 128, 256),
    (64, 128, 128),
    (64, 128, 256),
    (64, 256, 128),
    (128, 128, 128),
    (128, 256, 128),
]


def bench_ms(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def tflops(M, N, K, ms):
    return 2 * M * N * K / (ms / 1e3) / 1e12


def make_baseline_runner(N, K, tile_m, tile_n, tile_k, M):
    M32, N32 = (M + 31) // 32 * 32, (N + 31) // 32 * 32
    a = torch.randn(M32, K, device=device)
    b = torch.randn(N32, K, device=device)
    a_q, sa, _ = fp4_utils.per_1x32_f4_quant(a)
    a_q = a_q[:M]
    b_q, sb, _ = fp4_utils.per_1x32_f4_quant(b)
    b_q = b_q[:N]
    b_shuf = fp4_utils.shuffle_weight_w4(b_q, 16, False, False)
    sa_shuf = fp4_utils.shuffle_scale_w4(sa, 1, False)
    sb_shuf = fp4_utils.shuffle_scale_w4(sb, 1, False)
    c = torch.zeros((M, N), dtype=torch.bfloat16, device=device)
    bias = torch.empty(0, dtype=torch.bfloat16, device=device)
    _tb = lambda t: t if t.dtype in (torch.uint8, torch.int8) else t.view(torch.uint8)
    args = (c.view(-1), _tb(a_q).contiguous().view(-1), _tb(b_shuf).contiguous().view(-1),
            _tb(sa_shuf).contiguous().view(-1), _tb(sb_shuf).contiguous().view(-1),
            bias, M, N, torch.cuda.current_stream())
    launch_fn = compile_mxfp4_gemm(N=N, K=K, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, out_dtype="bf16")
    compiled = flyc.compile(launch_fn, *args)
    return lambda: compiled(*args)


def make_mbs_runner(N, K, tile_m, tile_n, tile_k, M):
    M32, N32 = (M + 31) // 32 * 32, (N + 31) // 32 * 32
    a = torch.randn(M32, K, device=device)
    b = torch.randn(N32, K, device=device)
    a_q, sa, m8a, _ = oas_mbs_quant.per_1x32_f4_quant_oas_mbs(a)
    a_q = a_q[:M]
    b_q, sb, m8b, _ = oas_mbs_quant.per_1x32_f4_quant_oas_mbs(b)
    b_q = b_q[:N]
    b_shuf = fp4_utils.shuffle_weight_w4(b_q, 16, False, False)
    sa_shuf = fp4_utils.shuffle_scale_w4(sa, 1, False)
    sb_shuf = fp4_utils.shuffle_scale_w4(sb, 1, False)
    mbs_a_k = oas_mbs_quant.shuffle_mbs_scale_w4(m8a, M32).to(device)
    mbs_b_k = oas_mbs_quant.shuffle_mbs_scale_w4(m8b, N32).to(device)
    c = torch.zeros((M, N), dtype=torch.bfloat16, device=device)
    bias = torch.empty(0, dtype=torch.bfloat16, device=device)
    _tb = lambda t: t if t.dtype in (torch.uint8, torch.int8) else t.view(torch.uint8)
    args = (c.view(-1), _tb(a_q).contiguous().view(-1), _tb(b_shuf).contiguous().view(-1),
            _tb(sa_shuf).contiguous().view(-1), _tb(sb_shuf).contiguous().view(-1),
            _tb(mbs_a_k).contiguous().view(-1), _tb(mbs_b_k).contiguous().view(-1),
            bias, M, N, torch.cuda.current_stream())
    launch_fn = compile_mxfp4_gemm_mbs(N=N, K=K, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, out_dtype="bf16")
    compiled = flyc.compile(launch_fn, *args)
    return lambda: compiled(*args)


def best_tile_for(N, K, M, maker):
    best = None
    for bm, bn, bk in TILE_CANDIDATES:
        if K % bk != 0 or N % bn != 0:
            continue
        try:
            fn = maker(N, K, bm, bn, bk, M)
            ms = bench_ms(fn)
        except Exception as e:  # noqa: BLE001
            continue
        if best is None or ms < best[0]:
            best = (ms, (bm, bn, bk))
    return best


if __name__ == "__main__":
    assert get_rocm_arch() == "gfx950"
    print(f"{'N':>7} {'K':>7} {'M':>5} | {'base_ms':>8} {'base_TF':>8} {'base_tile':>14} | "
          f"{'mbs_ms':>8} {'mbs_TF':>8} {'mbs_tile':>14} | {'overhead%':>9}")
    rows = []
    for N, K in NK_SHAPES:
        for M in M_VALUES:
            base = best_tile_for(N, K, M, make_baseline_runner)
            mbs = best_tile_for(N, K, M, make_mbs_runner)
            if base is None or mbs is None:
                print(f"{N:>7} {K:>7} {M:>5} | SKIP (no valid tile config)")
                continue
            base_ms, base_tile = base
            mbs_ms, mbs_tile = mbs
            base_tf = tflops(M, N, K, base_ms)
            mbs_tf = tflops(M, N, K, mbs_ms)
            overhead = (mbs_ms / base_ms - 1) * 100
            print(f"{N:>7} {K:>7} {M:>5} | {base_ms:8.3f} {base_tf:8.1f} {str(base_tile):>14} | "
                  f"{mbs_ms:8.3f} {mbs_tf:8.1f} {str(mbs_tile):>14} | {overhead:9.1f}")
            rows.append((N, K, M, base_ms, base_tf, base_tile, mbs_ms, mbs_tf, mbs_tile, overhead))
    print("\nDONE.")
