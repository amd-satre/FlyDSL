"""Correctness + benchmark: compile_mxfp6_gemm dense_fp6=True.

Tests that the dense-load path produces bit-exact output vs padded (same MFMA,
different A-load), then benchmarks both on DeepSeek-R1 shapes.

Usage:
  HIP_VISIBLE_DEVICES=1 python3 tests/kernels/test_dense_fp6_gemm.py [--bench]
"""
from __future__ import annotations
import argparse, math, os, statistics, sys
import torch
from triton.testing import do_bench

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from kernels.mxfp4_preshuffle import compile_mxfp6_gemm
from tests.kernels.utils import fp4_utils

DEV = "cuda"
NTRIALS = 5
WARMUP  = 20
REP     = 80

# DeepSeek-R1 dense + MoE shapes (N, K, label)
SHAPES = [
    (7168, 16384, "o_proj"),
    (36864,  7168, "dense_gate_up"),
    (7168, 18432, "dense_down"),
    (4096,  7168, "moe_gate_up"),
    (7168,  2048, "moe_down"),
]

# Tile configs for dense_fp6: must satisfy tile_m*tile_k//32 ≤256 or %256==0
DENSE_TILES = [
    (32,  256, 256),   # 32*256/32=256 → full mode, 1 blk/thread
    (64,  256, 256),   # 512 → full mode, 2 blk/thread
    (128, 256, 256),   # 1024 → full mode, 4 blk/thread
    (64,  128, 256),   # 256 → full mode, 1 blk/thread (tile_k=128)
    (128, 128, 256),   # 512 → full mode, 2 blk/thread
    (16,  256, 256),   # 128 < 256 → partial mode
]


def _prep(M, N, K, dense: bool):
    """Prepare quantized inputs for padded or dense fp6 layout."""
    a32  = torch.randn(M, K, device=DEV, dtype=torch.float32)
    b32  = torch.randn(N, K, device=DEV, dtype=torch.float32)

    a_q, scale_a, _ = fp4_utils.per_1x32_f6_quant(a32)   # padded: (M, K) uint8
    b_q, scale_b, _ = fp4_utils.per_1x32_f4_quant(b32)
    b_sh = fp4_utils.shuffle_weight_w4(b_q[:N], 16, False, False)
    sa   = fp4_utils.shuffle_scale_w4(scale_a, 1, False)
    sb   = fp4_utils.shuffle_scale_w4(scale_b[:N], 1, False)

    if dense:
        # Convert padded (M, K) → dense (M, K*3//4): strip 8 zero bytes per block
        nblk    = K // 32
        a_dense = a_q[:M].view(M, nblk, 32)[:, :, :24].contiguous().view(M, K * 3 // 4)
        a_in    = _b(a_dense)
    else:
        a_in = _b(a_q[:M])

    c   = torch.zeros((M, N), device=DEV, dtype=torch.bfloat16)
    bias = torch.empty(0, dtype=torch.bfloat16, device=DEV)
    return c, a_in, _b(b_sh), _b(sa), _b(sb), bias


def _b(t):
    return t.contiguous().view(-1) if t.dtype == torch.uint8 else t.view(torch.uint8).contiguous().view(-1)


def build(M, N, K, dense: bool, tile_m=64, tile_n=256, tile_k=256):
    import flydsl.compiler as flyc
    c, a_in, b_sh, sa, sb, bias = _prep(M, N, K, dense)
    m_pad = (M + tile_m - 1) // tile_m * tile_m

    # Pad M to tile boundary
    if m_pad > M:
        a_in = torch.cat([a_in, torch.zeros(
            (m_pad - M) * (K * 3 // 4 if dense else K),
            dtype=torch.uint8, device=DEV)])
        c = torch.zeros((m_pad, N), device=DEV, dtype=torch.bfloat16)

    launch = compile_mxfp6_gemm(
        N=N, K=K, M_hint=m_pad,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        out_dtype="bf16",
        dense_fp6=dense,
    )
    args = (c.view(-1), a_in, b_sh, sa, sb, bias, m_pad, N, torch.cuda.current_stream())
    compiled = flyc.compile(launch, *args)
    fn = lambda: compiled(*args)
    fn(); torch.cuda.synchronize()   # warmup
    return fn, c, m_pad, M


def check_correctness(N, K, tile_m=64, tile_n=256, tile_k=256):
    """Compare dense vs padded output on same inputs — must be bit-exact."""
    M = tile_m * 2  # use at least 2 tiles for coverage
    c, a_in, b_sh, sa, sb, bias = _prep(M, N, K, dense=False)

    import flydsl.compiler as flyc

    # Padded reference
    launch_pad = compile_mxfp6_gemm(N=N, K=K, M_hint=M,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, out_dtype="bf16")
    c_pad = torch.zeros((M, N), device=DEV, dtype=torch.bfloat16)
    args_pad = (c_pad.view(-1), a_in, b_sh, sa, sb, bias, M, N, torch.cuda.current_stream())
    compiled_pad = flyc.compile(launch_pad, *args_pad)
    compiled_pad(*args_pad); torch.cuda.synchronize()

    # Dense variant (same quantized values, different A layout)
    nblk    = K // 32
    a_q_pad = a_in.view(-1).view(M, K)  # padded (M, K) uint8
    a_dense = a_q_pad.view(M, nblk, 32)[:, :, :24].contiguous().view(M, K * 3 // 4)
    a_den_flat = _b(a_dense)

    launch_den = compile_mxfp6_gemm(N=N, K=K, M_hint=M,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, out_dtype="bf16", dense_fp6=True)
    c_den = torch.zeros((M, N), device=DEV, dtype=torch.bfloat16)
    args_den = (c_den.view(-1), a_den_flat, b_sh, sa, sb, bias, M, N, torch.cuda.current_stream())
    compiled_den = flyc.compile(launch_den, *args_den)
    compiled_den(*args_den); torch.cuda.synchronize()

    match = torch.equal(c_pad, c_den)
    max_diff = (c_pad.float() - c_den.float()).abs().max().item()
    return match, max_diff


def bench_one(M, N, K, dense: bool, tile_m, tile_n, tile_k):
    try:
        fn, c, m_pad, _ = build(M, N, K, dense, tile_m, tile_n, tile_k)
        samples = [do_bench(fn, warmup=WARMUP, rep=REP) * 1e3 for _ in range(NTRIALS)]
        mean = statistics.mean(samples)
        std  = statistics.stdev(samples) if len(samples) > 1 else 0.0
        return mean, std
    except Exception as e:
        return float("nan"), float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench",  action="store_true")
    ap.add_argument("--quick",  action="store_true", help="Only M=64,256")
    ap.add_argument("--shape",  default="all")
    args = ap.parse_args()

    print(f"Device: {torch.cuda.get_device_name(0)}\n")

    # ── Correctness ────────────────────────────────────────────────────────
    print("=== Correctness check (dense vs padded, bit-exact) ===")
    all_pass = True
    for tile_m, tile_n, tile_k in DENSE_TILES:
        N, K = 4096, 7168  # moe_gate_up as canary shape
        try:
            ok, max_diff = check_correctness(N, K, tile_m, tile_n, tile_k)
            status = "PASS ✓" if ok else f"FAIL ✗ (max_diff={max_diff:.4f})"
            print(f"  tile=({tile_m},{tile_n},{tile_k}): {status}")
            if not ok:
                all_pass = False
        except Exception as e:
            print(f"  tile=({tile_m},{tile_n},{tile_k}): ERROR: {e}")
            all_pass = False
    print(f"\nAll correctness checks: {'PASS ✓' if all_pass else 'FAIL ✗'}")

    if not args.bench:
        print("\nRun with --bench to benchmark all shapes.")
        return

    # ── Benchmark ─────────────────────────────────────────────────────────
    Ms = [64, 256] if args.quick else [32, 64, 128, 256, 512, 1024, 2048]
    shapes = [(N, K, lbl) for N, K, lbl in SHAPES
              if args.shape == "all" or args.shape in lbl]

    print(f"\n=== Benchmark: padded vs dense fp6 (µs, mean±std over {NTRIALS} trials) ===")
    print(f"Same tile used for both variants (apples-to-apples). NTRIALS={NTRIALS}.")
    print()

    def _pick_fair_tile(M, N, K):
        """Pick largest valid full-mode tile that works for dense_fp6 (divisible by 256)."""
        for tm in [128, 64, 32]:
            for tk in [256, 128]:
                total_blks = tm * tk // 32
                if total_blks % 256 == 0 or total_blks <= 256:
                    tn = 256 if N % 256 == 0 else 128
                    return tm, tn, tk
        return 32, 128, 256

    hdr = f"{'shape':15s} {'M':5s} {'tile':12s} | {'padded µs':12s} {'dense µs':12s} {'ratio':8s} | {'TFLOPS_pad':10s} {'TFLOPS_den':10s}"
    print(hdr)
    print("-" * len(hdr))

    for N, K, lbl in shapes:
        for M in Ms:
            tm, tn, tk = _pick_fair_tile(M, N, K)

            p_us, p_std = bench_one(M, N, K, False, tm, tn, tk)
            d_us, d_std = bench_one(M, N, K, True,  tm, tn, tk)

            if math.isfinite(p_us) and math.isfinite(d_us):
                ratio = d_us / p_us
                flops = 2 * M * N * K
                tf_p  = flops / (p_us * 1e-6) / 1e12
                tf_d  = flops / (d_us * 1e-6) / 1e12
                flag  = "✓" if ratio < 0.97 else ("≈" if ratio < 1.03 else "✗")
                print(f"{lbl:15s} {M:5d} {str((tm,tn,tk)):12s} | {p_us:7.2f}±{p_std:4.2f}    {d_us:7.2f}±{d_std:4.2f}    {ratio:7.3f}× {flag} | {tf_p:10.0f} {tf_d:10.0f}", flush=True)
            else:
                print(f"{lbl:15s} {M:5d} {str((tm,tn,tk)):12s} | {'ERR':12s} {'ERR':12s}")

    print("\nDone.")


if __name__ == "__main__":
    main()
