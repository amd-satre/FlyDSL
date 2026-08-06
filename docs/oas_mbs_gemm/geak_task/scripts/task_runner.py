#!/usr/bin/env python3
"""Task runner for optimizing kernels/mxfp4_preshuffle_mbs.py (OAS+MBS MXFP4
GEMM on gfx950), goal: match kernels/mxfp4_preshuffle.py's (plain MXFP4,
production/tuned) latency on DeepSeek-R1-shaped GEMMs.

This task dir is intentionally self-contained and small (a few kernel/util
files copied out of the full FlyDSL checkout at /scratch/satre/FlyDSL) so
GEAK's per-engineer/per-round workspace snapshots stay cheap. The FlyDSL
Python package itself (built native extensions) is NOT copied -- it's
imported from the fixed, shared install via sys.path.

DOCKER DELEGATION: this environment's GPU/ROCm/FlyDSL stack lives inside the
`satre-oas-mbs-flydsl` Docker container (pinned to GPU 4 via
HIP_VISIBLE_DEVICES), not on the bare host. If `flydsl`/`torch` aren't
importable directly (i.e. we're running on the host), this script
transparently re-execs itself inside that container at the SAME path (the
container bind-mounts /scratch/satre 1:1), so callers (GEAK's gpu_lock.sh,
direct invocation, etc.) don't need to know about Docker at all.
"""
import json
import os
import subprocess
import sys

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLYDSL_ROOT = "/scratch/satre/FlyDSL"
CONTAINER = "satre-oas-mbs-flydsl"
GPU_ID = os.environ.get("MBS_TASK_GPU", "4")


def _in_container_env():
    try:
        import flydsl  # noqa: F401
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def _delegate_to_docker(mode):
    """Re-exec this same script inside the container (same path, since
    /scratch/satre is bind-mounted 1:1), mirroring stdout/stderr/exit code."""
    cmd = [
        "docker", "exec", "-w", TASK_DIR, "-e", f"HIP_VISIBLE_DEVICES={GPU_ID}",
        CONTAINER, "python3", os.path.abspath(__file__), mode,
    ]
    proc = subprocess.run(cmd)
    sys.exit(proc.returncode)


sys.path.insert(0, FLYDSL_ROOT)  # for the `flydsl` package itself (built extensions)
sys.path.insert(0, TASK_DIR)  # LAST insert wins position 0: must resolve `kernels.*`/`tests.*`
# to TASK_DIR's own (possibly-edited) copies, not shadow them with FLYDSL_ROOT's shared ones.
os.chdir(TASK_DIR)

TASK_NAME = "flydsl/mxfp4_oas_mbs_gemm"

# DeepSeek-R1-shaped GEMMs (per user request): M sweeps decode->prefill, N/K
# from DeepSeek-R1's attention/FFN projection dims. A representative subset
# (not the full 24-point sweep) to keep repeated harness runs fast.
TEST_SHAPES = [
    # (N, K, M, tile_m, tile_n, tile_k) -- tile config fixed per shape family
    # (from docs/oas_mbs_gemm/bench_deepseek_shapes.py's earlier sweep) so
    # GEAK optimizes kernel code, not tile search.
    (7168, 7168, 64, 32, 128, 256),
    (7168, 7168, 1024, 64, 128, 256),
    (7168, 7168, 3000, 128, 256, 128),
    (36864, 7168, 64, 64, 256, 128),
    (36864, 7168, 1024, 128, 256, 128),
    (36864, 7168, 3000, 128, 256, 128),
    (7168, 2048, 64, 32, 128, 256),
    (7168, 2048, 1024, 64, 128, 256),
    (7168, 2048, 3000, 128, 256, 128),
]


def _bench_ms(fn, warmup=5, iters=20):
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _make_mbs_runner(N, K, M, tile_m, tile_n, tile_k):
    import torch

    from kernels.mxfp4_preshuffle_mbs import compile_mxfp4_gemm_mbs
    from tests.kernels.utils import fp4_utils, oas_mbs_quant
    import flydsl.compiler as flyc

    device = torch.device("cuda")
    M32, N32 = (M + 31) // 32 * 32, (N + 31) // 32 * 32
    torch.manual_seed(0)
    a = torch.randn(M32, K, device=device)
    b = torch.randn(N32, K, device=device)
    a_q, sa, m8a, _ = oas_mbs_quant.per_1x32_f4_quant_oas_mbs(a)
    a_q = a_q[:M]
    b_q, sb, m8b, _ = oas_mbs_quant.per_1x32_f4_quant_oas_mbs(b)
    b_q = b_q[:N]
    b_shuf = fp4_utils.shuffle_weight_w4(b_q, 16, False, False)
    sa_shuf = fp4_utils.shuffle_scale_w4(sa, 1, False)
    sb_shuf = fp4_utils.shuffle_scale_w4(sb, 1, False)
    mbs_a = oas_mbs_quant.shuffle_mbs_scale_w4(m8a, M32).to(device)
    mbs_b = oas_mbs_quant.shuffle_mbs_scale_w4(m8b, N32).to(device)
    c = torch.zeros((M, N), dtype=torch.bfloat16, device=device)
    bias = torch.empty(0, dtype=torch.bfloat16, device=device)

    def _tb(t):
        return t if t.dtype in (torch.uint8, torch.int8) else t.view(torch.uint8)

    args = (
        c.view(-1), _tb(a_q).contiguous().view(-1), _tb(b_shuf).contiguous().view(-1),
        _tb(sa_shuf).contiguous().view(-1), _tb(sb_shuf).contiguous().view(-1),
        _tb(mbs_a).contiguous().view(-1), _tb(mbs_b).contiguous().view(-1),
        bias, M, N, torch.cuda.current_stream(),
    )
    launch_fn = compile_mxfp4_gemm_mbs(N=N, K=K, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, out_dtype="bf16")
    compiled = flyc.compile(launch_fn, *args)

    def run():
        compiled(*args)

    # Return a fresh torch reference too, for correctness.
    def torch_ref():
        x = fp4_utils.mxfp4_to_f32(a_q) * fp4_utils.e8m0_to_f32(sa[:M].repeat_interleave(32, dim=1))
        w = fp4_utils.mxfp4_to_f32(b_q) * fp4_utils.e8m0_to_f32(sb[:N].repeat_interleave(32, dim=1))
        factor_a = 1.0 + m8a[:M].float() / 256.0
        factor_b = 1.0 + m8b[:N].float() / 256.0
        MACRO = 128
        xr = x.reshape(M, K // MACRO, MACRO)
        wr = w.reshape(N, K // MACRO, MACRO)
        acc = torch.zeros(M, N, device=device, dtype=torch.float32)
        for kb in range(K // MACRO):
            local = torch.mm(xr[:, kb, :], wr[:, kb, :].T)
            sigma = (1.0 / factor_a[:, kb]).unsqueeze(1) * (1.0 / factor_b[:, kb]).unsqueeze(0)
            acc += local * sigma
        return acc

    return run, torch_ref, c


def _make_baseline_runner(N, K, M, tile_m, tile_n, tile_k):
    import torch

    from kernels.mxfp4_preshuffle import compile_mxfp4_gemm
    from tests.kernels.utils import fp4_utils
    import flydsl.compiler as flyc

    device = torch.device("cuda")
    M32, N32 = (M + 31) // 32 * 32, (N + 31) // 32 * 32
    torch.manual_seed(0)
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

    def _tb(t):
        return t if t.dtype in (torch.uint8, torch.int8) else t.view(torch.uint8)

    args = (
        c.view(-1), _tb(a_q).contiguous().view(-1), _tb(b_shuf).contiguous().view(-1),
        _tb(sa_shuf).contiguous().view(-1), _tb(sb_shuf).contiguous().view(-1),
        bias, M, N, torch.cuda.current_stream(),
    )
    launch_fn = compile_mxfp4_gemm(N=N, K=K, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, out_dtype="bf16")
    compiled = flyc.compile(launch_fn, *args)

    def run():
        compiled(*args)

    return run


def run_compile():
    try:
        from kernels.mxfp4_preshuffle_mbs import compile_mxfp4_gemm_mbs  # noqa: F401

        run, _, _ = _make_mbs_runner(*TEST_SHAPES[0])
        run()
        return True, None
    except Exception as e:  # noqa: BLE001
        import traceback

        return False, traceback.format_exc()


def run_correctness():
    for N, K, M, tm, tn, tk in TEST_SHAPES:
        try:
            run, torch_ref, c = _make_mbs_runner(N, K, M, tm, tn, tk)
            run()
        except Exception as e:  # noqa: BLE001
            import traceback

            return False, f"shape N={N},K={K},M={M}: exception during run: {traceback.format_exc()}"
        c_ref = torch_ref()
        c_f32 = c.float()
        mean_abs_err = (c_f32 - c_ref).abs().mean().item()
        ref_scale = c_ref.abs().mean().item()
        if mean_abs_err > 0.02 * ref_scale:
            return False, (
                f"shape N={N},K={K},M={M}: mean_abs_err={mean_abs_err:.4f} "
                f"({100 * mean_abs_err / ref_scale:.2f}% of ref mean abs) exceeds 2% tolerance"
            )
    return True, None


def run_performance():
    test_cases = []
    for N, K, M, tm, tn, tk in TEST_SHAPES:
        run_mbs, _, _ = _make_mbs_runner(N, K, M, tm, tn, tk)
        ms_mbs = _bench_ms(run_mbs)
        run_base = _make_baseline_runner(N, K, M, tm, tn, tk)
        ms_base = _bench_ms(run_base)
        test_cases.append({
            "test_case_id": f"N{N}_K{K}_M{M}",
            "execution_time_ms": ms_mbs,
            "params": {
                "N": N, "K": K, "M": M, "tile": [tm, tn, tk],
                "baseline_mxfp4_ms": ms_base,
                "overhead_pct_vs_baseline_mxfp4": 100.0 * (ms_mbs / ms_base - 1.0),
            },
        })
    return test_cases


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("compile", "correctness", "performance"):
        print("Usage: task_runner.py {compile|correctness|performance}")
        sys.exit(2)
    mode = sys.argv[1]

    if not _in_container_env():
        _delegate_to_docker(mode)
        return  # unreachable, _delegate_to_docker exits

    build_dir = os.path.join(TASK_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)

    if mode == "compile":
        ok, err = run_compile()
        report = {"status": "ok" if ok else "fail", "error": err}
        with open(os.path.join(build_dir, "compile_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Compilation: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

    elif mode == "correctness":
        ok, err = run_correctness()
        report = {"status": "ok" if ok else "fail", "error": err, "num_shapes": len(TEST_SHAPES)}
        with open(os.path.join(build_dir, "correctness_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

    elif mode == "performance":
        test_cases = run_performance()
        report = {"test_cases": test_cases}
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        for case in test_cases:
            p = case["params"]
            print(
                f"Perf: {case['execution_time_ms']:.4f} ms ({case['test_case_id']}) "
                f"baseline_mxfp4={p['baseline_mxfp4_ms']:.4f}ms "
                f"overhead={p['overhead_pct_vs_baseline_mxfp4']:.1f}%"
            )
        sys.exit(0)


if __name__ == "__main__":
    main()
