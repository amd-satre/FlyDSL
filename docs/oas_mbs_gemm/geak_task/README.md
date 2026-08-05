# Task: optimize MXFP4 OAS+MBS GEMM to match plain MXFP4 GEMM latency

## What this kernel does

`kernels/mxfp4_preshuffle_mbs.py` (`compile_mxfp4_gemm_mbs`) implements an
MXFP4 (E2M1) GEMM on AMD gfx950 enhanced with **Overflow-Aware Scaling (OAS)**
and **Macro Block Scaling (MBS)**, per arXiv:2603.08713 ("Unveiling the
Potential of Quantization with MXFP4"). These techniques recover accuracy
lost to MXFP4's coarse quantization (measured: +0.55 to +0.62 dB QSNR vs true
BF16 output, at zero cost to OAS and ~5-7% theoretical cost to MBS per the
paper) but the current implementation has real, unnecessary overhead.

`kernels/mxfp4_preshuffle.py` (`compile_mxfp4_gemm`) is the production,
already-tuned **plain MXFP4 GEMM** (no OAS/MBS) — this is the ceiling we're
trying to approach. It is NOT the target of optimization; do not edit it,
but it's the reference for "what good performance looks like" on this
hardware/kernel style (async-copy DMA, MFMA instruction scheduler, same
overall tiling structure).

## Goal

Reduce `kernels/mxfp4_preshuffle_mbs.py`'s latency (the `performance_command`
below reports it per shape) as close as possible to `kernels/
mxfp4_preshuffle.py`'s latency on the same shapes (reported in each test
case's `params.baseline_mxfp4_ms` for reference) — i.e. drive
`overhead_pct_vs_baseline_mxfp4` toward 0% (single-digit is the paper's own
claimed ceiling: ~6.2% average GEMM overhead) — **without breaking
correctness** (`correctness_command` must keep passing) or changing the
kernel's numerics (OAS/MBS math itself must not be altered/weakened — the
speedup must come from HOW it's computed, not skipping what it's supposed to
compute).

## Shapes (DeepSeek-R1-shaped, per user request)

M in {64, 1024, 3000} (decode -> prefill range) x (N,K) in
{(7168,7168), (36864,7168), (7168,2048)} — 9 cases total, see
`scripts/task_runner.py::TEST_SHAPES` (tile config is fixed per shape from a
prior manual sweep — this task is about kernel code, not tile search).

## Known findings from manual tuning so far (read before starting)

See `docs_from_prior_tuning.md` in this directory for the full diagnostic
trail. Summary: the fundamental OAS+MBS algorithm (per-128-K-macro-block
local-accumulate + Hadamard-merge) only costs ~5% in isolation (matching the
paper). The gap to close is **`load_mbs()`**'s memory-load pattern: it issues
8 small buffer loads (4 A-side dwords + 4 B-side bytes) synchronously, right
before the compute for every K-tile iteration, with **no prefetch/pipelining**
— unlike the production kernel's carefully double-buffered A-tile DMA, these
sit directly on the critical path. A fast reciprocal (`rocdl.rcp`) already
replaced a division bottleneck (162% -> 113.5% overhead at one reference
shape); pipelining the MBS loads one iteration ahead (prefetch macro-block
`kt+1`'s scales while computing macro-block `kt`, mirroring how A's LDS is
already double-buffered) is the next, not-yet-attempted lever.

## Environment note (read before running anything)

This box's ROCm/PyTorch/FlyDSL stack lives inside a Docker container
(`satre-oas-mbs-flydsl`, pinned to GPU 4 via `HIP_VISIBLE_DEVICES`), not on
the bare host. **You don't need to do anything special about this** —
`scripts/task_runner.py` detects whether `flydsl`/`torch` are importable and
transparently re-execs itself inside the container if not (the container
bind-mounts `/scratch/satre` 1:1, so paths match on both sides). Just run
`python3 scripts/task_runner.py {compile|correctness|performance}` as usual.

The FlyDSL package itself (built native extensions) is NOT part of this task
dir — it's the fixed, shared install at `/scratch/satre/FlyDSL`, imported via
an absolute `sys.path` entry in `task_runner.py`. Only edit files inside
*this* directory; `/scratch/satre/FlyDSL` is shared, out-of-scope infra.
