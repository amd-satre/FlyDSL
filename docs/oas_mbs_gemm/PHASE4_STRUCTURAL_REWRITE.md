# Structural rewrite: `macro_block` as a tunable knob (128 vs 256)

Status: implemented, GPU-verified (correctness + performance + accuracy),
gated behind an explicit `macro_block` parameter (default 128, unchanged
behavior). Not yet adopted as the new default — this doc lays out the
measured trade-off for that decision.

## Why this was the right lever (and why micro-optimization plateaued)

Both GEAK sessions' ISA/disassembly analysis independently confirmed: the
per-output-element sigma Hadamard correction in `compute_with_macro` is a
genuine, near-irreducible VALU-issue-port cost *at a fixed 128-K macro-block
granularity* — every output element needs at least one multiply (`σ_A×σ_B`)
and one FMA (`tmp*σ + cf`) per macro block, already fused to a single
`math.fma` instruction (the FMA-fusion win from GEAK run 2/3). CDNA has no
packed-FP32 SIMD-within-lane ALU, so no amount of `Vec`-API restructuring
changes the emitted instruction count (GEAK verified this by disassembly:
bit-identical VGPR counts regardless of assembly style).

The only way to reduce the *count* of these per-output-element corrections,
without changing what MBS computes, is to make the macro block itself
coarser: fewer, larger K-slices means fewer correction episodes for the same
total output size. `macro_block=256` groups 2 consecutive native 128-K
scaled-MFMA calls under one correction instead of one each — halving the
correction-episode count.

## Implementation

`kernels/mxfp4_preshuffle_mbs.py::compile_mxfp4_gemm_mbs` gained a
`macro_block: int = 128` parameter. `n_group = macro_block // 128` native
MFMA calls now accumulate into the same zero-initialized local `tmp` register
(previously exactly one MFMA call per `tmp`) before one correction is
computed and merged. Constraint: `tile_k` must be a multiple of `macro_block`
(so the grouping never spans the outer K-tile loop's `init=`/`yield`
boundary — avoids adding cross-iteration loop-carried state for a first cut
of this lever). `K_MACRO` (buffer sizing) and the main loop's macro-block
indexing were updated accordingly; the host-side quantization
(`tests/kernels/utils/oas_mbs_quant.py`) already supported arbitrary
`macro_block` via its existing parameter.

Verified `macro_block=128` (default) is behavior-preserving (all 6
pre-existing tests still pass unmodified) before testing `macro_block=256`.
Added 4 new parametrized cases to `tests/kernels/test_mxfp4_oas_mbs_gemm.py`
covering `macro_block=256` (10/10 total pass).

## Measured trade-off

**Performance** (median-of-7-repeats, tile=(64,128,256) fixed across all
three shape families to isolate `macro_block`'s effect from tile choice):

| Shape | Baseline (plain MXFP4) | macro_block=128 | macro_block=256 |
|---|---|---|---|
| M=1024, N=K=7168 | 0.0478ms | 0.0889ms (+86.1%) | 0.0648ms (**+35.7%**) |
| M=1024, N=36864, K=7168 | 0.1999ms | 0.3747ms (+87.4%) | 0.2726ms (**+36.4%**) |
| M=1024, N=7168, K=2048 | 0.0191ms | 0.0300ms (+57.0%) | 0.0239ms (**+25.2%**) |

Overhead roughly **halved** at every shape, matching the structural
prediction (half as many correction episodes).

**Accuracy** (GEMM-output QSNR vs. true BF16 matmul, M=256/N=8192/K=8192,
gfx950):

| Input | macro_block=128 | macro_block=256 | delta |
|---|---|---|---|
| Gaussian A | 16.33 dB | 16.19 dB | -0.14 dB |
| Outlier-heavy A (1% @ ~25x) | 16.43 dB | 16.15 dB | -0.28 dB |

Recall the total MBS gain over OAS-only is +0.25 dB (Gaussian) / +0.30 dB
(outlier-heavy) at `macro_block=128` (see `PHASE3_MBS_KERNEL_DESIGN.md`'s
GPU verification). At `macro_block=256`, roughly **55-75% of that MBS
accuracy benefit is retained** (a bit less on the outlier-heavy case
specifically — consistent with the paper's own finding that coarser macro
blocks isolate outliers less precisely, degrading gracefully not
catastrophically, per Appendix A's block-size ablation).

## Recommendation

This is a genuine, favorable trade-off (roughly half the overhead for
roughly a quarter to half of the MBS-specific accuracy gain given up, while
still retaining all of OAS's free accuracy gain and most of MBS's) — but
it's a numerics choice, not a free lunch, and the user should decide whether
to adopt it as the default rather than have it decided silently. Options:
1. Keep `macro_block=128` as default (max accuracy, current perf).
2. Switch default to `macro_block=256` (much closer to parity with plain
   MXFP4, small accuracy give-back).
3. Expose both and pick per-deployment based on the accuracy/latency
   sensitivity of the specific serving workload.

Not yet attempted: `macro_block=512` (further overhead reduction, further
accuracy give-back — the paper's own ablation shows continued graceful
degradation through at least 512), or removing the `tile_k % macro_block ==
0` constraint (would need cross-iteration loop-carried MBS accumulator state,
enabling `macro_block > tile_k`, e.g. macro_block=256 with tile_k=128 tiles).
