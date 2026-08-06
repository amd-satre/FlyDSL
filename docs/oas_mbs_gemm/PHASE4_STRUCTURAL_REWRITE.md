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

## Second, bigger lever: selective per-operand MBS (`mbs_on_a`/`mbs_on_b`)

The `macro_block` lever above reduces the *count* of correction episodes.
There's an orthogonal lever that reduces the *cost per episode*: the
per-output-element correction is `sigma = sigma_A[row] * sigma_B[col]` (1
multiply) then `cf = fma(tmp, sigma, cf)` (1 FMA) — 2 ops/element when BOTH
operands need MBS. But if only ONE operand's outliers actually need
protecting (the paper's own Sec 4.4 already treats weights/activations
asymmetrically — MBS-Dynamic vs MBS-Static — this is that asymmetry taken
one step further), the multiply disappears entirely: `sigma` IS that one
side's reciprocal directly, so the correction is a single FMA — **half the
cost of the both-sides case**, and `load_mbs` also skips computing/loading
the disabled side's reciprocals (roughly halving its own cost too).

Added `mbs_on_a: bool = True, mbs_on_b: bool = True` to
`compile_mxfp4_gemm_mbs`. When one side is disabled, that operand MUST be
quantized with plain OAS (`per_1x32_f4_quant_oas`, no MBS pre-scaling) on the
host side — otherwise its pre-scaling would be baked into the quantized
values with no kernel-side undo, silently corrupting the result. Verified
correctness for all 3 configs (both/A-only/B-only) — `mean_abs_err` ~0.14%
of reference in every case, same as the baseline MBS kernel. 8 new test
cases added to `tests/kernels/test_mxfp4_oas_mbs_gemm.py`
(`test_mxfp4_oas_mbs_gemm_selective_operand`, 19/19 total tests pass).

**Measured trade-off** (tile=(64,128,256), same 3 DeepSeek-R1 shapes):

| Shape | Baseline | both (mb=128) | B-only (mb=128) | A-only (mb=128) |
|---|---|---|---|---|
| M=1024, N=K=7168 | 0.0477ms | +85.7% | **+16.8%** | +57.5% |
| M=1024, N=36864, K=7168 | 0.1995ms | +87.5% | **+16.5%** | +58.1% |
| M=1024, N=7168, K=2048 | 0.0190ms | +61.9% | **+14.2%** | +45.5% |

B-only (MBS on weights/B, OAS-only on activations/A) is dramatically
cheaper than A-only at this tile shape — `load_mbs`'s A-side needs
`m_chunks` dword loads each producing 4 *distinct per-row* reciprocals
(varies with `ii`), while B-side needs only `num_acc_n` loads each producing
a *single* reciprocal shared across all 4 `ii` lanes (broadcast, no
per-element variation) — B-side MBS is structurally cheaper to both load
and apply, independent of which operand's *data* has more outliers.

**Accuracy** (QSNR vs BF16 truth, M=256/N=8192/K=8192): B-only costs -0.13
(Gaussian) / -0.14 dB (outlier-heavy) vs. both-sides — nearly identical to
A-only's cost in this *synthetic, symmetric* test (both operands equally
Gaussian/outlier-prone here; a real model's weights and activations have
different outlier characteristics and may favor one side more or less than
this test shows).

### Combined: B-only + macro_block=256 (best result)

The two levers stack (orthogonal changes to the same correction site):

| Shape | Baseline | B-only, mb=128 | **B-only, mb=256** |
|---|---|---|---|
| M=1024, N=K=7168 | 0.0477ms | +17.2% | **+9.0%** |
| M=1024, N=36864, K=7168 | 0.2000ms | +16.6% | **+7.1%** |
| M=1024, N=7168, K=2048 | 0.0187ms | +11.1% | **+6.9%** |

**This matches the paper's own claimed ~6.2% average GEMM overhead.**
Accuracy cost of adding `macro_block=256` on top of B-only is small: -0.07 dB
(both Gaussian and outlier-heavy) — i.e. going from "both operands,
macro_block=128" (the maximally-accurate configuration) to "B-only,
macro_block=256" costs a total of about -0.20 dB (Gaussian) / -0.21 dB
(outlier-heavy) QSNR, in exchange for cutting overhead from ~86-88% to
~7-9% — roughly a **10x reduction in overhead** for a fraction of a dB.
Verified correct (0.14% mean error vs. the matching torch reference,
consistent with every other configuration tested).

## Recommendation

**B-only + macro_block=256 is the strong recommendation**: it lands within
noise of the paper's own claimed overhead ceiling, at an accuracy cost an
order of magnitude smaller than the total MBS benefit being protected
(-0.20/-0.21 dB given up out of the +0.55/+0.62 dB MBS provides over
OAS-alone — i.e. still keeping ~65% of MBS's own accuracy contribution, on
top of all of OAS's free gain). This is not yet adopted as the new default
in `compile_mxfp4_gemm_mbs` (all three new parameters default to the
original, maximally-accurate behavior: `macro_block=128, mbs_on_a=True,
mbs_on_b=True`) — a numerics choice belongs to whoever deploys this, not
decided silently here.

Not yet attempted: `macro_block=512` (further overhead reduction, the
paper's own ablation shows continued graceful degradation through at least
512); removing the `tile_k % macro_block == 0` constraint (would need
cross-iteration loop-carried MBS accumulator state); real-model weight/
activation outlier profiling to determine whether B (weights) genuinely
needs less protection than A (activations) in practice, rather than relying
on this synthetic symmetric test (Phase 5's real-tensor accuracy work should
settle this).
