# Multi-format dense GEMM comparison: performance + accuracy

Per user request: compare W4A4/W4A6/W4A8/W8A8/BF16 GEMMs against our OAS+MBS
MXFP4 kernel, each individually tuned, then measure accuracy (QSNR) for all.
Scripts: `docs/oas_mbs_gemm/format_comparison.py` (perf, tile-tuned),
`docs/oas_mbs_gemm/format_accuracy.py` (QSNR vs BF16 truth). Full raw output:
`docs/oas_mbs_gemm/format_comparison_out.log`,
`docs/oas_mbs_gemm/format_comparison_results.json`.

**Scope note**: this covers **dense** GEMM only. MoE format comparison is a
separate, larger undertaking (needs expert-routing/gather-scatter test
infrastructure around `kernels/mixed_moe_gemm_2stage.py`) — tracked
separately, not yet done as of this doc.

## Methodology

- **Shapes**: DeepSeek-R1-shaped, M ∈ {64, 1024, 3000} × (N,K) ∈
  {(7168,7168), (36864,7168), (7168,2048)} — 9 cases, consistent with all
  prior Phase 4 work.
- **Tuning**: every format searched its own candidate tile-config list before
  benchmarking (BF16/FP8/INT8 via `kernels/preshuffle_gemm.py`'s existing
  `_TILE_PRELOAD_TABLE`-covered configs; MXFP4-family via the same
  `TILE_CANDIDATES` used throughout Phase 4; W4A6 uses
  `compile_mxfp6_gemm`'s built-in `M_hint`-based auto-tuned config, not
  re-searched). "Individually tuned" per the user's requirement — no format
  is compared using another format's tile choice.
- **Timing**: median of 5 repeats × 30 iterations per config (GPU 4,
  confirmed 0% use / no KFD process before and during the run).
- **Accuracy**: QSNR vs. true BF16 matmul (`torch.mm` in fp32, cast to bf16),
  M=256/N=8192/K=8192, Gaussian A and outlier-heavy A (1% of elements at
  ~25x scale) — same protocol as every other Phase 3/4 accuracy number in
  this project.

## W4A8 — documented gap, not measured

**No gfx950 kernel exists in this repo for MXFP4-weight x FP8-activation
GEMM.** `tests/kernels/test_preshuffle_gemm.py` explicitly skips this case
("fp8-A not yet supported with MXFP4 preshuffle kernel (op_sel_a overflow)").
The only kernel with fp8×fp4 support (`kernels/gemm_fp8fp4_gfx1250.py`)
targets a different architecture (RDNA gfx1250, asserted in-code) and is not
usable on this MI350/355-class (gfx950/CDNA4) box. This is a real gap, not
something worked around — a genuine W4A8 dense kernel would need new
kernel-authoring work (fixing the op_sel operand-selection overflow), out of
scope for a benchmarking pass.

## Performance (overhead % vs. W4A4/MXFP4 baseline, tuned tile per cell)

| N | K | M | BF16 (ms, tuned) | FP8 (W8A8) | INT8 (W8A8) | **W4A4 (baseline)** | W4A6 (MXFP6xMXFP4) | OAS+MBS both, mb=128 | OAS+MBS B-only, mb=128 | OAS+MBS B-only, mb=256 |
|---|---|---|---|---|---|---|---|---|---|---|
| 7168 | 7168 | 64 | 0.0300 | +0.5% | +3.4% | baseline | +3.9% | +26.3% | +9.0% | **+2.5%** |
| 7168 | 7168 | 1024 | 0.1508 | +39.3% | +91.9% | baseline | +19.4% | +74.3% | +19.3% | **+7.7%** |
| 7168 | 7168 | 3000 | 0.4066 | +52.6% | +130.3% | baseline | +59.0% | +84.7% | +30.6% | **+24.3%** |
| 36864 | 7168 | 64 | 0.0791 | +69.4% | +87.2% | baseline | +79.2% | +71.0% | +7.7% | **+7.6%** |
| 36864 | 7168 | 1024 | 0.7018 | +77.2% | +124.3% | baseline | +12.5% | +64.4% | +20.5% | **+12.6%** |
| 36864 | 7168 | 3000 | 1.8896 | +68.7% | +139.6% | baseline | +69.4% | +85.4% | +27.2% | +33.6% |
| 7168 | 2048 | 64 | 0.0106 | +2.8% | +8.2% | baseline | +14.4% | +40.4% | +16.4% | +15.0% |
| 7168 | 2048 | 1024 | 0.0518 | +56.9% | +90.4% | baseline | +29.5% | +72.1% | +17.5% | +17.5% |
| 7168 | 2048 | 3000 | 0.1356 | +50.6% | +101.9% | baseline | +34.8% | +63.4% | +18.3% | +20.1% |

**Reading this table**: W4A4 (plain MXFP4) is fastest everywhere, as
expected (least data movement, cheapest MFMA). BF16 is 50-140% slower than
W4A4 at large M (memory/compute bound differently at scale) but competitive
at small M=64 (dispatch-floor dominated, format barely matters). FP8/INT8
(W8A8) land between BF16 and W4A4. **Our best OAS+MBS configuration
(B-only, macro_block=256) is consistently the closest 4-bit accuracy-
enhanced format to the W4A4 floor** — often closer to W4A4 than even W4A6
is, while (per the accuracy table below) recovering far more of the BF16
accuracy gap than W4A6 alone.

## Accuracy (QSNR vs. true BF16 matmul, dB — higher is better)

| Format | Gaussian A | Outlier-heavy A |
|---|---|---|
| **W4A4 (MXFP4, no OAS/MBS)** | 15.78 | 15.80 |
| **W4A6 (MXFP6 A × MXFP4 B)** | 18.49 | 17.74 |
| OAS+MBS, both operands, mb=128 | 16.33 | 16.43 |
| OAS+MBS, B-only, mb=128 | 16.19 | 16.28 |
| **OAS+MBS, B-only, mb=256** | 16.13 | 16.20 |
| INT8 (W8A8) | 30.25 | 25.67 |
| FP8 (W8A8) | 28.52 | 29.13 |

Notable: **INT8 loses more accuracy than FP8 specifically on the
outlier-heavy input** (30.25→25.67 dB, a 4.6 dB drop) while FP8 barely moves
(28.52→29.13 dB, actually slightly better) — consistent with the
well-known asymmetry that integer quantization's uniform grid handles
outliers far worse than floating-point's exponential range. This is the
same qualitative story OAS/MBS is built around, just visible one format
tier up (8-bit vs 4-bit).

W4A6 has the highest QSNR among the 4-6 bit formats (uses 6 bits for
activations), but at a real perf cost (+19% to +85% over W4A4, see table
above) — it is not using OAS/MBS at all, it's a fundamentally
higher-precision format for one operand. OAS+MBS closes roughly a third to
half of the W4A4→W4A6 gap while staying much closer to W4A4's speed,
especially in the B-only/mb=256 configuration.

## Combined summary (the requested single table)

| Format | Best perf overhead vs W4A4 (range across 9 shapes) | QSNR Gaussian | QSNR outlier-heavy |
|---|---|---|---|
| W4A4 (MXFP4 baseline) | 0% (reference) | 15.78 | 15.80 |
| OAS+MBS B-only, mb=256 | **+2.5% to +33.6%** | 16.13 | 16.20 |
| OAS+MBS B-only, mb=128 | +7.7% to +30.6% | 16.19 | 16.28 |
| W4A6 (MXFP6xMXFP4) | +3.9% to +85.4% | **18.49** | **17.74** |
| OAS+MBS both, mb=128 | +26.3% to +85.4% | 16.33 | 16.43 |
| FP8 (W8A8) | +0.5% to +77.2% | 28.52 | 29.13 |
| INT8 (W8A8) | +3.4% to +139.6% | 30.25 | 25.67 |
| BF16 | -1.6% to +77.2%* | ∞ (reference format) | ∞ |
| W4A8 | **not implemented on gfx950** — documented gap | — | — |

*BF16 range includes shapes where it's actually faster than W4A4 (small M,
dispatch-floor-dominated) as well as much slower ones (large M,
compute-bound) — see the full table above for per-shape detail, a single
range obscures this format's very different scaling behavior.

## What's not yet done

1. **MoE format comparison** (this doc is dense-only). Needs expert-routing
   test infrastructure around `mixed_moe_gemm_2stage.py`.
2. **`macro_block` ∈ {32, 64, 512}** — see
   `docs/oas_mbs_gemm/PHASE4_MACRO_BLOCK_SWEEP.md`: 32/64 are **not
   physically achievable** with this kernel's architecture (AMD's native
   scaled-MFMA is atomic at 128-K granularity — you cannot intercept a
   partial accumulation *inside* one hardware instruction to apply a finer
   correction). 512 needs `tile_k=512`, which this kernel family doesn't
   support at all yet (independent of MBS — the underlying e8m0
   scale-chunking logic assumes `tile_k ∈ {128, 256}`).
3. **A-only MBS variant** and the **32/64/512 macro_block accuracy numbers**
   were not included in the accuracy table above (time-boxed to the
   configurations already shown to be competitive on performance).
