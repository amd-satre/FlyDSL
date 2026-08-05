# Phase 4 tuning notes: MBS kernel performance

## DeepSeek-R1-shaped microbenchmark (per user request)

Shapes: M in {64,128,256,512,1024,1536,2048,3000}, (N,K) in
{(7168,7168), (36864,7168), (7168,2048)}. Script:
`docs/oas_mbs_gemm/bench_deepseek_shapes.py`.

**Before any Phase 4 tuning** (`phase4_deepseek_bench_before_tuning.txt`):
production MXFP4 baseline hits up to ~3200 TFLOPS (M=2048, N=K=7168); the
correctness-first MBS kernel from Phase 3 (no scheduler, no async-copy) was
75-300% slower across the sweep -- exactly as flagged in
`PHASE3_MBS_KERNEL_DESIGN.md`'s "Status" section as the known next step.

## Diagnostic sequence (2026-08-05)

Rather than guess, isolated the overhead source with targeted ablations at a
representative shape (M=1024, N=K=7168, tile 64x256x128; baseline=0.0503ms):

1. **Ported the scheduler + async-copy DMA** from `mxfp4_preshuffle.py`
   (`hot_loop_scheduler`, `build_scheduler`, `dma_a_to_lds`) into
   `mxfp4_preshuffle_mbs.py`, unmodified, MBS loads not counted in the
   schedule. Correctness held (6/6 tests), but performance barely moved
   (174% -> 162% overhead at this shape) -- the bottleneck wasn't
   instruction scheduling.
2. **Ablation A** -- stubbed `load_mbs()` to return constant `1.0`
   reciprocals (no memory loads, no readfirstlane, keeping the
   zero-init-local-accumulator + Hadamard-merge structure): **0.0532ms,
   ~4.7% overhead** -- right in line with the paper's own claimed ~6.2%
   average GEMM overhead. This is the key finding: **the fundamental
   per-128-K-macro-block local-accumulate/merge algorithm is cheap**; the
   Phase 3 kernel's actual `load_mbs()` implementation was the entire
   problem, not the MBS algorithm itself.
3. **Fix 1**: hoisted `rocdl.readfirstlane` calls out of the `mi`/`ni` loops
   in `load_mbs()` -- the uniform (workgroup-level) address part doesn't
   depend on `mi`/`ni` at all, so redundantly re-broadcasting it 4x (once per
   `mi`) was pure waste. Barely moved the needle (162% -> still ~162-174%),
   ruling out SALU/broadcast overhead as dominant.
4. **Ablation B** -- kept all loads/readfirstlane, stubbed only the
   reciprocal math (`_byte_to_recip`) to skip division: **0.0533ms**,
   matching Ablation A almost exactly. This isolated the true dominant cost:
   **`arith.divf`**. `_byte_to_recip` did two divisions per call
   (`m8/256` and `1/factor`), called `m_chunks*4 + num_acc_n` = 20 times per
   `kh` per K-tile iteration -- `arith.divf` is not a cheap single
   instruction on this hardware (full IEEE division, multi-instruction
   Newton-Raphson sequence), and 20+ of them per iteration dominated
   everything else combined.
5. **Fix 2** (the real fix): `factor = 1 + m8/256` is always in `[1, 2)`,
   well-conditioned for a single-instruction hardware reciprocal --
   replaced `arith.divf(one, factor)` with `rocdl.rcp(T.f32, factor)`
   (`v_rcp_f32`), and replaced `m8/256` with `m8 * (1/256)` (compile-time
   reciprocal constant, no division at all). Result: **113.5% overhead**
   (0.0502ms -> 0.1072ms) -- roughly halved from the pre-fix 162-300% range,
   still well above the ~5-7% floor established by Ablations A/B.

## Current status and remaining gap

The ~4.7% floor (Ablation A, no memory traffic) vs. the current ~113.5% (with
real loads + fast reciprocal) implicates the **MBS scale loads themselves as
unpipelined, latency-exposed memory traffic** as the next bottleneck: 4 A-side
dword loads + 4 B-side byte loads per `kh` per K-tile iteration are issued
synchronously right before `compute_with_macro`, with no prefetch/overlap --
unlike the production kernel's carefully double-buffered A-tile DMA, these
loads sit directly on the critical path every iteration. Candidate next
optimizations (not yet attempted): pipeline `load_mbs()` one iteration ahead
(prefetch macro-block `kt+1`'s scales while computing macro-block `kt`, same
double-buffering pattern as A's LDS), and/or reduce the load count via
mi/ni-pairing (matching `load_sc()`'s `m_pairs`/`n_pairs` pattern instead of
one load per `mi`/`ni`).

Per user direction, handing further optimization to GEAK
(`github.com/AMD-AGI/GEAK`) rather than continuing manual ablation -- see
`docs/oas_mbs_gemm/GEAK_OPTIMIZATION_LOG.md` for that effort's setup and
findings.
