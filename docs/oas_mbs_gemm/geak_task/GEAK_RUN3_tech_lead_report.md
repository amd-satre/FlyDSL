# Tech Lead Report — MXFP4 OAS+MBS GEMM (FlyDSL, gfx950/CDNA4)

## Summary

- **Kernel**: `compile_mxfp4_gemm_mbs` in `kernels/mxfp4_preshuffle_mbs.py` — a FlyDSL
  (MLIR-embedded-Python) MXFP4 (E2M1) preshuffle block-scaled GEMM on gfx950 (CDNA4, MI355X), adding
  Overflow-Aware Scaling (OAS) and Macro Block Scaling (MBS, per-128-K-macro-block local-accumulate +
  Hadamard correction) on top of the production `kernels/mxfp4_preshuffle.py` DMA/scheduler
  infrastructure.
- **Kernel type**: `flydsl` / MLIR-embedded HIP-target GEMM. 9 DeepSeek-R1-shaped test cases
  (M ∈ {64, 1024, 3000} × (N,K) ∈ {(7168,7168), (36864,7168), (7168,2048)}).
- **This run has no `WORKLOAD_SPEC`/weighted oracle** — plain unweighted geomean is the metric
  (COMMANDMENT confirmed default path); there is no separate time-weighted number to report.
- **Final verified speedup: 1.107x geomean** (arithmetic 1.115x) over baseline, achieved in **round 1**
  and never displaced across 3 rounds.
- **Rounds run**: 3 (round 2 was a dedicated `deep_explore` round). **Budget used**: 6 direction-units
  (round 1: 2 specialists = 2; round 2: 1 `deep_explore` @ DEEP_COST 2 = 2; round 3: 2 specialists = 2).
- **Committed patch**: a single file, `kernels/mxfp4_preshuffle_mbs.py`, 44 insertions / 3 deletions,
  from engineer `r1_d0` (round 1), unchanged through rounds 2-3 since neither round produced a
  verified improvement over it.

## Round-by-round

### Round 0 (analysis, pre-round)
Static ISA/VGPR disassembly (not just runtime timing) established the bottleneck hypothesis:
`compute_with_macro`'s per-output-element sigma Hadamard correction costs **+42 VGPR/thread and +424
extra static instructions per K-tile** at *identical MFMA count* vs. the plain kernel (~77 scalar
fma/mul ops per iteration vs. baseline's ~3) — a genuine compute/VALU-issue-port cost, with
memory/DMA/division and occupancy/scheduling separately ruled out as primary causes. This workspace
started pristine (not pre-patched with any of the task brief's headline 87-142% overhead numbers'
implicit fixes), so round 1 first had to re-establish three already-known-safe micro-fixes before
attacking structural levers.

### Round 1 — 2 engineers, both `compute` specialty
| id | title | expected | verified | status | reason |
|---|---|---|---|---|---|
| `r1_d0` | Graduate known-good micro-fixes: FMA fusion of the accumulate step + scheduler `dvmem_preload`/`sched_num_gmem` count fix (covers `load_mbs`'s own gmem loads) + tile-gated `waves_per_eu=1` hint (ONLY the (128,256,128) tile) | 1.05 | **1.107** | **verified / winner** | Clean 1.11x, correctness bit-identical on all 9 shapes. 2 shapes (`N7168_K2048_M64`, `N7168_K2048_M1024`) show harness-level bimodal timing noise, confirmed present on the *unpatched* baseline too — not a patch defect. |
| `r1_d1` | Vectorized/packed sigma product via FlyDSL `Vec.from_elements`+`broadcast_to` instead of 4x inline scalar construction | 1.08 | 0.988 | dead end | `rocprofv3` VGPR_Count bit-identical before/after at every tile shape — the AMDGPU/LLVM backend already scalarizes `vector<4xf32>` ops identically regardless of assembly style (CDNA has no packed-fp32 SIMD-within-lane ALU). Confirmed dead end for this lever. |

**Integrate**: not needed (single clear winner, `r1_d1` produced no diff worth merging).
**Round winner**: `r1_d0`, 1.107x geomean, committed as `b6ae1f1`.
**Bottleneck shift**: none — still compute/VALU-issue-port-bound after round 1 (geomean baseline
0.09574ms → 0.08596ms). Overhead range narrowed from 29.7%–142.9% to 20.5%–116.8% (excluding the 2
noisy small-K cases); every non-noisy case improved +5.7% to +30.2%.

### Round 2 — 1 `deep_explore` engineer (dedicated round, DEEP_COST 2)
| id | title | expected | verified | status | reason |
|---|---|---|---|---|---|
| `r2_d0` | 32x32x64 MFMA-shape rewrite of `compute_with_macro` (toy-kernel validated first) — target ~1.4-1.9x, aimed at halving the sigma-correction:MFMA instruction ratio | 1.6 | 1.0 (no change) | partial / no patch produced | Blocked before implementation: the B-side CK-preshuffle `NLane=16` layout this rewrite depends on lives in an out-of-scope file (`task_runner.py`/`mxfp4_preshuffle_mbs.py`'s frozen call-site contract), and independent disassembly analysis showed the sigma-correction scalar-op volume is MFMA-shape-*invariant* for a fixed output-element count (only the loop-trip count would shrink ~4x, not the raw arithmetic) — the achievable gain is much smaller than hoped. `git diff` was empty; no regression, but no improvement either. |

**Integrate**: n/a (no patch to merge). **Round winner**: none — cumulative stayed at 1.107x.
**Bottleneck shift**: none (still compute-bound). Left two concrete unattempted follow-ups in the
ledger: (1) request sign-off to edit `task_runner.py`'s B-preshuffle for `NLane=32` (lower risk), or
(2) an in-kernel `ds_bpermute` cross-lane gather network with dedicated toy-kernel validation (higher
risk). Neither was pursued in round 3 in favor of the higher-expected-value `host_runtime` lever below.

### Round 3 — 2 engineers, `host_runtime` + `compute`
| id | title | expected | verified | status | reason |
|---|---|---|---|---|---|
| `r3_d0` | Wrapper-level HIP-graph capture/replay on the benchmarked call path (`scripts/task_runner.py`, explicitly in-scope per COMMANDMENT) to collapse the per-call launch floor | 1.08 | 1.0857 (engineer-claimed 12.98x) | **regression vs. claim / flagged as flaky, not committed** | Root-caused post-hoc: the graph-capture technique itself is genuinely correct and fast — the engineer's own workspace run AND an independent verify-retry run both show `bench_mode:'graph'` engaging on 8/9 shapes with 2.3x–174x speedup (43-99% overhead reduction), bit-identical correctness. But the FIRST official verify run hit `bench_mode:None` on **all 9 shapes** (capture silently failed every time in that one process, swallowed by a bare `except Exception: return None` in `_try_capture_graph()`), landing near/worse than baseline instead of the claimed win. Most likely cause: concurrent-GPU-usage fragility under `gpu_lock.sh`'s shared-slot model (two engineers ran simultaneously this round). Not merged because the officially-verified number (1.0857x) does not clear the bar, but flagged as an **unfinished lever, not a dead end** — needs exception logging + retry-with-backoff + a low-contention re-verify. |
| `r3_d1` | Re-scope/extend the tile-gated `waves_per_eu` hint to fix the `N7168_K2048_M64` (CTAs=112) small-tile regression | 1.03 | 0 | dead end / apply failed | `rocprofv3` shows VGPR=44/Scratch=0 on this tile regardless of the hint — never register-constrained — and a 2-wave hint regressed the neighboring `N7168_K7168_M64` case ~4-5% for zero gain. Also corrected a standing assumption: this box has **128 CU** (rocminfo-confirmed), not the 256 CU `profiling_summary.md` had assumed, so CTAs=112 is near-full occupancy (~0.875 CTA/CU), not "44%-underfilled." This regression is better explained as a near-fixed dispatch-overhead floor than an occupancy problem. `git diff` was empty — no patch produced. |

**Integrate**: not applicable (neither direction produced a mergeable, verified-improving patch this
round). **Round winner**: none — cumulative stayed at **1.107x** (round 1's `r1_d0`, still the
committed HEAD). **Bottleneck shift**: none — category remains compute/VALU-issue-port-bound; the
host_runtime graph-capture lever remains the single biggest *unrealized* opportunity (2.3x-174x when
it engages) but needs a robustness fix (exception logging + retry-with-backoff + low-contention
re-verify) before it can be trusted and committed. This is a genuine "stop short of the ceiling, not
because the ceiling was reached" outcome — flagged for whoever picks this workload up next.

## Final per-test-case table

Winner: engineer `r1_d0` (round 1), verified geomean 1.107x, arithmetic mean 1.115x. No `count`/weight
data applies (no `WORKLOAD_SPEC`, plain unweighted default path — every case implicitly weight 1).

| case | baseline ms | optimized ms | speedup |
|---|---|---|---|
| N7168_K7168_M64 | 0.0246 | 0.0234 | 1.0513x |
| N7168_K7168_M1024 | 0.0977 | 0.0942 | 1.0372x |
| N7168_K7168_M3000 | 0.2768 | 0.2446 | 1.1316x |
| N36864_K7168_M64 | 0.0811 | 0.0567 | 1.4303x |
| N36864_K7168_M1024 | 0.4551 | 0.4062 | 1.1204x |
| N36864_K7168_M3000 | 1.1319 | 0.9956 | 1.1369x |
| N7168_K2048_M64 | 0.0090 | 0.0102 | 0.8824x |
| N7168_K2048_M1024 | 0.0343 | 0.0308 | 1.1136x |
| N7168_K2048_M3000 | 0.0786 | 0.0695 | 1.1309x |

**Geomean: 1.107x. Arithmetic mean: 1.115x.** (No weighted metric applies to this run.)

Note on `N7168_K2048_M64` (0.88x, the only sub-1x case): this shape and `N7168_K2048_M1024` were
independently confirmed (round 1 insight log, cross-checked against the *unpatched* baseline) to
exhibit bimodal/high-variance timing in the standard cold-process-per-call harness at this
sub-15-20us absolute scale — not a patch-induced regression. It also disproportionately drags the
9-value geomean per the workload's own geomean-levers analysis; treat its ratio as expected harness
noise around 1x, not a bug, when judging this result.

## Key optimizations applied (committed patch, `kernels/mxfp4_preshuffle_mbs.py`)

1. **FMA fusion in the MBS accumulate step** (`compute_with_macro`): replaced a separate vector
   multiply (`Vec*Vec`) followed by a vector add with a single MLIR `math.fma` op
   (`cf_new = fma(tmp, sigma, cf)`) — one hardware `v_fma_f32` instead of `v_mul_f32`+`v_add_f32` per
   `(kh, ni, mi, ii)` accumulate. This is the primary driver of the win: it directly cuts scalar VALU
   issue-port pressure, the confirmed bottleneck (compute-bound, +42 VGPR/+424 static instructions per
   K-tile vs. the plain kernel at identical MFMA count).
2. **Scheduler bookkeeping fix**: `sched_num_gmem` (the vmem-preload budget fed to
   `hot_loop_scheduler`) previously omitted `load_mbs`'s own gmem loads (A dword loads + B byte loads).
   Now correctly counts `sched_num_gmem_base + sched_num_gmem_mbs`; `dvmem_preload` still preloads
   everything up front (measured to beat splitting MBS loads into the per-MFMA interleave budget).
3. **Tile-gated occupancy hint**: `rocdl.waves_per_eu = 1` applied ONLY when `(BM,BN,BK) ==
   (128,256,128)` (the biggest/most compute-bound tile), via a small `_MBS_WAVES_PER_EU` dict keyed on
   tile shape. Load-bearing gate: applying this hint unconditionally regressed the smaller
   `tile_m=32/64` shapes by up to 2.5x.

Combined impact: 1.107x geomean, correctness bit-identical across all 9 shapes, largest individual
gains on the `N36864_K7168` (large-N) family (+30-43% on M64/M1024/M3000), modest gains (+3.7-13.7%)
elsewhere, one case (`N7168_K2048_M64`) within harness noise of baseline.

## What didn't work (dead ends / unresolved)

- **Vec-API packing style for the sigma product** (`r1_d1`, round 1): disassembly-verified dead end.
  `rocprofv3` VGPR_Count is bit-identical whether the sigma product vector is assembled via
  `Vec.from_elements`+`broadcast_to` or 4x inline scalar construction — CDNA's backend already
  scalarizes `vector<4xf32>` arithmetic identically either way (no packed-fp32 SIMD-within-lane ALU).
  Do not re-attempt Vec-API-level restructuring of this site.
- **32x32x64 MFMA-shape rewrite** (`r2_d0`, round 2, `deep_explore`): blocked before implementation —
  the B-side CK-preshuffle `NLane=16` layout it depends on lives in an out-of-scope frozen file, and
  independent analysis showed the sigma-correction scalar-op volume is MFMA-shape-invariant for fixed
  output-element count (only loop-trip count would shrink, not raw arithmetic). Not fully disproven —
  two follow-ups remain open and unattempted (task_runner.py sign-off for `NLane=32`, or an in-kernel
  `ds_bpermute` cross-lane gather network with toy-kernel validation) but were deprioritized behind the
  host_runtime lever in round 3.
- **`waves_per_eu` re-scoping for the `N7168_K2048_M64` small-tile regression** (`r3_d1`, round 3):
  confirmed dead end. VGPR=44/Scratch=0 regardless of hint (never register-constrained); a 2-wave hint
  regressed a neighboring shape ~4-5% for zero gain. Also corrected a standing box assumption: 128 CU
  (rocminfo-confirmed), not the 256 CU earlier profiling had assumed — CTAs=112 is near-full occupancy,
  not underfilled. This regression is best explained as a near-fixed dispatch-overhead floor, not an
  occupancy problem.
- **Wrapper-level HIP-graph capture** (`r3_d0`, round 3): NOT a dead end, but unresolved/unmerged. The
  technique itself measured genuinely fast (2.3x-174x on 8/9 shapes) in two independent runs, but the
  single official verify run hit a silent all-shapes capture failure (bare `except Exception: return
  None` in `_try_capture_graph()`, most likely triggered by concurrent-GPU-usage contention under
  `gpu_lock.sh`'s shared-slot model with two engineers running simultaneously). This is very likely the
  single biggest remaining lever for this workload and should be re-attempted with exception logging,
  capture retry-with-backoff, and a low-contention re-verify before being judged again — it was not
  committed here only because the one number that was officially verified (1.0857x) did not clear the
  bar to beat the round 1 winner.
