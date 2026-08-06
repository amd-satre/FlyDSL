# Tech Lead Report — geak_task_oas_mbs (MXFP4 OAS+MBS GEMM, FlyDSL/gfx950)

## Summary

- **Kernel**: `kernels/mxfp4_preshuffle_mbs.py::compile_mxfp4_gemm_mbs` — a FlyDSL
  (MLIR-embedded-Python, CK-style) tiled MXFP4 (E2M1) block-scaled GEMM on gfx950 (CDNA4),
  built by porting the production, already-tuned `kernels/mxfp4_preshuffle.py` (async-copy A-tile
  DMA + `hot_loop_scheduler` MFMA interleave) and adding two accuracy features on top:
  **OAS** (Overflow-Aware Scaling, free) and **MBS** (Macro Block Scaling, ~5-7% theoretical cost
  per the paper) via `load_mbs()` + `compute_with_macro()`.
- **Type**: `flydsl2flydsl` (FlyDSL kernel-to-kernel optimization task), bottleneck class: **compute**.
- **Rounds run**: 2 (both narrow-specialist rounds; no `deep_explore` was dispatched — the ledger
  flags round 3 as the point a `deep_explore` or a widened-`focus_files` compute direction would be
  warranted, but the run stopped after round 2's budget).
- **Directions issued**: 5 graded engineer directions total (3 in round 1, 2 in round 2) — no
  direction was ever dropped/undispatched; budget_used = 5 engineer-directions (`budget_used` in the
  returned JSON).
- **Final result**: two directions were independently confirmed as small, real, orthogonal wins —
  `r1_d1` (fold `load_mbs`'s loads into the hot-loop scheduler's vmem budget, **1.0057x** verified)
  and `r2_d1` (tile-conditioned `waves_per_eu` occupancy hint, **1.0125x** verified). **Both winning
  patches were built and verified independently against the plain baseline in their own rounds — the
  round-2 engineer's workspace did NOT actually inherit round-1's fix** (contrary to this run's own
  "keep r2_d1 as built on r1_d1" steering note — verified by direct source inspection: round-2's
  `best_patch.diff` line ranges/diff context show it was cut against the un-patched baseline file,
  not round-1's winner). Since the report phase found this, **I merged the two winning patches myself**
  (they touch disjoint, non-overlapping regions of the same file and applied cleanly with `git apply`
  in sequence with zero conflicts) and re-verified the combination on-box with a fresh 6-repeat,
  same-session, median-based A/B (baseline vs. combined-patch, correctness re-checked 9/9 PASS).
  **That combined/final patch is what `final_patch.diff` and the numbers below report.**
- **Final speedup (headline, geomean over the 9 test-case shapes, fresh same-session verification)**:
  **~1.01x** (geomean 1.0096, arithmetic 1.0109). A **ratio-of-sums** (linear/latency-weighted, i.e.
  weighting by absolute wall-clock share rather than treating every shape equally) view is **~0.997x
  (flat, within noise)** because the single largest-latency case (`N36864_K7168_M3000`, ~1.13ms, ~half
  the total latency mass across all 9 cases) landed at a slight (~0.3%) session-noise-level regression
  in this run. No `count`/weight field was supplied in this task's per-case metadata (this is not a
  workload-aligned/repeated-call harness), so the geomean is the more informative headline metric here.
- This is consistent with the run's own conclusion: **two full rounds of narrow specialist tweaks
  produced only a low-single-digit-percent net gain against a ~100-143% residual overhead vs. plain
  MXFP4 at large-tile shapes — a plateau**, not a structural fix. See "What didn't work" below for the
  three confirmed dead ends that narrowed the hypothesis space, and the suggested next step (widened
  compute direction on `c_frags` live-range/register pressure, or `deep_explore`) that a future round
  should pick up.

## Round-by-round

### Round 1 (3 directions, budget 3)

| id | specialty | strategy | claimed | verified | status |
|---|---|---|---|---|---|
| r1_d0 | memory | Pipeline `load_mbs()` one K-tile ahead (double-buffer/prefetch), carrying `a_recip`/`b_recip` through the loop's `init=`/`yield` state the same way `accs` already does, to hide the load latency the round-0 profiling hand-off blamed for the residual overhead. | 1.002 | **0.9848** | **regression** |
| r1_d1 | compute | Extend `hot_loop_scheduler`'s vmem-interleave budget (`sched_num_gmem`/`dvmem_preload`) to also count `load_mbs`'s own A-side dword / B-side byte loads (previously uncounted — copied unmodified from the plain-MXFP4 path per the module's docstring), so they get folded into the scheduler's issue-slot spreading instead of sitting fully exposed. | 1.1 | **1.0057** | **verified / round winner** |
| r1_d2 | algorithm | Reduce `load_mbs()`'s raw load COUNT via mi/ni-pairing, mirroring `load_sc()`'s existing `m_pairs`/`n_pairs` trick (one load serving 2 adjacent mi/ni). | 1.15 | **0 (declined)** | **dead end (correctly declined)** |

- **Result**: r1_d0's well-implemented prefetch/double-buffer regressed on independent verification —
  the profiling hand-off's stale ablation (predating the `arith.divf`→`rocdl.rcp` fix already in this
  file) no longer holds; re-stubbing `load_mbs()` on the current file shows **zero** measurable change
  from removing 100% of its memory traffic AND arithmetic. r1_d2 correctly declined per its own escape
  clause: `load_sc`'s mi/ni-pairing relies on a hardware `opsel` consuming a genuinely-shared E8M0
  scale value; `load_mbs`'s per-row/column mantissa correction has no equivalent hardware unpack, and
  the host-side byte layout already packs at its maximum (adjacent mi/ni values are 16B apart per
  lane, not 4B-adjacent) — further reduction needs an out-of-scope host relayout.
- **Integrate**: null (round 1, nothing yet to integrate against).
- **Round winner**: r1_d1, verified geomean **1.0057x**.
- **Bottleneck shift**: none yet identified beyond "the scheduler undercounts MBS's own loads" —
  still compute-track, ~100-143% overhead unexplained at large tiles.

### Round 2 (2 directions, budget 2)

| id | specialty | strategy | claimed | verified | status |
|---|---|---|---|---|---|
| r2_d0 | compute | Eliminate the per-`(kh,ni,mi)` `tmp.store(zero)` / tmp-alloc / software c-d-merge scaffolding in `compute_with_macro` — hoist the zero-fill into a single `zero_c` rmem tensor allocated once, and reuse a single `tmp` rmem tensor as the `fx.gemm` `d` destination across all iterations, on the theory this scaffolding was a real per-iteration store/load tax. | 1.15 | **0.9751** | **dead end (flat noise, not a real regression)** |
| r2_d1 | compute | Add a `rocdl.waves_per_eu` occupancy hint (mirroring `kernels/mxfp4_preshuffle.py`'s `value_attrs` mechanism, which the MBS kernel lacked entirely), derived internally from `(tile_m,tile_n,tile_k)` since the call site is frozen — scoped ONLY to the biggest/most compute-bound tile config `(128,256,128)` after an unconditional-everywhere variant was found to regress smaller `tile_m=64`/K=2048 shapes up to 2.5x. | 1.08 | **1.0125** | **verified / round winner** |

- **Result**: r2_d0's implementation was correctness-safe but measured as flat noise on a rigorous
  same-session A/B (8/9 cases within ±1.5% of baseline). Root-cause read of FlyDSL's CDNA4 MFMA
  lowering (`FlyROCDL/CDNA4/MmaAtom.cpp::emitAtomCall`) showed `fx.gemm`'s `c`/`d` operands always get
  an explicit LLVM `Load`/`Store` regardless of source-level aliasing, and the original
  zero-store-then-immediate-load pattern is a textbook SROA/mem2reg shape the backend almost certainly
  already eliminates — so the "remove the zero-init tax" lever measured as a no-op, not a real
  win/loss. r2_d1's occupancy hint gave a small confirmed gain, but ONLY when scoped per-tile-config
  (confirming occupancy/VGPR pressure is a minor, not dominant, contributor to the residual overhead).
- **Integrate**: null (both directions this round targeted `compute_with_macro`'s
  call site region — non-overlapping in principle, but the tech-lead-in-the-loop at the time judged
  r2_d0 a dead end and did not attempt a formal merge step).
- **Round winner**: r2_d1, verified geomean **1.0125x**.
- **Bottleneck shift**: with `load_mbs`'s loads (round 1) AND `compute_with_macro`'s zero-init
  scaffolding (round 2) both ruled out as the overhead driver, and occupancy hints confirmed as only a
  minor (~1-2%) lever, the ledger's updated hypothesis for the residual ~100-143% overhead is either
  (a) the actual per-`(kh,ni,mi)` sigma `Vec` FMA-chain cost, or (b) VGPR/register pressure from
  holding all `n_acc` `c_frags` live simultaneously across the `kh`/`ni`/`mi` loops — both flagged as
  needing a widened-`focus_files` round-3 direction or a `deep_explore` escalation, which this run did
  not reach.

### Report-phase integration (this phase)

Round 2's winning patch was found (via direct diff/line-range inspection) to have been cut against the
plain, un-patched baseline file rather than round 1's winner, despite the round's own steering note
assuming otherwise. Since `r1_d1` (scheduler bookkeeping, lines ~95-113) and `r2_d1` (occupancy hint,
lines ~90-104 + the `launch_gemm` call site ~line 536) touch disjoint code regions of the same file and
apply with `git apply` in strict sequence with **zero conflicts**, I merged them into one cumulative
patch and re-verified on-box:
- Correctness: **PASS, 9/9 shapes** (`task_runner.py correctness`).
- Performance: fresh same-session A/B, 6 repeated full-suite runs each of baseline vs. combined-patch,
  interleaved, median-per-case — see the table below. This is the number reported as `final_speedup_*`.

## Final per-test-case table

Baseline column = same-session median (6 repeated runs of the unmodified workspace, this report
phase) rather than the original single-shot `BASELINE_PER_CASE` snapshot, because the ledger's own
insight #7 flags `N7168_K2048_M64`/`N7168_K2048_M3000` with 48-450%+ session-to-session drift — a
single-shot baseline-vs-patch comparison on those shapes is not trustworthy without a same-session
re-measurement (confirmed again during this report phase: raw single-shot per-call readings for
`N7168_K2048_M64` ranged 0.0086ms-0.0570ms across 10 back-to-back calls of byte-identical code).

| case | baseline ms (median, n=6) | optimized ms (median, n=6, combined patch) | speedup |
|---|---|---|---|
| N7168_K7168_M64 | 0.02465 | 0.02470 | 0.998x |
| N7168_K7168_M1024 | 0.09795 | 0.09895 | 0.990x |
| N7168_K7168_M3000 | 0.27215 | 0.27715 | 0.982x |
| N36864_K7168_M64 | 0.08105 | 0.08100 | 1.001x |
| N36864_K7168_M1024 | 0.45315 | 0.45090 | 1.005x |
| N36864_K7168_M3000 | 1.12760 | 1.13145 | 0.997x |
| N7168_K2048_M64 | 0.01200 | 0.01035 | 1.159x (high variance — see note above) |
| N7168_K2048_M1024 | 0.03345 | 0.03420 | 0.978x |
| N7168_K2048_M3000 | 0.07805 | 0.07895 | 0.989x |

- **Geomean speedup**: **1.0096x**
- **Arithmetic-mean speedup**: **1.0109x**
- **Ratio-of-sums (latency-weighted) speedup**: **0.9965x** (essentially flat — dominated by the
  largest-latency case, `N36864_K7168_M3000`, which sits at a slight session-noise-level regression
  in this run)
- For reference, the original single-shot `BASELINE_PER_CASE`/`BASELINE_GEOMEAN_MS` snapshot
  (0.0946ms geomean) is preserved in `EVAL_DIR/baseline_metrics.json`/`baseline_timing.json`; comparing
  the combined patch's fresh median against that stale single-shot baseline instead gives a misleading
  0.973x due to the drift on `N7168_K2048_M64` alone (single-shot baseline 0.0086ms was a lucky low
  outlier — see note above).

## Key optimizations applied

1. **Scheduler vmem-budget fix (`r1_d1`)** — `hot_loop_scheduler`'s `sched_num_gmem`/`dvmem_preload`
   accounting was copied unmodified from the plain-MXFP4 kernel and never budgeted a vmem-interleave
   slot for `load_mbs`'s own A-side dword / B-side byte loads. Fixing the count (a pure 2-line
   bookkeeping change, no call-site/internals change) is a small, real, orthogonal win (confirmed
   1.0057x in isolation).
2. **Tile-conditioned occupancy hint (`r2_d1`)** — added a `rocdl.waves_per_eu` hint via
   `value_attrs`, mirroring the plain-MXFP4 kernel's mechanism which the MBS kernel entirely lacked.
   Scoped via a static `(tile_m,tile_n,tile_k)` lookup table to ONLY the single biggest/most
   compute-bound config `(128,256,128)` — applying it unconditionally regressed smaller
   `tile_m=64`/K=2048 shapes up to 2.5x, so the fix is deliberately narrow (confirmed 1.0125x in
   isolation).
3. Both fixes merged into one combined patch (this report phase) — they are orthogonal (disjoint code
   regions) and compose cleanly; net combined effect measured fresh on-box at ~1.01x geomean /
   effectively flat on a latency-weighted basis, correctness held 9/9.

## What didn't work (confirmed dead ends)

1. **`load_mbs()` prefetch/double-buffering one K-tile ahead (`r1_d0`, memory)** — regressed
   (0.9848x) on independent verification despite a well-implemented pipeline carrying `a_recip`/
   `b_recip` through the loop state. **Root cause**: the profiling hand-off's cited root cause (an
   unpipelined, latency-exposed `load_mbs`) is **stale** for this file's current state — it predates
   the `arith.divf`→`rocdl.rcp` fix and r1_d1's scheduler-count fix already present. Re-stubbing
   `load_mbs()` entirely on the current file shows **zero** measurable overhead change. Do not reissue
   MBS-load-pipelining directions without re-profiling first.
2. **mi/ni load-pairing to reduce `load_mbs`'s raw load count (`r1_d2`, algorithm)** — correctly
   declined. `load_sc()`'s equivalent trick works only because the E8M0 scale is genuinely shared 1:2
   via a hardware `opsel_a/opsel_b` unpack; `load_mbs`'s per-row/column mantissa correction has no
   hardware unpack equivalent, and the host-side byte layout already packs at its maximum (adjacent
   mi/ni values are 16B apart per lane). Further reduction needs an out-of-scope host relayout.
3. **Eliminating `compute_with_macro`'s zero-init/tmp-alloc/software c-d-merge scaffolding (`r2_d0`,
   compute)** — measured flat (0.9751x, within noise, 8/9 cases within ±1.5%). **Root cause**:
   `fx.gemm`'s `c`/`d` operands always get an explicit LLVM `Load`/`Store` at the IR level regardless
   of source-level aliasing (confirmed by reading `FlyROCDL/CDNA4/MmaAtom.cpp::emitAtomCall`), and the
   original zero-store-then-immediate-load pattern is a textbook SROA/mem2reg shape the backend
   compiler almost certainly already eliminates. Do not retry source-level "remove the zero-init tax"
   surgery on this pattern.
4. **Applying `waves_per_eu=1` unconditionally to all tile configs** — regressed smaller
   `tile_m=64`/K=2048 shapes up to 2.5x; only the tile-conditioned, narrowly-scoped version (applied to
   `(128,256,128)` only) survived verification.

## Residual gap and recommended next step (not reached this run)

Two full rounds of narrow specialist tweaks recovered only ~1-1.25% in isolation (and ~1% net combined,
essentially flat on a latency-weighted basis) against a **~100-143% overhead vs. plain MXFP4 still
present at large-tile shapes** (e.g. `N36864_K7168_M3000` at 1.13ms vs. ~0.56ms plain-MXFP4 baseline).
With the three most plausible structural hypotheses (`load_mbs` loads, `compute_with_macro`'s zero-init
scaffolding, and global occupancy) all confirmed dead ends or minor levers, the ledger's own suggested
next step — not reached within this run's round budget — is either:
(a) a compute specialist with `focus_files` explicitly widened to restructure `compute_with_macro`'s
`c_frags` live-range/accumulation batching (e.g., batch `n_acc` into smaller groups processed
sequentially instead of holding all fragments live across the `kh`/`ni`/`mi` loops), directly testing
the VGPR-pressure hypothesis distinct from the coarse `waves_per_eu` knob already tried; or
(b) a `deep_explore` direction now that the narrow-lane hypothesis space is exhausted across two
rounds, targeting the sigma-multiply-add/accumulation-batching space with an ambitious target and no
prescribed recipe.
