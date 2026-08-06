# Tech Lead Report — geak_task_oas_mbs (MXFP4 OAS+MBS preshuffled GEMM, FlyDSL/gfx950)

## Summary

- **Kernel**: `kernels/mxfp4_preshuffle_mbs.py::compile_mxfp4_gemm_mbs` — an MXFP4 preshuffled GEMM
  extended with OAS (outlier-aware scaling) + MBS (macro-block scaling) numerics on AMD Instinct
  MI355X (gfx950/CDNA4). Entry point emits a FlyDSL-compiled `launch_gemm` used across 9
  (N,K,M) shapes.
- **Type**: FlyDSL (HIP-generation DSL), compute-bound kernel; 1 dispatch/call (no host-overhead
  floor issue — ruled out by profiling).
- **Final speedup**: **1.00x geomean** (no net change vs. this run's starting baseline).
  Arithmetic-mean speedup: 1.00x. This task is NOT workload-aligned (no `count`/weight fields in the
  per-case table), so geomean is the headline metric; there is no separate weighted metric to report.
- **Rounds run**: 2 (+1 diagnostic-only deep_explore round counted as round 1). **Budget used**: 3
  engineer-directions total (1 in round 1, 2 in round 2) out of the allotted budget; the run ended
  with `CUMULATIVE_SPEEDUP=1` and `FINAL_WINNER=null` — i.e. no direction this run was accepted as a
  new committed baseline.
- **Context**: the workspace entered this run already carrying a *prior run's* accepted micro-fix
  (scheduler vmem-budget count correction + tile-conditioned `waves_per_eu` hint, commit `13158ae`,
  itself only ~0.978-1.01x vs. the pristine kernel — essentially noise-level). All of this run's own
  rounds tried to build on top of that starting point and none beat it by a confirmed margin, so the
  final committed kernel is byte-for-byte identical to what this run started with.

## Round-by-round

### Round 1 — `deep_explore` (ISA/VGPR ground-truth + sigma-FMA root-cause), 1 direction, budget cost 2 (DEEP_COST)

| id | specialty | strategy | verified speedup | result |
|---|---|---|---|---|
| r1_d0 | deep_explore | Ground-truth ISA/VGPR diagnostic of `compute_with_macro` vs. baseline `compute()` at identical MFMA count; look for a numerics-safe fix to the sigma-Hadamard correction's VALU cost | ~1.00x (diagnostic-only, no code change) | **partial** — high-value root-cause finding, zero delivered speedup |

- **Finding**: disassembly confirmed the ~99-144% overhead vs. plain MXFP4 is a genuine VALU-issue-port
  cost: +42 VGPR/thread and +424 extra static instructions per K-tile iteration at the *same* MFMA
  instruction count, driven by the per-output-element sigma Hadamard correction (baseline ~3 fma/mul
  per iter vs. MBS's ~77+). This ISA-level result **retired two standing hypotheses as confirmed
  dead-ends via direct measurement**: occupancy (`waves_per_eu` 1↔2 — no change) and instruction
  scheduling (manual hints + 4 depths of SW pipelining — no change).
- **Integrate**: none (diagnostic only; git diff vs. round-1-2-merged baseline was empty).
- **Round winner**: none (no code change).
- **Bottleneck shift**: confirmed compute/VALU-issue-bound on the sigma-FMA chain in
  `compute_with_macro`; ruled out occupancy and scheduling as levers.

### Round 2 — 2 directions (compute + memory), budget cost 2

| id | specialty | strategy | verified speedup | result |
|---|---|---|---|---|
| r2_d0 | compute | Tighten sigma-apply arithmetic + register lifetime in `compute_with_macro` | 1.001x (officially verified) / 1.126-1.127x (engineer's own local A/B, see caveat below) | **partial / unresolved** |
| r2_d1 | memory | Move `_byte_to_recip` from VALU (`v_rcp_f32`) to an LDS-indexed lookup table | ~0.997x (noise) | **dead_end / failed** |

- **r2_d0 result and caveat**: the engineer discovered and diagnosed a **verification-infrastructure
  bug** in this task's harness: `scripts/task_runner.py` runs `sys.path.insert(0, TASK_DIR)`
  immediately followed by `sys.path.insert(0, FLYDSL_ROOT)`; because each `insert(0, ...)` pushes to
  the front, the net search order is `[FLYDSL_ROOT, TASK_DIR, ...]`, so the shared box-wide
  `/scratch/satre/FlyDSL/kernels/mxfp4_preshuffle_mbs.py` **shadows** the engineer's edited workspace
  copy at import time. The engineer reproduced this with an import-canary print (never fired) and a
  hard `SyntaxError` injection into the workspace copy (correctness still reported PASS) — i.e. the
  standard COMMANDMENT correctness/perf commands can silently verify the *wrong, unmodified* kernel.
  The engineer's own local A/B with a temporary path-order workaround (reverted before generating
  `best_patch.diff`, so the submitted patch is clean) measured **1.126-1.127x median across 20
  runs/2 batches**, correctness PASS throughout — well outside noise — while the *officially*
  re-verified number under the (buggy) standard harness came back at 1.001x, matching exactly the
  "silently re-measures the unmodified seed kernel" signature. **This direction was therefore treated
  as unresolved, not disproven**, and flagged for a re-verify under corrected `sys.path` ordering.
  No round-3 was run in this budget window to perform that re-verify, so the patch was **not**
  promoted to the committed baseline under the official-verification gate (1.001x measured gain is
  not distinguishable from noise) — this is the single biggest open item from this run (see below).
- **r2_d1 result**: numerically exact, ISA-visible reduction in static `v_rcp_f32` instruction count,
  but zero measurable wall-clock change on any of the 9 shapes, including the large-N/high-M cases.
  This is the **third** confirmed dead-end for issue-port-shifting compute tweaks on this kernel
  (after occupancy retune and scheduling hints/pipelining), reinforcing that the bottleneck is the
  sigma-FMA/Vec accumulate chain's latency/critical-path, not issue pressure at any smaller site
  tried so far.
- **Integrate**: not attempted — neither direction cleared the bar to become a new baseline (r2_d0's
  *official* number was noise-level pending re-verify; r2_d1 was a clean dead-end).
- **Round winner**: none promoted (r2_d0 nominally "won" the round at 1.001x officially verified, but
  the tech lead/ledger explicitly flags this as suspect and unresolved rather than a real win — hence
  `FINAL_WINNER=null` / `CUMULATIVE_SPEEDUP=1` for the run).
- **Bottleneck shift**: unchanged — still compute/latency-bound on the sigma-FMA accumulate chain.
  Round's real headline is the infra bug that must be fixed before any further kernel-only patch on
  this task can be trusted at face value.

## Final per-test-case table

No new patch was accepted this run, so the final committed kernel is unchanged from the run's
starting baseline (itself a prior run's carried-forward micro-fix). All speedups are 1.00x by
construction; "optimized ms" below is the same measurement as "baseline ms" (this run's own
baseline_metrics.json, re-confirmed by round 1/2 engineers' independent performance_report.json runs
within measurement noise, e.g. N7168_K7168_M64: 0.0242 vs. 0.0235-0.0241 ms across runs).

| case | N | K | M | baseline ms | optimized ms | speedup |
|---|---|---|---|---|---|---|
| N7168_K7168_M64 | 7168 | 7168 | 64 | 0.0242 | 0.0242 | 1.00x |
| N7168_K7168_M1024 | 7168 | 7168 | 1024 | 0.1007 | 0.1007 | 1.00x |
| N7168_K7168_M3000 | 7168 | 7168 | 3000 | 0.2775 | 0.2775 | 1.00x |
| N36864_K7168_M64 | 36864 | 7168 | 64 | 0.0810 | 0.0810 | 1.00x |
| N36864_K7168_M1024 | 36864 | 7168 | 1024 | 0.4552 | 0.4552 | 1.00x |
| N36864_K7168_M3000 | 36864 | 7168 | 3000 | 1.1407 | 1.1407 | 1.00x |
| N7168_K2048_M64 | 7168 | 2048 | 64 | 0.0112 | 0.0112 | 1.00x |
| N7168_K2048_M1024 | 7168 | 2048 | 1024 | 0.0324 | 0.0324 | 1.00x |
| N7168_K2048_M3000 | 7168 | 2048 | 3000 | 0.0737 | 0.0737 | 1.00x |

**Geomean speedup: 1.00x** (baseline geomean 0.0970 ms, optimized geomean 0.0970 ms)
**Arithmetic-mean speedup: 1.00x** (baseline mean 0.2441 ms, optimized mean 0.2441 ms)
**Weighted speedup: n/a** (task is not workload-aligned; no per-case `count`/weight data supplied)

For reference, every case remains well above the plain-MXFP4-without-OAS/MBS baseline
(`baseline_mxfp4_ms_median` in the per-case params), by +13.8% to +144.3% depending on shape — this
overhead is the OAS/MBS numerics tax that all rounds this run tried and failed to meaningfully cut.

## Key optimizations applied

**None net-new this run.** The only optimization present in the final committed kernel is inherited
from a prior run (commit `13158ae`, pre-dating this run's round 1): a scheduler vmem-budget
bookkeeping fix (counting `load_mbs`'s A-dword/B-byte gmem loads in `sched_num_gmem` so
`hot_loop_scheduler`'s interleave budget accounts for them) plus a tile-conditioned
`rocdl.waves_per_eu` occupancy hint (`{(32,128,256): None, (64,128,256): None, (128,256,128): 1}`).
That fix measured only ~0.978-1.01x (noise-level) even in its own originating run, and this run's
diagnostics (round 1 ISA ground-truth) independently confirmed occupancy retuning has zero effect on
this kernel — so it should be considered inert/neutral rather than a real win, kept only because it
is harmless and already the starting point.

## What didn't work (dead-ends from the ledger)

1. **Occupancy retuning** (`waves_per_eu` 1↔2, round 1 diagnostic + carried from a prior run) — no
   measurable change. The kernel is VALU-issue-bound, not occupancy-bound. Do not retry plain
   occupancy knobs without also cutting VALU volume.
2. **Instruction scheduling / SW pipelining** (manual `sched_group_barrier` hints on/off, 4 pipeline
   depths, round 1 diagnostic + carried) — zero effect. The default scheduler is already at parity
   with manual hints on this kernel. Do not retry scheduling-only directions.
3. **LDS lookup table for `_byte_to_recip`** (r2_d1, memory specialty) — numerically exact,
   ISA-visible reduction in `v_rcp_f32` count, but zero wall-clock change on all 9 shapes. Third
   confirmed dead-end for issue-port-shifting tweaks; the real bottleneck is almost certainly the
   sigma-FMA/Vec accumulate chain's *latency*, not issue-port *pressure*, at any of the smaller sites
   tried (occupancy, scheduling, byte-to-recip relocation).
4. **Sigma-apply arithmetic + register lifetime tightening** (r2_d0, compute specialty) — **not a
   confirmed dead-end**, but ended the run unresolved: officially verified at 1.001x (noise), which
   the engineer's own diagnosis attributes to a `scripts/task_runner.py` `sys.path` ordering bug
   (`insert(0, TASK_DIR)` then `insert(0, FLYDSL_ROOT)` nets `[FLYDSL_ROOT, TASK_DIR, ...]`, so the
   shared `/scratch/satre/FlyDSL/kernels/mxfp4_preshuffle_mbs.py` shadows the workspace copy at
   import time — reproduced with an import canary and a `SyntaxError` injection that still reported
   correctness PASS). The engineer's own local A/B with the path order corrected (workaround reverted
   before submitting `best_patch.diff`) measured 1.126-1.127x median, correctness PASS. **This is the
   single most actionable open item**: fix the `sys.path` insertion order (or sync `FLYDSL_ROOT`'s
   copy) in `scripts/task_runner.py`, then re-verify `round_2/engineer_0/best_patch.diff` under the
   corrected harness before either committing it (~1.13x, the best candidate seen this run) or
   retiring it as a dead end.
5. **32x32x64 MFMA-shape restructuring of `compute_with_macro`** and **vectorized/packed sigma
   computation reused across output elements** — identified as the two remaining levers with a
   plausible path to closing the residual VALU-issue-bound gap, but neither was attempted this run
   (round 1 judged the MFMA-shape rewrite too high-risk to try directly without first validating the
   operand/accumulator mapping on a throwaway toy kernel; that validation step was never scheduled).
   Recommended as the primary direction for any follow-on run, once the verification-infra bug above
   is fixed.

## Bottom line

This run made **no net progress** (1.00x geomean) versus its own starting point, but it delivered two
things of real value for any follow-on attempt: (1) a confirmed, ISA-level root cause for the
OAS/MBS overhead (VALU-issue-port cost of the per-output-element sigma Hadamard correction in
`compute_with_macro`, not occupancy/scheduling), which retires three dead-end lever classes; and (2) a
diagnosed verification-infrastructure bug (`scripts/task_runner.py` `sys.path` shadow-import) that
likely masked a genuine ~1.13x win (`round_2/engineer_0/best_patch.diff`) as a false ~1.0x no-op. The
highest-leverage next step is not a new kernel direction but fixing the harness bug and re-verifying
that already-written patch.
