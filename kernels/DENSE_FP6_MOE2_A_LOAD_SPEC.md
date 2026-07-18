# Dense fp6 A-load — stage-2 (moe2) implementation spec

Target: `mixed_moe_gemm_2stage.py::compile_mixed_moe_gemm2`, gated on `dense_fp6=True`
(already plumbed). Goal: A-load pulls dense 24 B/K=32-block (K*3//4 B/row, 1.5× fp4)
instead of FP8-padded 32 B/block (2.0× fp4). **Layout-only → must be bit-exact.**

## Invariant (validated by check_dense_moe.py)
`MoE_out(SGLANG_W4A6_DENSE=1) == MoE_out(padded)` exactly. Default off = padded, untouched.

## What stays THE SAME (do not change)
- LDS tile layout: still 32 B/block. `col_offset_base = lane_div_16 * a_per_lane_kpack_bytes(32)`.
- The MFMA reads packed fp6 from the low 24 B of each 32 B LDS slot (`cbsz=2`); the 8 pad
  bytes in LDS are NEVER read → leave them as garbage (no need to zero).
- B load, scales, sort, epilogue, accumulate — all unchanged.
- The a2 quant already emits dense 24 B/block when `dense=True` (host wiring done).

## What changes (only the HBM→LDS A path), when dense_fp6:
1. **Buffer size** (`x_nbytes` / `a_rsrc` num_records): `(tokens*topk) * (k * 3 // 4)` bytes
   (was `(tokens*topk) * k * elem_bytes / a_elem_vec_pack`).
2. **Row stride** `c_k_div4`: `(k * 3 // 4) // 4` dwords/row (was `k//4` for fp6 padded).
3. **Load loop** `load_x_tile` — the crux. The 16 B dword-chunk mapping cannot align to
   24 B dense blocks, so restructure to **per-K=32-block loads**:
   - Assign blocks to threads: `total_blocks = tile_m * (tile_k//32)`. Full mode
     (`total_blocks % total_threads == 0`): each thread does `total_blocks//total_threads`
     blocks. Partial mode (`total_blocks <= total_threads`, e.g. tile_m=16): threads
     `[0,total_blocks)` do 1 block, rest idle, store guarded by `scf.if tx < total_blocks`.
     (Mirror `preshuffle_gemm_a6w4.py` dense_fp6 block-assignment exactly.)
   - Per block: dense HBM offset = `row * (k*3//4) + kblock * 24` bytes; load 24 B via
     **3× `buffer_load_dwordx2`** (3×8 B) with OOB protection from `x_rsrc`.
   - Row decode (sorted token→`t*topk+s`) is unchanged; only the byte offset math is dense.
4. **LDS write**: write the 24 real bytes into the low 24 B of the block's 32 B LDS slot
   (same slot address as padded; just don't touch the top 8 B).

## Reference to copy from (bit-for-bit logic, adapt indices to gemm2)
`mxfp6_experiments/src/flydsl/preshuffle_gemm_a6w4.py` — the `dense_fp6` path:
`_a_nrec_row_bytes = K*3//4`, the partial/full block-assignment validation, the
`3×buffer_load_dwordx2` per block, and the LDS-expand. gemm2 differs only in that A rows
come via `sorted_token_ids` (t*topk+s) rather than a plain row index.

## Test loop (each iteration)
```
VLLM_FLYDSL_REPO=.../third_party/FlyDSL VLLM_FLYDSL_FP6_QUANT_REPO=.../third_party/mxfp6_experiments \
SGLANG_W4A6_REQUIRE_KERNEL=1 HIP_VISIBLE_DEVICES=0 \
python .w4a6_logs/check_dense_moe.py 256 2048
# PASS = bit-exact True AND dense_us < padded_us. moe2 K=inter_dim=2048 (K%256==0 ✓).
```

## Baseline to beat (invoked kernel, padded, full moe1+moe2+quant+sort)
M=256: 284.8 µs · M=2048: 1149.2 µs. Dense targets the moe2 A-load (~1.5× vs 2.0× fp4);
expect the moe2 component to drop; confirm net full-MoE speedup + bit-exact.

## CRITICAL implementation constraint (discovered profiling the load path)
gemm2's A-LDS is **XOR16-swizzled** (`store_x_tile_to_lds` -> `lds_store_16b_xor16`
/ `swizzle_xor16`; read back by `lds_load_packs_k64`). And the padded load maps threads
to 16B dword-chunks whose real(24)/pad(8) split within a K=32 block is a **runtime**
offset. Consequences:
- A cheap "load 24B, write low-24B of a linear 32B slot" does NOT work (LDS isn't linear).
- Assembling padded 16B chunks from dense in-registers does NOT work cleanly (per-chunk
  real/pad split is runtime, not constexpr).
- REQUIRED approach: restructure the dense load to **per-K=32-block ownership** (each
  thread owns whole 24B blocks, statically), load 24B via 3x buffer_load_dwordx2, then
  store into the swizzled LDS so `lds_load_packs_k64` reads it identically to padded.
  Mirror `preshuffle_gemm_a6w4.py` dense_fp6 block-assignment, but adapt the LDS store to
  gemm2's XOR16 swizzle (the standalone may use a simpler LDS layout — verify + adapt).
This is the crux; it needs the iterative compile->check_dense_moe.py(bit-exact)->bench loop.

## Microbench context (why this is worth it) — recorded 2026-07-17
Gap is entirely large-M (prefill), A-load-bound: ours/mxfp4 = 0.97x @256, 1.18x @512,
1.57x @1024, 1.95x @2048. Dense (1.5x vs 2.0x fp4 A-load) targets exactly this regime.

## BLOCKER discovered (cycle 1-3, empirical) — gemm2 is DMA-to-LDS, explicit store is STALE
- The dense QUANT is verified correct: dense codes' 24B/block == padded codes' low-24B (bit-exact).
- gemm2's SERVING config loads A via `dma_x_tile_to_lds` (raw_ptr_buffer_load_lds): it applies
  swizzle_xor16 to the HBM READ address and writes LDS lane-sequentially. The reader
  `lds_load_packs_k64` is co-designed with THIS DMA layout.
- The explicit `store_x_tile_to_lds` (which the standalone's dense is built on) is UNUSED in
  gemm2's DMA config and is STALE: forcing it (SGLANG_W4A6_FORCE_EXPLICIT=1) with the DMA-mode
  reader gives WRONG output (explicit-padded != DMA-padded, max_abs 131072). So building dense
  on the explicit store writes an LDS layout the DMA-mode reader can't read -> bit-exact FALSE
  (observed: 98304/32768 diffs).
- Therefore dense in gemm2 needs ONE of:
  (A) match the DMA path's exact LDS layout with dense-sourced explicit stores (replicate
      raw_ptr_buffer_load_lds's per-lane placement + the HBM-side swizzle, adapted to 24B blocks), OR
  (B) fix gemm2's explicit store+reader to be a matched pair (like the standalone), select it
      for dense (override BOTH store AND lds_load_packs_k64), then the standalone dense ports cleanly.
- (B) is likely cleaner: override store_x_tile_to_lds (dense) AND lds_load_packs_k64 to the
  standalone's matched swizzle scheme, all gated on dense_fp6. Needs the standalone's reader logic.
- Env-gated scaffolding is committed-quality and OFF by default (padded serving untouched).

## ROOT CAUSE after 5 debug cycles (the real co-design blocker)
Verified: dense quant is 100% correct (codes AND scales bit-exact vs padded). Bug is purely
the kernel's dense A LDS placement. Tried: lds_store_16b_xor16, then direct crd2idx mirroring
the reader — BOTH give the SAME structured error (max_abs 98304@M256 / 32768@M2048, i.e. a
few K-blocks' worth), independent of store method => the mismatch is in the LDS<->reader
PERMUTATION, not the store call.
Why: gemm2's DMA path (dma_x_tile_to_lds) reads HBM at a SWIZZLED byte-column
(global_byte_idx = row_k_dw*4 + swizzle_xor16(row, col*4)) assuming the PADDED 32B/block HBM
layout, and writes LDS lane-SEQUENTIALLY; the reader lds_load_packs_k64 then reads LDS
SWIZZLED. This HBM-swizzle + sequential-LDS + swizzled-read chain is a co-design that bakes a
permutation the straightforward explicit store does NOT reproduce (and the HBM-swizzle assumes
32B/block, which dense 24B/block breaks). So writing logical (row, blk_k*32) swizzled in LDS
does not land where the reader expects for this DMA-co-designed config.
=> Correct dense needs to reproduce the DMA path's exact permutation for 24B blocks, OR change
the LDS to hold dense 24B/block + a matching new reader/MFMA feed (bigger). This is the deep
co-design work; needs more cycles with an LDS-dump/per-element trace (bit-exact-only feedback
is too coarse). Scaffolding remains OFF by default; padded serving verified intact (288us@256).

## BREAKTHROUGH (cycle ~9): self-consistent linear store+reader = 88% bit-exact
Approach that works: for dense, override BOTH store_x_tile_to_lds AND lds_load_packs_k64
with a NON-swizzled LINEAR LDS layout (lds[row*_eff_lds_stride + col_bytes]) — bypasses
gemm2's DMA/swizzle co-design; compute_tile picks up the overridden reader via late binding.
Result: 88% of outputs bit-exact (was 100% wrong); ~12% differ, ROW-DEPENDENT (some rows
100% correct), small diffs (e.g. row0: only 1 of 6 elems off). => approach is correct;
narrow residual in a subset (~1/8, one K-tile or padding/sort-row handling).
Next probes: (a) K-tail tile (main loop is 2-tile ping-pong + tail; inter_dim=2048/tile_k=256
= 8 tiles) — check the tail prefetch/compute; (b) padding rows (t>=tokens clamp to row_ts=0)
vs DMA's handling; (c) whether _eff_lds_stride vs tile_k mismatch leaves a col range.

## Error is PADDING/small-M edge (frac wrong: 11.9%@256, 4.4%@512, 0.00%@2048)
Decreasing with M => not a K-tile bug; it's the sorted-padding-row handling at small M
(fewer valid tokens -> more padding blocks). At M=2048 (prefill, where the mxfp4 gap is)
dense is essentially bit-exact (1 negligible element). Residual fix = match DMA's padding-row
behavior for t>=tokens (both clamp row_ts->0, but a subtle diff remains at partial blocks).
## PERF reality: linear (non-swizzled, explicit) dense is SLOWER than padded DMA
dense_us ~1267 vs padded ~1148 @M2048 (0.91x). The correctness-first linear store+reader
loses the DMA's efficiency + adds LDS bank conflicts, outweighing the 24B-vs-32B HBM saving.
To get the WIN, the dense store needs the swizzle (bank-conflict-free) AND ideally the DMA
path (buffer_load_lds) adapted to 24B blocks. So: approach VALIDATED (near-bit-exact), but
delivering speedup needs (1) padding-residual fix + (2) swizzled/DMA dense store.
