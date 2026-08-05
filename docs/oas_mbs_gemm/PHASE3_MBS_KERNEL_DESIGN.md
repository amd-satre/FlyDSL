# MBS kernel design for `kernels/mxfp4_preshuffle.py`

Status: algorithm validated (host-side quant + full GEMM simulation, both
GPU-verified against BF16 truth); kernel-loop implementation not yet written.
This doc is the precise, actionable spec for that implementation step.

## Why this is a natural fit for this kernel specifically

`kernels/mxfp4_preshuffle.py` uses AMD gfx950's *native* hardware scaled-MFMA
(`MFMA_Scale(16, 16, 128, Float4E2M1FN, ...)`): one `fx.gemm(...)` call already
consumes exactly **K=128** elements per (mi, ni) output tile, with a per-32
E8M0 scale operand. MBS's macro-block size (128, per the paper's Sec 4.3/
Appendix A) is **exactly** this kernel's native per-instruction K-granularity.
This is a lucky structural alignment: on NVIDIA/CUTLASS the paper needed
warp-specialized TMEM double-buffering (Sec 4.3.2/Appendix E) purely to
intercept at 128-K boundaries; here, every single `fx.gemm(...)` call already
*is* one 128-K macro block's contribution to one 16x16 output tile. No new
loop nesting is needed — only what happens to that call's result.

## What must change

### 1. Host-side: new scale tensors (quantization time, no kernel yet)

Already implemented and GPU-verified in
`tests/kernels/utils/oas_mbs_quant.py`:
- `per_1x32_f4_quant_oas(x)` — drop-in OAS-enhanced quantizer (block=32,
  3-candidate E8M0 exponent search). **Zero kernel changes required**; already
  proven end-to-end on the unmodified kernel (`verify_oas_gemm_gpu.py`:
  +0.30/+0.32 dB QSNR vs BF16 truth on Gaussian/outlier-heavy A, gfx950).
- `mbs_factor_static_e0m8(macro_absmax)` — Eq. 3 mantissa extraction.
- `per_1x32_f4_quant_oas_mbs(x, macro_block=128)` — pre-scales each 128-wide
  macro block by its factor before OAS quantization; returns the mantissa
  byte tensor (shape `[rows, K/128]`) the kernel epilogue must consume.

**New for the kernel**: the mantissa tensor must be laid out
**`[K/128, M_padded]`** (transposed from the natural `[M, K/128]`), so that for
a fixed macro-block index, the M consecutive rows a lane needs (`row_m,
row_m+1, row_m+2, row_m+3` — see below) are contiguous in memory and loadable
with one 4-byte (`i32`) buffer load, mirroring how `shuffle_scale_w4` already
reorganizes the E8M0 scale for lane-local access. Same transpose for B's
mantissa tensor: `[K/128, N_padded]`. This is a new `shuffle_mbs_scale_w4`-style
host function, not yet written.

### 2. Kernel: per-call local accumulate + row/col Hadamard correction

Current `compute()` (kernels/mxfp4_preshuffle.py:454-485): for each
`(kh, ni, mi)`, `fx.gemm(atom, cf, av[...], bv[...], cf, scale_a=, scale_b=)`
accumulates **directly and irreversibly** into the persistent `c_frags[idx]`
(`cf`), which already holds the running sum from all prior K-tiles. MBS needs
this call's *isolated* contribution before it's merged, so it can be scaled by
this specific macro block's `(1/factor_a) * (1/factor_b)` first:

```python
# Per (kh, ni, mi), replace the direct in-place accumulate with:
tmp = fx.make_rmem_tensor(4, Float32)
tmp.store(Vec.filled(4, 0.0, Float32))
fx.gemm(scale_atoms[(kh * 2 + im, kh * 2 + in_b)], tmp,
        av[mi * k_halves + kh], bv[ni * k_halves + kh], tmp,
        scale_a=sa_v[mp_i], scale_b=sb_v[np_i])
# tmp[0..3] = this macro block's isolated contribution for M-rows
#   row_m+0 .. row_m+3 (see epilogue's row_m formula), N-col = col (fixed per lane)
sigma = Vec.from_elements([
    _raw(mbs_a_recip[mi][0] * mbs_b_recip[ni]),   # ii=0: row_m+0
    _raw(mbs_a_recip[mi][1] * mbs_b_recip[ni]),   # ii=1: row_m+1
    _raw(mbs_a_recip[mi][2] * mbs_b_recip[ni]),   # ii=2: row_m+2
    _raw(mbs_a_recip[mi][3] * mbs_b_recip[ni]),   # ii=3: row_m+3
], Float32)
cf_new = Vec(cf.load()) + Vec(tmp.load()) * sigma
cf.store(cf_new)
```

`mbs_a_recip[mi]` = 4 per-row reciprocal MBS factors for the current macro
block (`kt` if `BK==128`, `kt*2+kh` if `BK==256` — same `chunk_kt`-style index
already used for the existing e8m0 scale load, just at 128 granularity instead
of 256). `mbs_b_recip[ni]` = 1 per-column reciprocal factor (columns are fixed
per lane; only rows vary with `ii`).

**This must be true float reciprocal division** (`1.0 / (1 + m8/256)`), not a
power-of-two shift like the existing E8M0 scale — the existing `scale_a=`/
`scale_b=` MMA operands only accept E8M0 (power-of-two) scales, which is
exactly why MBS's linear-mantissa correction can't ride that path and must be
a separate epilogue-style FMA per call, matching the paper's own Sec 4.3.2
description of doing this correction on "Vector Cores" concurrently with the
Tensor Core MMA stream, not inside the MMA operand itself.

### 3. New per-macro-block scale load routine

Needs a new function alongside `load_sc()` (kernels/mxfp4_preshuffle.py:428),
call it `load_mbs(macro_kt)`:
- For A: one `buffer_load` of 4 contiguous bytes (i32) per `mi` per macro
  block, from the `[K/128, M_padded]`-transposed mantissa buffer at
  `(macro_kt, bx_m + mi*16 + lane_div_16*4)`; convert each byte to
  `1.0 / (1.0 + byte/256.0)` (4 scalar FP32 reciprocals per `mi`).
- For B: one scalar byte load per `ni` per macro block (column is fixed per
  lane, so 1 value, not 4), same reciprocal conversion.
- Reuse the existing `m_pairs`/`n_pairs`-style pairing if profitable, but
  correctness first: start with one load per `mi`/`ni` per macro block, tune
  later (Phase 4).

### 4. Threading through `compile_mxfp4_gemm` / `launch_gemm`

Two new `fx.Tensor` args end-to-end: `arg_mbs_a`, `arg_mbs_b` (uint8), passed
like `arg_scale_a`/`arg_scale_b` today. Add a `use_mbs: bool` flag to
`_compile_mxfp_blockscale_gemm` / `compile_mxfp4_gemm` so the existing
(already-tuned) non-MBS path is untouched when `use_mbs=False` — this new
logic must be **strictly additive**, gated behind the flag, so
`test_preshuffle_gemm.py`'s existing 102-pass baseline keeps passing
unmodified.

## Verification plan for the implementation step

1. Smallest possible shape first (e.g. `M=64,N=128,K=256`, i.e. `K_TILES=1-2`)
   with `use_mbs=True`, comparing kernel output to
   `verify_mbs_gemm_math.py`'s pure-PyTorch simulation (already validated
   against BF16 truth) — not against BF16 directly, to isolate kernel-authoring
   bugs from quantization-approximation error.
2. Only once that matches within FP32 rounding tolerance, sweep to the full
   shape suite in `test_mfma_w4_flyc_preshuffle` and re-run the outlier-heavy
   comparison from `verify_oas_gemm_gpu.py` with MBS enabled.
3. `debug-flydsl-kernel` skill's playbook (clear `~/.flydsl/cache`, all-1s
   test, single-partition test) if results are wrong rather than crashing.

## Not yet done (this is the actual remaining Phase 3 work)

- `shuffle_mbs_scale_w4`-equivalent host transpose function.
- `load_mbs()` kernel routine.
- The `tmp`/Hadamard-correction rewrite of `compute()`, gated by `use_mbs`.
- Wiring `arg_mbs_a`/`arg_mbs_b` through `compile_mxfp4_gemm`/`launch_gemm`.
- New correctness tests mirroring `test_mfma_w4_flyc_preshuffle`.
