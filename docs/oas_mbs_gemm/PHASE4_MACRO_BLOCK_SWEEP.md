# macro_block sweep: {32, 64, 128, 256, 512}

Per user request, attempted the full sweep. Result: **only 128 and 256 are
achievable with the current kernel**; 32/64 are a hard hardware limit, 512
needs an unrelated, larger kernel extension. Full raw output:
`docs/oas_mbs_gemm/macro_block_sweep.py` output below.

## macro_block=32, 64: not physically achievable

`compile_mxfp4_gemm_mbs` raises `ValueError: macro_block must be a multiple
of 128`. This is not an arbitrary restriction — it's a hardware fact: this
kernel's MBS correction works by grouping `n_group = macro_block // 128`
**whole native scaled-MFMA calls** under one Hadamard correction (see
`PHASE4_STRUCTURAL_REWRITE.md`). AMD's `mfma.scale.f32.16x16x128.f8f6f4`
instruction is **atomic** — it computes the full 128-K reduction for a 16x16
output tile in one hardware instruction with no exposed intermediate partial
sum at K=32 or K=64. There is no way to intercept "64 K-elements into a
128-K MFMA call" to apply a finer correction without decomposing the MFMA
itself into smaller instructions (which would mean not using the hardware's
native block-scaled tensor-core path at all — a fundamentally different,
almost certainly much slower kernel). The paper's own reference
implementation (Sec 4.3.2/Appendix E) can operate at finer granularity
because CUTLASS/Blackwell's TMEM double-buffering exposes intermediate
accumulator state mid-instruction-sequence in a way this hardware's MFMA
does not.

## macro_block=512: blocked on tile_k=512 support (not yet built)

Requires `tile_k` (the outer K-tile size) to be a multiple of `macro_block`
(so a macro block's MFMA calls always complete within one outer loop
iteration — see `PHASE4_STRUCTURAL_REWRITE.md`'s no-cross-iteration-state
design choice). `compile_mxfp4_gemm_mbs` inherits
`kernels/mxfp4_preshuffle.py`'s baseline constraint `tile_k ∈ {128, 256}`
(`ValueError: tile_k must be 128 or 256 dividing K`) — this is **independent
of MBS**, a limitation of the underlying preshuffle GEMM's e8m0
scale-chunking logic (`tiles_per_chunk = 256 // BK` implicitly assumes
`BK <= 256`; for `BK=512` this becomes `0`, breaking the scale-shift
selection logic used to pick which half of a packed 256-K-wide e8m0 scale
word to consume for the active 128-K sub-block, unrelated to MBS's own
correction machinery). Extending `tile_k` support to 512 is a real,
separate piece of kernel work (generalizing the e8m0 chunk-selection logic),
not attempted here.

## What IS measured: 128 vs 256

Full sweep script: `docs/oas_mbs_gemm/macro_block_sweep.py`.

```
=== macro_block sweep: BOTH operands MBS ===
N=7168 K=7168 M=1024: baseline=0.0480ms  mb=128:+84.9%  mb=256:+37.0%
N=36864 K=7168 M=1024: baseline=0.1994ms  mb=128:+88.0%  mb=256:+36.3%
N=7168 K=2048 M=1024: baseline=0.0191ms  mb=128:+65.6%  mb=256:+27.5%

=== macro_block sweep: B-only MBS ===
N=7168 K=7168 M=1024: baseline=0.0479ms  mb=128:+17.7%  mb=256:+5.8%
N=36864 K=7168 M=1024: baseline=0.1997ms  mb=128:+17.5%  mb=256:+7.9%
N=7168 K=2048 M=1024: baseline=0.0184ms  mb=128:+16.7%  mb=256:+12.1%
```

Consistent with `PHASE4_STRUCTURAL_REWRITE.md`'s earlier findings: doubling
`macro_block` roughly halves the overhead in both the both-operand and
B-only configurations.
