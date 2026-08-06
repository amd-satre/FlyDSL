# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MXFP4 (E2M1) preshuffle GEMM with Overflow-Aware Scaling (OAS) + Macro
Block Scaling (MBS), per arXiv:2603.08713 Sec 4.2/4.3.

This ports the production ``kernels/mxfp4_preshuffle.py``'s async-copy DMA and
MFMA instruction scheduler (Phase 4 tuning, see
``docs/oas_mbs_gemm/PHASE4_TUNING_NOTES.md``) onto the MBS-augmented main loop
from Phase 3 (``docs/oas_mbs_gemm/PHASE3_MBS_KERNEL_DESIGN.md``). The MBS
scale loads (``load_mbs``) are deliberately NOT counted in the scheduler's
vmem/ds interleave counts -- they're a small, fixed 1-dword-per-mi +
1-byte-per-ni load per K-tile, negligible next to the B-tile stream -- so the
scheduler hints are copied unmodified from the production kernel.

MBS: each 16x16x128 scaled-MFMA call already covers exactly one 128-K macro
block (this hardware's native scaled-MFMA granularity happens to match MBS's
macro-block size exactly). For each such call we accumulate into a fresh
zero-initialized local fragment (instead of the persistent running
accumulator), then apply the Hadamard correction
``local *= (1/mbs_factor_a[row]) * (1/mbs_factor_b[col])`` before adding it
into the running accumulator -- this is the per-macro-block "epilogue
interception" the paper describes for CUTLASS/TMEM (Sec 4.3.2), just simpler
here because the call boundary already IS the macro-block boundary.

MBS scale layout (host side, see ``tests/kernels/utils/oas_mbs_quant.py`` /
``shuffle_mbs_scale_w4``): stored TRANSPOSED as ``[K/128, M_padded]`` (A) /
``[K/128, N_padded]`` (B) uint8 mantissa bytes, so that for one macro block,
the 4 consecutive M-rows (or N-cols) a lane needs are contiguous in memory.
"""

from typing import Optional

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import BFloat16, Float4E2M1FN, Float16, Float32, Int8, Int32, T
from flydsl.expr.typing import Vector as Vec

from kernels.mxfp4_preshuffle import _bq_view, _scale_mma_atoms


def _raw(v):
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def compile_mxfp4_gemm_mbs(
    *,
    N: int,
    K: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    out_dtype: str = "bf16",
):
    """MXFP4 preshuffle GEMM with OAS+MBS -- correctness-first, unscheduled.

    Returns fn(C, A, B, scale_a, scale_b, mbs_a, mbs_b, bias, M, N, stream).
    A: MXFP4 (E2M1), 2 codes/byte. B: CK-preshuffled MXFP4. scale_a/scale_b:
    e8m0 (per-32). mbs_a/mbs_b: uint8 mantissa, transposed [K/128, M or N].
    """
    BM, BN, BK = tile_m, tile_n, tile_k
    if BK not in (128, 256) or K % BK != 0:
        raise ValueError(f"tile_k must be 128 or 256 dividing K; got tile_k={BK}, K={K}")
    if K % 256 != 0:
        raise ValueError(f"K must be a multiple of 256 (e8m0 scale chunk); got K={K}")
    out_elem = BFloat16 if out_dtype == "bf16" else Float16

    a_row_bytes = K // 2
    A_ROW_B = BK // 2
    A_LDS_B = BM * A_ROW_B
    A_ROW_I32 = A_ROW_B // 4

    K_HALF = K // 2
    KH4 = K_HALF // 4
    K_TILES = K // BK
    k_halves = BK // 128
    tiles_per_chunk = 256 // BK
    m_chunks = BM // 16
    num_acc_n = (BN // 4) // 16
    _scale_chunk_dw = (K // 32 // 4 // 2) * 64
    _scale_k0_dw = 64

    n_coop = A_LDS_B // 256 // 16

    n_pairs = max(1, num_acc_n // 2)
    m_pairs = max(1, m_chunks // 2)

    # Occupancy hint (rocdl.waves_per_eu), mirroring kernels/mxfp4_preshuffle.py's
    # value_attrs mechanism. compile_mxfp4_gemm_mbs has no waves_per_eu kwarg (the
    # task_runner call site is frozen), so the value is derived internally purely
    # from tile_m/tile_n/tile_k -- only the 3 (tile_m,tile_n,tile_k) configs that
    # appear across TEST_SHAPES are keyed; anything else falls back to None (no
    # hint, i.e. compiler auto-picked occupancy, identical to prior behavior).
    _MBS_WAVES_PER_EU = {
        (32, 128, 256): None,
        (64, 128, 256): None,
        (128, 256, 128): 1,
    }
    waves_per_eu = _MBS_WAVES_PER_EU.get((BM, BN, BK), None)

    # MBS: macro-block index granularity is 128 (matches one 16x16x128 MFMA
    # call). M/N padded to 32 like the existing e8m0 scale bound.
    K_MACRO = K // 128

    # Scheduler counts (sched_group_barrier interleave), per loop iter --
    # base counts copied unmodified from kernels/mxfp4_preshuffle.py's fp4
    # (non-fp6) path (A coop + B + scales). Extended below to also cover
    # load_mbs's own gmem loads (A-side dword loads: one per m_chunks, B-side
    # byte loads: one per num_acc_n, each issued once per k_halves macro
    # block per K-tile iteration) -- these were previously NOT counted here,
    # so hot_loop_scheduler's vmem interleave budget didn't know about them
    # and they weren't spread across the MFMA issue stream by build_scheduler/
    # vmem_schedule (see module docstring + DIRECTION r1_d1). This is a pure
    # bookkeeping/count fix; load_mbs's call site and internals are untouched.
    sched_mfma_total = k_halves * m_chunks * num_acc_n
    sched_num_ds_load = m_chunks * k_halves  # A LDS reads/thread (read_a)
    sched_num_gmem_base = n_coop + num_acc_n * k_halves + m_pairs + n_pairs  # A coop + B + scales (orig)
    sched_num_gmem_mbs = (m_chunks + num_acc_n) * k_halves  # load_mbs: A dword + B byte loads, per macro block
    sched_num_gmem = sched_num_gmem_base + sched_num_gmem_mbs  # A coop + B + scales + MBS
    sched_num_a_dswr = 0  # async copy -> no explicit A LDS writes to schedule
    enable_scheduler = num_acc_n <= 2 or True  # always async-copy here, like use_async_copy=True upstream
    dsrd_preload = sched_num_ds_load
    # Preload ALL vmem loads (A coop + B + scales + MBS) up front, same
    # preload-everything relationship as the original code (dvmem_preload ==
    # sched_num_gmem) -- measured to beat splitting MBS loads into the
    # per-mfma interleave budget (that variant regressed the small-M cases).
    dvmem_preload = sched_num_gmem

    @fx.struct
    class SharedA:
        a0: fx.Array[Int8, A_LDS_B, 16]
        a1: fx.Array[Int8, A_LDS_B, 16]

    @flyc.kernel
    def kernel_gemm(
        arg_c: fx.Tensor,
        arg_a: fx.Int64,
        arg_b: fx.Int64,
        arg_scale_a: fx.Int64,
        arg_scale_b: fx.Int64,
        arg_mbs_a: fx.Int64,
        arg_mbs_b: fx.Int64,
        arg_bias: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
    ):
        scale_atoms = _scale_mma_atoms()

        tid = fx.thread_idx.x
        bid_x, bid_y, _ = fx.block_idx
        wave = rocdl.readfirstlane(T.i32, fx.Int32(tid) // fx.Int32(64))
        lane = fx.Int32(tid) % fx.Int32(64)
        lane_div_16 = lane // fx.Int32(16)
        lane_mod_16 = lane % fx.Int32(16)
        bx_m = bid_x * fx.Int32(BM)
        by_n = bid_y * fx.Int32(BN)

        _i8g = fx.PointerType.get(T.i8, address_space=fx.AddressSpace.Global, alignment=16)
        a_nrec = fx.Int64(i32_m) * fx.Int64(a_row_bytes)
        a_flat = fx.rocdl.make_buffer_tensor(
            fx.Tensor(fx.make_view(fx.inttoptr(_i8g, fx.Int64(arg_a)), fx.make_layout(65536 * a_row_bytes, 1))),
            max_size=False,
            num_records_bytes=a_nrec,
        )
        a_flat_div = fx.logical_divide(a_flat, fx.make_layout(1, 1))
        lds = fx.SharedAllocator().allocate(SharedA).peek()
        sA0_i32 = fx.recast_iter(Int32, lds.a0.ptr)
        lds_db = fx.Int32(fx.ptrtoint(lds.a1.ptr)) - fx.Int32(fx.ptrtoint(lds.a0.ptr))
        lds_db_i32 = lds_db // fx.Int32(4)
        lds_copy = fx.make_copy_atom(fx.UniversalCopy128b(), Int32)
        dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        _i8s = fx.PointerType.get(Int8.ir_type, fx.AddressSpace.Shared, 512)
        sA0_i8 = fx.recast_iter(_i8s, lds.a0.ptr)

        def _iter_of(parity):
            return fx.add_offset(sA0_i32, parity * lds_db_i32)

        def _lds_view(base_iter, off_i32):
            return fx.make_view(fx.add_offset(base_iter, off_i32), fx.make_layout(4, 1))

        def dma_a_to_lds(kt, parity):
            # Direct gmem->LDS DMA (buffer_load_lds), same row-major LDS layout
            # as coop_load_a. Issued after the B/scale loads so it overlaps
            # the MFMAs (copied unmodified from kernels/mxfp4_preshuffle.py).
            base_off = rocdl.readfirstlane(T.i32, parity * lds_db + wave * fx.Int32(64 * 16))
            lds_ptr = fx.add_offset(sA0_i8, base_off)
            base_k_byte = kt * fx.Int32(A_ROW_B)
            for i in range_constexpr(n_coop):
                if const_expr(i > 0):
                    lds_ptr = fx.add_offset(lds_ptr, fx.Int32(256 * 16))
                lin = (fx.Int32(i * 256) + fx.Int32(tid)) * fx.Int32(16)
                row = lin // fx.Int32(A_ROW_B)
                col = lin % fx.Int32(A_ROW_B)
                gmem_byte = (bx_m + row) * fx.Int32(a_row_bytes) + base_k_byte + col
                dst = fx.make_view(lds_ptr, fx.make_layout(1, 1))
                src = fx.slice(a_flat_div, (None, gmem_byte))
                fx.copy(dma_atom, src, dst)

        def _read16(base_iter, off_i32):
            t = fx.make_rmem_tensor(4, Int32)
            fx.copy(lds_copy, _lds_view(base_iter, off_i32), t)
            return t

        def read_a(parity):
            base_iter = _iter_of(parity)
            av = []
            for mi in range_constexpr(m_chunks):
                for kh in range_constexpr(k_halves):
                    off = (
                        (fx.Int32(mi * 16) + lane_mod_16) * fx.Int32(A_ROW_I32)
                        + fx.Int32(kh * 16)
                        + lane_div_16 * fx.Int32(4)
                    )
                    av.append(_read16(base_iter, off))
            return av

        n_col_base = by_n + wave * fx.Int32(BN // 4)
        bq_views = [
            _bq_view(arg_b, n_col_base + fx.Int32(ni * 16), KH4, K_TILES, k_halves) for ni in range_constexpr(num_acc_n)
        ]
        b_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), 32)
        bs_copy = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)
        mbs_copy = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)
        mbs_b_copy = fx.make_copy_atom(fx.rocdl.BufferCopy8b(), 32)

        _i32g = fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4)
        _i8g_s = fx.PointerType.get(T.i8, address_space=fx.AddressSpace.Global, alignment=4)
        _sc_layout = fx.make_layout(1 << 28, 1)
        a_sc_nrec = fx.Int64((i32_m + fx.Int32(31)) // fx.Int32(32)) * fx.Int64(_scale_chunk_dw) * fx.Int64(4)
        b_sc_nrec = fx.Int64((N // 32) * _scale_chunk_dw * 4)
        sa_flat = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(
                fx.Tensor(fx.make_view(fx.inttoptr(_i32g, fx.Int64(arg_scale_a)), _sc_layout)),
                max_size=False,
                num_records_bytes=a_sc_nrec,
            ),
            fx.make_layout(1, 1),
        )
        sb_flat = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(
                fx.Tensor(fx.make_view(fx.inttoptr(_i32g, fx.Int64(arg_scale_b)), _sc_layout)),
                max_size=False,
                num_records_bytes=b_sc_nrec,
            ),
            fx.make_layout(1, 1),
        )

        # MBS mantissa buffers: transposed [K/128, M_padded] / [K/128, N_padded]
        # uint8, flat byte-addressed. Bound to the real (macro-block-aligned)
        # extents so OOB reads return 0 -> mbs factor 1.0 (no-op) past M/N.
        _mbs_layout = fx.make_layout(1 << 30, 1)
        m_pad32 = (i32_m + fx.Int32(31)) // fx.Int32(32) * fx.Int32(32)
        mbs_a_nrec = fx.Int64(K_MACRO) * fx.Int64(m_pad32)
        mbs_b_nrec = fx.Int64(K_MACRO) * fx.Int64((N + 31) // 32 * 32)
        mbs_a_flat = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(
                fx.Tensor(fx.make_view(fx.inttoptr(_i8g_s, fx.Int64(arg_mbs_a)), _mbs_layout)),
                max_size=False,
                num_records_bytes=mbs_a_nrec,
            ),
            fx.make_layout(1, 1),
        )
        mbs_b_flat = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(
                fx.Tensor(fx.make_view(fx.inttoptr(_i8g_s, fx.Int64(arg_mbs_b)), _mbs_layout)),
                max_size=False,
                num_records_bytes=mbs_b_nrec,
            ),
            fx.make_layout(1, 1),
        )

        a_sc_base = [
            (bx_m // fx.Int32(32) + fx.Int32(mp)) * fx.Int32(_scale_chunk_dw) for mp in range_constexpr(m_pairs)
        ]
        nsb = by_n // fx.Int32(32) + wave * fx.Int32(BN // 128)
        b_sc_base = [(nsb + fx.Int32(np)) * fx.Int32(_scale_chunk_dw) for np in range_constexpr(n_pairs)]
        sc_lane = lane_div_16 * fx.Int32(16) + lane_mod_16

        n_acc = m_chunks * num_acc_n

        def load_b(kt):
            ops = []
            for ni in range_constexpr(num_acc_n):
                for kh in range_constexpr(k_halves):
                    bf = fx.make_rmem_tensor(4, Int32)
                    fx.copy_atom_call(b_copy, bq_views[ni][lane_div_16, lane_mod_16, kt, kh, None], bf)
                    ops.append(bf)
            return ops

        def load_sc(chunk_kt):
            koff = chunk_kt * fx.Int32(_scale_k0_dw)
            sa = [
                Vec(
                    fly.copy_atom_call_ssa(
                        [T.vec(1, T.i32)],
                        bs_copy,
                        sa_flat[None, rocdl.readfirstlane(T.i32, a_sc_base[mp] + koff) + sc_lane],
                    )
                )[0]
                for mp in range_constexpr(m_pairs)
            ]
            sb = [
                Vec(
                    fly.copy_atom_call_ssa(
                        [T.vec(1, T.i32)],
                        bs_copy,
                        sb_flat[None, rocdl.readfirstlane(T.i32, b_sc_base[np] + koff) + sc_lane],
                    )
                )[0]
                for np in range_constexpr(n_pairs)
            ]
            return sa, sb

        def _byte_to_recip(byte_i32_raw):
            # (1 + m8/256)^-1, computed in fp32. byte_i32_raw: raw i32 ir.Value,
            # 0..255 (sign doesn't matter -- always non-negative in that range).
            #
            # `factor` is always in [1, 2) (well-conditioned), so a single
            # hardware v_rcp_f32 (rocdl.rcp) is enough -- no Newton-Raphson
            # refinement needed. Measured: arith.divf-based division here was
            # the dominant cost of the entire MBS kernel (~130us -> ~53us,
            # i.e. bringing this kernel to within a few % of baseline, when
            # ablated away in isolation) -- see docs/oas_mbs_gemm/PHASE4_TUNING_NOTES.md.
            m8_f = arith.uitofp(T.f32, byte_i32_raw)
            one = arith.constant(1.0, type=T.f32)
            inv256 = arith.constant(1.0 / 256.0, type=T.f32)
            factor = arith.addf(one, arith.mulf(m8_f, inv256))
            return fx.Float32(rocdl.rcp(T.f32, factor))

        def load_mbs(macro_kt):
            # A: one dword per mi = 4 contiguous row-bytes at
            # (macro_kt * m_pad32 + bx_m + mi*16 + lane_div_16*4).
            #
            # `macro_kt * m_pad32 + bx_m` is uniform across the ENTIRE
            # workgroup (doesn't depend on mi, a compile-time constant, or any
            # per-lane value) -- hoist the readfirstlane out of the mi loop
            # entirely instead of redundantly re-broadcasting it m_chunks
            # times. `mi*16` is added afterward as a plain scalar op (still
            # compile-time constant, no extra lane divergence).
            a_uniform_base = rocdl.readfirstlane(T.i32, macro_kt * m_pad32 + bx_m)
            a_recip = []
            for mi in range_constexpr(m_chunks):
                byte_off = a_uniform_base + fx.Int32(mi * 16) + lane_div_16 * fx.Int32(4)
                word = Vec(
                    fly.copy_atom_call_ssa([T.vec(1, T.i32)], mbs_copy, mbs_a_flat[None, byte_off])
                )[0]
                w = _raw(word)
                recips = []
                for ii in range_constexpr(4):
                    shifted = arith.shrui(w, arith.constant(ii * 8, type=T.i32))
                    byte_ii = arith.andi(shifted, arith.constant(0xFF, type=T.i32))
                    recips.append(_byte_to_recip(byte_ii))
                a_recip.append(recips)
            # B: one byte per ni (column fixed per lane). Same hoisting: the
            # workgroup-uniform part goes through readfirstlane once, outside
            # the ni loop; lane_mod_16 (per-lane) is added after.
            n_pad32 = (N + 31) // 32 * 32
            b_uniform_base = rocdl.readfirstlane(T.i32, macro_kt * fx.Int32(n_pad32) + by_n + wave * fx.Int32(BN // 4))
            b_recip = []
            for ni in range_constexpr(num_acc_n):
                byte_off = b_uniform_base + fx.Int32(ni * 16) + lane_mod_16
                byte = Vec(fly.copy_atom_call_ssa([T.vec(1, T.i8)], mbs_b_copy, mbs_b_flat[None, byte_off]))[0]
                byte_i32 = arith.extui(T.i32, _raw(byte))
                b_recip.append(_byte_to_recip(byte_i32))
            return a_recip, b_recip

        def compute_with_macro(accs, av, bv, sa_v, sb_v, macro_recips, scale_shift):
            if const_expr(scale_shift is not None):
                sh = _raw(scale_shift)
                sa_v = [arith.shrui(_raw(v), sh) for v in sa_v]
                sb_v = [arith.shrui(_raw(v), sh) for v in sb_v]
            c_frags = [fx.make_rmem_tensor(4, Float32) for _ in range_constexpr(n_acc)]
            for idx in range_constexpr(n_acc):
                c_frags[idx].store(Vec(accs[idx]))
            for kh in range_constexpr(k_halves):
                a_recip, b_recip = macro_recips[kh]
                for ni in range_constexpr(num_acc_n):
                    np_i, in_b = ni // 2, ni % 2
                    for mi in range_constexpr(m_chunks):
                        mp_i, im = mi // 2, mi % 2
                        cf = c_frags[mi * num_acc_n + ni]
                        tmp = fx.make_rmem_tensor(4, Float32)
                        tmp.store(Vec.filled(4, 0.0, Float32))
                        fx.gemm(
                            scale_atoms[(kh * 2 + im, kh * 2 + in_b)],
                            tmp,
                            av[mi * k_halves + kh],
                            bv[ni * k_halves + kh],
                            tmp,
                            scale_a=sa_v[mp_i],
                            scale_b=sb_v[np_i],
                        )
                        sigma = Vec.from_elements(
                            [_raw(a_recip[mi][ii] * b_recip[ni]) for ii in range_constexpr(4)], Float32
                        )
                        cf_new = Vec(cf.load()) + Vec(tmp.load()) * sigma
                        cf.store(cf_new)
            for idx in range_constexpr(n_acc):
                accs[idx] = c_frags[idx].load().ir_value()
            return accs

        # Scheduler hints: interleave the MFMAs with the vmem loads + A LDS
        # read/writes -- copied unmodified from kernels/mxfp4_preshuffle.py.
        def build_scheduler(numer, denom):
            if const_expr(denom <= 0):
                return []
            if const_expr(numer <= 0):
                return [0] * denom
            out = []
            prev = 0
            for i in range_constexpr(denom):
                cur = ((i + 1) * numer + (denom - 1)) // denom
                out.append(cur - prev)
                prev = cur
            return out

        def hot_loop_scheduler():
            mfma_total = sched_mfma_total
            dswr_tail = min(sched_num_a_dswr, mfma_total)
            dsrd_preload_eff = min(int(dsrd_preload), sched_num_ds_load)
            dvmem_preload_eff = min(int(dvmem_preload), sched_num_gmem)
            vmem_remaining = sched_num_gmem - dvmem_preload_eff
            dsrd_remaining = sched_num_ds_load - dsrd_preload_eff
            if const_expr(0 < vmem_remaining < mfma_total):
                vmem_schedule = build_scheduler(vmem_remaining, vmem_remaining) + [0] * (mfma_total - vmem_remaining)
            else:
                vmem_schedule = build_scheduler(vmem_remaining, mfma_total)
            dsrd_schedule = build_scheduler(dsrd_remaining, mfma_total)
            dswr_start = max(mfma_total - dswr_tail - 2, 0)
            last_dsrd_mfma_idx = -1
            for sched_idx in range_constexpr(mfma_total):
                if const_expr(dsrd_schedule[sched_idx]):
                    last_dsrd_mfma_idx = sched_idx
            dswr_start = max(dswr_start, last_dsrd_mfma_idx + 1)
            idx_ds_read = dsrd_preload_eff
            idx_gmem_load = dvmem_preload_eff
            idx_ds_write = 0
            if const_expr(dvmem_preload_eff):
                rocdl.sched_vmem(dvmem_preload_eff)
            if const_expr(dsrd_preload_eff):
                rocdl.sched_dsrd(dsrd_preload_eff)
            for mfma_idx in range_constexpr(mfma_total):
                rocdl.sched_mfma(1)
                n_dsrd = dsrd_schedule[mfma_idx]
                if const_expr(n_dsrd and (idx_ds_read < sched_num_ds_load)):
                    if const_expr(idx_ds_read + n_dsrd > sched_num_ds_load):
                        n_dsrd = sched_num_ds_load - idx_ds_read
                    if const_expr(n_dsrd):
                        rocdl.sched_dsrd(n_dsrd)
                        idx_ds_read += n_dsrd
                n_vmem = vmem_schedule[mfma_idx]
                if const_expr(n_vmem and (idx_gmem_load < sched_num_gmem)):
                    if const_expr(idx_gmem_load + n_vmem > sched_num_gmem):
                        n_vmem = sched_num_gmem - idx_gmem_load
                    if const_expr(n_vmem):
                        rocdl.sched_vmem(n_vmem)
                        idx_gmem_load += n_vmem
                if const_expr((idx_ds_write < dswr_tail) and (mfma_idx >= dswr_start)):
                    rocdl.sched_dswr(1)
                    idx_ds_write += 1
            if const_expr(idx_ds_write < sched_num_a_dswr):
                rocdl.sched_dswr(sched_num_a_dswr - idx_ds_write)
            rocdl.sched_barrier(0)

        accs_init = [Vec.filled(4, 0.0, Float32).ir_value() for _ in range_constexpr(n_acc)]

        dma_a_to_lds(fx.Int32(0), fx.Int32(0))
        rocdl.s_waitcnt(0)
        gpu.barrier()
        for iv, state in range(fx.Index(0), fx.Index(K_TILES), fx.Index(1), init=accs_init):
            accs = list(state)
            kt = fx.Int32(iv)
            cur = kt % fx.Int32(2)
            nxt = (kt + fx.Int32(1)) % fx.Int32(2)
            nkt = kt + fx.Int32(1)
            pf_kt = nkt - nkt // fx.Int32(K_TILES)
            chunk_kt = kt if tiles_per_chunk == 1 else kt // fx.Int32(tiles_per_chunk)
            scale_shift = None if tiles_per_chunk == 1 else (kt % fx.Int32(tiles_per_chunk)) * fx.Int32(16)
            av = read_a(cur)
            bv = load_b(kt)
            sa_v, sb_v = load_sc(chunk_kt)
            # This K-tile covers k_halves consecutive 128-K macro blocks,
            # starting at kt * k_halves.
            macro_recips = [load_mbs(kt * fx.Int32(k_halves) + fx.Int32(kh)) for kh in range_constexpr(k_halves)]
            dma_a_to_lds(pf_kt, nxt)  # A DMA AFTER B/scale/mbs loads -> overlaps the MFMAs
            accs = compute_with_macro(accs, av, bv, sa_v, sb_v, macro_recips, scale_shift)
            if const_expr(enable_scheduler):
                hot_loop_scheduler()
            rocdl.s_waitcnt(0)  # drain the A DMA before the barrier
            gpu.barrier()
            results = yield accs
        accs = results

        c_nrec = fx.Int64(i32_m) * fx.Int64(N) * fx.Int64(2)
        c_rsrc = buffer_ops.create_buffer_resource(arg_c, max_size=False, num_records_bytes=c_nrec)
        col_w = by_n + wave * fx.Int32(BN // 4) + lane_mod_16
        for mi in range_constexpr(m_chunks):
            row_m = bx_m + fx.Int32(mi * 16) + lane_div_16 * fx.Int32(4)
            for ni in range_constexpr(num_acc_n):
                col = col_w + fx.Int32(ni * 16)
                acc = Vec(accs[mi * num_acc_n + ni])
                for ii in range_constexpr(4):
                    val = acc[ii].to(out_elem)
                    off = (row_m + fx.Int32(ii)) * fx.Int32(N) + col
                    buffer_ops.buffer_store(val.ir_value(), c_rsrc, off)

    @flyc.jit
    def launch_gemm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        arg_mbs_a: fx.Tensor,
        arg_mbs_b: fx.Tensor,
        arg_bias: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        stream: fx.Stream,
    ):
        from flydsl.compiler.kernel_function import CompilationContext

        CompilationContext.get_current()
        a_addr = fx.Int64(fx.ptrtoint(fx.get_iter(arg_a)))
        b_addr = fx.Int64(fx.ptrtoint(fx.get_iter(arg_b)))
        sa_addr = fx.Int64(fx.ptrtoint(fx.get_iter(arg_scale_a)))
        sb_addr = fx.Int64(fx.ptrtoint(fx.get_iter(arg_scale_b)))
        mbsa_addr = fx.Int64(fx.ptrtoint(fx.get_iter(arg_mbs_a)))
        mbsb_addr = fx.Int64(fx.ptrtoint(fx.get_iter(arg_mbs_b)))
        M_max = 65536
        arg_c_2d = fx.Tensor(fx.make_view(fx.get_iter(arg_c), fx.make_layout((M_max, N), (N, 1))))
        gx = (i32_m + (BM - 1)) // BM
        gy = i32_n // BN
        kernel_gemm(
            arg_c_2d,
            a_addr,
            b_addr,
            sa_addr,
            sb_addr,
            mbsa_addr,
            mbsb_addr,
            arg_bias,
            i32_m,
            i32_n,
            value_attrs={"rocdl.waves_per_eu": waves_per_eu},
        ).launch(grid=(gx, gy, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm
