# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MXFP4 (E2M1) preshuffle GEMM with Overflow-Aware Scaling (OAS) + Macro
Block Scaling (MBS), per arXiv:2603.08713 Sec 4.2/4.3.

Correctness-first variant: this is deliberately a simplified copy of
``kernels/mxfp4_preshuffle.py``'s fp4 path (no async-copy DMA, no MFMA
instruction scheduler hints) with MBS added, per
``docs/oas_mbs_gemm/PHASE3_MBS_KERNEL_DESIGN.md``. Performance tuning
(matching or exceeding the production kernel's scheduler/async-copy tricks) is
Phase 4's job, not this file's.

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

    # MBS: macro-block index granularity is 128 (matches one 16x16x128 MFMA
    # call). M/N padded to 32 like the existing e8m0 scale bound.
    K_MACRO = K // 128

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

        a_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), 32)
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

        def _iter_of(parity):
            return fx.add_offset(sA0_i32, parity * lds_db_i32)

        def _lds_view(base_iter, off_i32):
            return fx.make_view(fx.add_offset(base_iter, off_i32), fx.make_layout(4, 1))

        def coop_load_a(kt, base_iter):
            base_k_byte = kt * fx.Int32(A_ROW_B)
            for i in range_constexpr(n_coop):
                lin = (fx.Int32(i * 256) + fx.Int32(tid)) * fx.Int32(16)
                row = lin // fx.Int32(A_ROW_B)
                col = lin % fx.Int32(A_ROW_B)
                gmem_byte = (bx_m + row) * fx.Int32(a_row_bytes) + base_k_byte + col
                reg = fx.make_rmem_tensor(4, Int32)
                fx.copy_atom_call(a_copy, a_flat_div[None, gmem_byte], reg)
                fx.copy(lds_copy, reg, _lds_view(base_iter, row * fx.Int32(A_ROW_I32) + col // fx.Int32(4)))

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
            m8_f = arith.uitofp(T.f32, byte_i32_raw)
            one = arith.constant(1.0, type=T.f32)
            two56 = arith.constant(256.0, type=T.f32)
            factor = arith.addf(one, arith.divf(m8_f, two56))
            return fx.Float32(arith.divf(one, factor))

        def load_mbs(macro_kt):
            # A: one dword per mi = 4 contiguous row-bytes at
            # (macro_kt * m_pad32 + bx_m + mi*16 + lane_div_16*4).
            a_recip = []
            for mi in range_constexpr(m_chunks):
                # Uniform (per-workgroup, not per-lane) part goes through
                # readfirstlane to get a scalar SGPR base -- lane_div_16 is
                # per-lane and must be added AFTER, like load_sc()'s sc_lane.
                uniform_base = macro_kt * m_pad32 + bx_m + fx.Int32(mi * 16)
                byte_off = rocdl.readfirstlane(T.i32, uniform_base) + lane_div_16 * fx.Int32(4)
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
            # B: one byte per ni (column fixed per lane).
            n_pad32 = (N + 31) // 32 * 32
            b_recip = []
            for ni in range_constexpr(num_acc_n):
                col = by_n + wave * fx.Int32(BN // 4) + fx.Int32(ni * 16) + lane_mod_16
                byte_off = macro_kt * fx.Int32(n_pad32) + col
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

        accs_init = [Vec.filled(4, 0.0, Float32).ir_value() for _ in range_constexpr(n_acc)]

        coop_load_a(fx.Int32(0), _iter_of(fx.Int32(0)))
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
            coop_load_a(pf_kt, _iter_of(nxt))
            av = read_a(cur)
            bv = load_b(kt)
            sa_v, sb_v = load_sc(chunk_kt)
            # This K-tile covers k_halves consecutive 128-K macro blocks,
            # starting at kt * k_halves.
            macro_recips = [load_mbs(kt * fx.Int32(k_halves) + fx.Int32(kh)) for kh in range_constexpr(k_halves)]
            accs = compute_with_macro(accs, av, bv, sa_v, sb_v, macro_recips, scale_shift)
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
        ).launch(grid=(gx, gy, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm
