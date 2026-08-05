"""
Standalone NumPy reference for Overflow-Aware Scaling (OAS) and Macro Block
Scaling (MBS), following arXiv:2603.08713 Sec 4.2/4.3. No GPU, no FlyDSL —
this is Phase 2 of docs/PROGRESS.md: verify our understanding of the algorithm
reproduces the paper's *claimed direction and rough magnitude* of QSNR gains
on synthetic data before writing any kernel code.

Run: python3 oas_mbs_reference.py
"""
import numpy as np

FP4_GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)
FP4_MAX = 6.0
rng = np.random.default_rng(0)


def qsnr_db(ref: np.ndarray, approx: np.ndarray) -> float:
    num = np.sum(ref.astype(np.float64) ** 2)
    den = np.sum((ref.astype(np.float64) - approx.astype(np.float64)) ** 2)
    if den == 0:
        return float("inf")
    return 10.0 * np.log10(num / den)


def quantize_fp4_grid(x: np.ndarray) -> np.ndarray:
    """Round to nearest signed E2M1 grid point, clipping at +-6.0."""
    sign = np.sign(x)
    mag = np.clip(np.abs(x), 0.0, FP4_MAX)
    idx = np.argmin(np.abs(mag[..., None] - FP4_GRID[None, :]), axis=-1)
    return sign * FP4_GRID[idx]


def block_scale_baseline(absmax: np.ndarray) -> np.ndarray:
    """Paper Sec 4.2 'standard computation': SF_fp32 = 6/absmax, then round
    DOWN to the nearest power of two (E8M0 masks mantissa bits) so absmax maps
    into (3, 6] -- i.e. never overflows/clips the block max."""
    absmax = np.maximum(absmax, 1e-12)
    sf_fp32 = FP4_MAX / absmax
    e = np.floor(np.log2(sf_fp32))
    return np.power(2.0, e)


def block_scale_oas(absmax: np.ndarray) -> np.ndarray:
    """Baseline scale, then apply OAS (Sec 4.2): if the baseline scale would
    map absmax into (3, 3.5], doubling the scale maps it to (6, 7] instead --
    clipped to 6 at the same *relative* error, but halves quantization error
    for every other (smaller-magnitude) element in the block."""
    sf = block_scale_baseline(absmax)
    scaled_max = absmax * sf
    bump = (scaled_max > 3.0) & (scaled_max <= 3.5)
    return np.where(bump, sf * 2.0, sf)


def quantize_mxfp4_block(x: np.ndarray, block: int, use_oas: bool) -> np.ndarray:
    """x: [..., N], quantize in contiguous blocks of `block` along last axis."""
    n = x.shape[-1]
    assert n % block == 0
    xb = x.reshape(*x.shape[:-1], n // block, block)
    absmax = np.max(np.abs(xb), axis=-1, keepdims=True)
    sf = block_scale_oas(absmax) if use_oas else block_scale_baseline(absmax)
    xq = quantize_fp4_grid(xb * sf) / sf
    return xq.reshape(x.shape)


def mbs_factor_static(macro_absmax: np.ndarray) -> np.ndarray:
    """Eq. 3: extract 8 MSBs of mantissa of (6/macro_absmax) as m8_MBS, return
    (1 + m8/256) in [1, 2)."""
    macro_absmax = np.maximum(macro_absmax, 1e-12)
    target = (FP4_MAX / macro_absmax).astype(np.float32)
    bits = target.view(np.uint32)
    m8 = ((bits & np.uint32(0x007F8000)) >> np.uint32(15)).astype(np.float64)
    return 1.0 + m8 / 256.0


def mbs_factor_dynamic(x_macro: np.ndarray, block: int, use_oas: bool, n_candidates: int = 16) -> np.ndarray:
    """Eq. 4-6: search candidate MBS factors (1 + j/n_candidates) and keep the
    one minimizing SSE of the resulting 16-sub-block MXFP4(+OAS) quantization
    over the whole macro block. x_macro: [..., 128]."""
    best_factor = np.ones(x_macro.shape[:-1], dtype=np.float64)
    best_sse = np.full(x_macro.shape[:-1], np.inf)
    for j in range(n_candidates):
        factor = 1.0 + j / n_candidates
        scaled = x_macro * factor
        xq = quantize_mxfp4_block(scaled, block, use_oas) / factor
        sse = np.sum((xq - x_macro) ** 2, axis=-1)
        improve = sse < best_sse
        best_sse = np.where(improve, sse, best_sse)
        best_factor = np.where(improve, factor, best_factor)
    return best_factor


def quantize_mbs(x: np.ndarray, block: int, macro_block: int, use_oas: bool, dynamic: bool) -> np.ndarray:
    n = x.shape[-1]
    assert n % macro_block == 0
    xm = x.reshape(*x.shape[:-1], n // macro_block, macro_block)
    macro_absmax = np.max(np.abs(xm), axis=-1)
    if dynamic:
        factor = mbs_factor_dynamic(xm, block, use_oas)
    else:
        factor = mbs_factor_static(macro_absmax)
    factor = factor[..., None]
    xq = quantize_mxfp4_block(xm * factor, block, use_oas) / factor
    return xq.reshape(x.shape)


def make_gaussian(rows: int, cols: int) -> np.ndarray:
    return rng.standard_normal((rows, cols))


def make_outlier_heavy(rows: int, cols: int, outlier_frac: float = 0.01, outlier_scale: float = 25.0) -> np.ndarray:
    """Heavy-tailed / outlier-injected synthetic tensor: mostly small Gaussian
    values with a rare (~1%) fraction of large-magnitude outliers, mimicking
    LLM activation outlier behavior the paper motivates MBS with (Sec 4.3)."""
    x = rng.standard_normal((rows, cols)) * 0.5
    mask = rng.random((rows, cols)) < outlier_frac
    x = np.where(mask, rng.choice([-1, 1], size=x.shape) * (outlier_scale + rng.standard_normal(x.shape)), x)
    return x


def report(name: str, x: np.ndarray):
    print(f"\n=== {name}  shape={x.shape} ===")
    base16 = quantize_mxfp4_block(x, block=16, use_oas=False)
    oas16 = quantize_mxfp4_block(x, block=16, use_oas=True)
    mbs_s = quantize_mbs(x, block=16, macro_block=128, use_oas=True, dynamic=False)
    mbs_d = quantize_mbs(x, block=16, macro_block=128, use_oas=True, dynamic=True)

    q_base = qsnr_db(x, base16)
    q_oas = qsnr_db(x, oas16)
    q_mbs_s = qsnr_db(x, mbs_s)
    q_mbs_d = qsnr_db(x, mbs_d)

    print(f"MXFP4-16 (no OAS):      QSNR = {q_base:6.2f} dB")
    print(f"MXFP4-16 + OAS:         QSNR = {q_oas:6.2f} dB  (delta vs base: {q_oas - q_base:+.2f} dB)")
    print(f"MXFP4-16 + OAS + MBS-S: QSNR = {q_mbs_s:6.2f} dB  (delta vs OAS: {q_mbs_s - q_oas:+.2f} dB)")
    print(f"MXFP4-16 + OAS + MBS-D: QSNR = {q_mbs_d:6.2f} dB  (delta vs OAS: {q_mbs_d - q_oas:+.2f} dB)")

    absmax = np.max(np.abs(x.reshape(-1, 16)), axis=-1)
    baseline_scaled_max = absmax * block_scale_baseline(absmax)
    bumped = (baseline_scaled_max > 3.0) & (baseline_scaled_max <= 3.5)
    print(f"Fraction of 16-blocks OAS bumps: {bumped.mean() * 100:.1f}%  (paper reports ~15%)")
    return q_base, q_oas, q_mbs_s, q_mbs_d


if __name__ == "__main__":
    print("Hypothesis #1: OAS improves QSNR over plain MXFP4-16 by ~0.5 dB, "
          "affecting ~15% of blocks.")
    print("Hypothesis #2: MBS-Static adds ~+1.1 dB over MXFP4-16-OAS; "
          "MBS-Dynamic adds ~+1.6 dB over MXFP4-16-OAS (paper Sec 4.3.3).")

    report("Gaussian activation-like tensor", make_gaussian(4096, 4096))
    report("Outlier-heavy tensor (1% outliers @ ~25x scale)", make_outlier_heavy(4096, 4096))

    print("\n=== Macro-block-size ablation (Appendix A): expect near-monotonic")
    print("    QSNR decrease as macro block grows from 32 to 512, with 128 a")
    print("    good compromise (paper: ~96% of MBS=32's QSNR retained at 128).")
    x = make_gaussian(2048, 2048)
    oas = qsnr_db(x, quantize_mxfp4_block(x, block=16, use_oas=True))
    for mb in (32, 64, 128, 256, 512):
        q = qsnr_db(x, quantize_mbs(x, block=16, macro_block=mb, use_oas=True, dynamic=False))
        print(f"  macro_block={mb:4d}: QSNR = {q:6.2f} dB  (vs OAS-only baseline {oas:6.2f} dB, "
              f"delta {q - oas:+.2f} dB)")
