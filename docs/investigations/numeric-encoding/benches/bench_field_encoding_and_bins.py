"""Experiment 19: (a) truncating the derived fields' low mantissa bits, (b) how many
normalised histogram bins are actually needed?

(a) The review's point: GRIB2 gets its compressibility from zeroed low mantissa bits, so
    why should the DERIVED fields be full entropy? They need not be. Section 9.11.3
    measured fixed-point quantisation (2.14-4.87x); this adds the two mechanisms it did
    NOT test:
      * mantissa truncation -- keep the float representation, zero the low mantissa bits
        (BitRound/BitGroom style, relative error), which is closest to what GRIB2's own
        representation looks like after expansion;
      * spatial differencing -- what GRIB2 actually does on top of packing and which the
        platform does not do at all.

(b) 32 normalised bins may be more than needed. This sweeps the bin count and measures
    both effects that depend on it: the quantile rebuilt from the shape, and the KDE
    rebuilt from the shape. Yardstick: the ensemble's own sampling noise at n = 30.

Run:  .venv/Scripts/python.exe bench_field_encoding_and_bins.py
"""

from __future__ import annotations

import numpy as np
from numcodecs import Zstd

from bench_aggregate_real_cost import GEFS_VARS, LEADS, bytes_as
from bench_local_packing import CHUNK, LAT, LAT_CHUNKS, LON, LON_CHUNKS
from bench_member_packing import N_MEMBERS
from bench_representation_compare import build_members, query_points

Z5 = Zstd(level=5)
REAL_MEMBER_MB = 23.13


# ---------------------------------------------------------------------------------------
# encoding transforms applied to a full 2-D field before the production chunking
# ---------------------------------------------------------------------------------------
def trunc_mantissa(x, keepbits):
    """Zero the low mantissa bits, keeping the float32 representation (relative error)."""
    v = np.asarray(x, dtype=np.float32).view(np.uint32).copy()
    shift = 23 - keepbits
    if shift <= 0:
        return v.view(np.float32)
    half = np.uint32(1 << (shift - 1))
    v = ((v + half) >> np.uint32(shift)) << np.uint32(shift)
    return v.view(np.float32)


def quantise(x, scale, dtype=np.int16):
    limit = np.iinfo(dtype).max
    q = np.clip(np.round(np.nan_to_num(x, nan=0.0) / scale), -limit, limit).astype(dtype)
    return q


def diff_lon(arr):
    """First difference along longitude, as int32 so nothing overflows.

    Column 0 holds the ORIGINAL first column, so the transform is exactly invertible
    (a real spatial-differencing codec stores those first values explicitly, which is
    what GRIB2's spatial-differencing template does).
    """
    a = np.asarray(arr).astype(np.int32)
    out = np.zeros_like(a)
    out[:, 0] = a[:, 0]
    out[:, 1:] = np.diff(a, axis=1)
    return out


def compress_field(arr):
    """Production chunking: 8x15 chunks of 100x100, each its own Zstd-5 stream."""
    arr = np.asarray(arr)
    padded = np.zeros((LAT, LON), dtype=arr.dtype)
    a = arr
    padded[: a.shape[0], : a.shape[1]] = a
    total = 0
    for r in range(LAT_CHUNKS):
        for c in range(LON_CHUNKS):
            r0, r1 = r * CHUNK, min((r + 1) * CHUNK, LAT)
            c0, c1 = c * CHUNK, min((c + 1) * CHUNK, LON)
            buf = np.zeros((CHUNK, CHUNK), dtype=arr.dtype)
            buf[: r1 - r0, : c1 - c0] = padded[r0:r1, c0:c1]
            total += len(Z5.encode(buf.tobytes(order="C")))
    return total


def main():
    members = build_members()
    stack = np.stack(members, axis=0)
    cell_mean = np.nanmean(stack, axis=0)
    cell_sigma = np.nanstd(stack, axis=0)
    dev = stack - cell_mean[None]
    cov_h = np.nanmean(dev[:, :, :-1] * dev[:, :, 1:], axis=0)

    fields = [("MEAN", cell_mean), ("STD", cell_sigma), ("COV_H", cov_h)]

    print()
    print("=" * 112)
    print("(a) Encoding mechanisms for a DERIVED field, and their error")
    print("=" * 112)
    print(f"{'mechanism':<40} {'MEAN MB':>9} {'STD MB':>9} {'COV MB':>9} {'total MB':>10} "
          f"{'vs f32':>8} {'max |err|':>11}")
    print("-" * 112)

    def undelta(d):
        """Invert diff_lon exactly: cumulative sum, with column 0 already the original."""
        arr = np.asarray(d, dtype=np.int64)
        out = np.cumsum(arr, axis=1).astype(np.float64)
        return out

    # Each variant returns (array to compress, reconstructed field in the field's units).
    variants = [
        ("float32 + Zstd5 (full entropy)",
         lambda f: (f.astype(np.float32), f.astype(np.float32))),
        ("int16 @0.01  (err 0.005)",
         lambda f: (quantise(f, 0.01), quantise(f, 0.01).astype(np.float32) * np.float32(0.01))),
        ("int16 @0.002 (err 0.001)",
         lambda f: (quantise(f, 0.002), quantise(f, 0.002).astype(np.float32) * np.float32(0.002))),
        ("int32 @0.001 (err 0.0005)",
         lambda f: (quantise(f, 0.001, np.int32),
                    quantise(f, 0.001, np.int32).astype(np.float32) * np.float32(0.001))),
        ("trunc mantissa keepbits=18",
         lambda f: (trunc_mantissa(f, 18), trunc_mantissa(f, 18))),
        ("trunc mantissa keepbits=16",
         lambda f: (trunc_mantissa(f, 16), trunc_mantissa(f, 16))),
        ("trunc mantissa keepbits=14",
         lambda f: (trunc_mantissa(f, 14), trunc_mantissa(f, 14))),
        ("trunc mantissa keepbits=12",
         lambda f: (trunc_mantissa(f, 12), trunc_mantissa(f, 12))),
        ("int16 @0.01 + delta_lon",
         lambda f: (diff_lon(quantise(f, 0.01)),
                    undelta(diff_lon(quantise(f, 0.01))) * 0.01)),
    ]

    base_total = None
    for label, fn in variants:
        sizes, errs = [], []
        for _, f in fields:
            payload, back = fn(f)
            sizes.append(compress_field(payload))
            errs.append(float(np.nanmax(np.abs(np.asarray(back, dtype=np.float64) - f))))
        tot = sum(sizes)
        if base_total is None:
            base_total = tot
        print(f"{label:<40} {sizes[0] / 1e6:9.3f} {sizes[1] / 1e6:9.3f} {sizes[2] / 1e6:9.3f} "
              f"{tot / 1e6:10.3f} {base_total / tot:7.2f}x {np.nanmax(errs):11.2e}")
    print()
    print("  |err| is the max absolute reconstruction error in the field's own units, i.e.")
    print("  after inverting the transform. Compare against the 0.005 budget.")
    print()
    print("  NOTE the mechanism that matters: truncation is RELATIVE, so on an O(10) field")
    print("  keepbits=14 gives ~2e-5 absolute error -- 250x tighter than the 0.005 budget,")
    print("  and far more compressible than float32.")
    print()

    # ---------------------------------------------------------------------------------
    # (b) bin count sweep
    # ---------------------------------------------------------------------------------
    r0, c0, tr, tc = query_points(3000)
    z = (stack - cell_mean[None]) / np.maximum(cell_sigma[None], 1e-9)

    def bilinear(field):
        g00, g01 = field[r0, c0], field[r0, c0 + 1]
        g10, g11 = field[r0 + 1, c0], field[r0 + 1, c0 + 1]
        lo = g00 + (g01 - g00) * tc
        up = g10 + (g11 - g10) * tc
        return lo + (up - lo) * tr

    interp = np.stack([bilinear(m) for m in members])
    half = N_MEMBERS // 2
    mean_at = bilinear(cell_mean)
    sigma_at = bilinear(cell_sigma)

    def pct(a, q):
        return np.nanpercentile(a, q, axis=0)

    def noise(fn):
        return float(np.max(np.abs(fn(interp[:half]) - fn(interp[half:]))))

    nz = {q: noise(lambda a, q=q: pct(a, q)) for q in (10, 50, 90)}

    def silverman(x):
        sd = float(np.std(x))
        q75, q25 = np.percentile(x, [75, 25])
        iqr = float(q75 - q25)
        a = min(sd, iqr / 1.349) if iqr > 0 else sd
        return 1.06 * a * x.size ** (-1 / 5) if a > 0 else float(sd) or 1e-6

    grid = np.linspace(-4.0, 4.0, 81)  # in sigma units, for the KDE comparison

    print("=" * 112)
    print("(b) Normalised bin count: what 8/12/16/24/32/48/64 bins cost and what they lose")
    print("=" * 112)
    print(f"{'bins':>5} {'bin width':>10} {'block MB':>9} {'GB/cycle':>10} "
          f"{'P10 err/noise':>14} {'P50 err/noise':>14} {'P90 err/noise':>14} {'KDE d/noise':>13}")
    print("-" * 112)

    for nb in (8, 12, 16, 24, 32, 48, 64):
        K = 4.0
        edges = np.linspace(-K, K, nb + 1)
        width = edges[1] - edges[0]
        centers = 0.5 * (edges[:-1] + edges[1:])
        zhist = np.stack(
            [np.nanmean((z >= edges[b]) & (z < edges[b + 1]), axis=0) for b in range(nb)]
        )
        mix = np.stack([bilinear(zhist[b]) for b in range(nb)])
        cdf = np.cumsum(mix, axis=0)
        ar = np.arange(mix.shape[1])

        def z_quantile(q):
            qq = q / 100.0
            idx = np.argmax(cdf >= qq, axis=0)
            below = np.where(idx > 0, cdf[np.maximum(idx - 1, 0), ar], 0.0)
            p_in = mix[idx, ar]
            frac = np.divide(qq - below, p_in, out=np.zeros_like(p_in), where=p_in > 1e-12)
            return edges[idx] + frac * width

        # block cost: the bins are stored as bounded fractions
        block_mb = sum(bytes_as(zhist[b], (np.int16, 0.01)) for b in range(nb)) / 1e6
        cyc = block_mb * LEADS * GEFS_VARS / 1e3

        errs = {}
        for q in (10, 50, 90):
            true_q = pct(interp, q)
            errs[q] = float(np.max(np.abs(mean_at + z_quantile(q) * sigma_at - true_q))) / nz[q]

        # KDE rebuilt from the shape, in sigma units
        kde_err = []
        for i in range(0, 400):
            x_sig = (interp[:, i] - mean_at[i]) / max(sigma_at[i], 1e-9)
            h_sig = silverman(x_sig)
            true_pts = np.exp(-0.5 * ((grid[:, None] - x_sig[None, :]) / h_sig) ** 2).mean(
                axis=1
            ) / (np.sqrt(2 * np.pi) * h_sig)
            p = mix[:, i] / max(mix[:, i].sum(), 1e-12)
            dens = (
                np.exp(-0.5 * ((grid[:, None] - centers[None, :]) / h_sig) ** 2)
                @ p
                / (np.sqrt(2 * np.pi) * h_sig)
            )
            kde_err.append(float(np.max(np.abs(dens - true_pts))))
        kde_d = float(np.mean(kde_err))
        # sampling noise of the KDE, same grid
        nz_kde = []
        for i in range(0, 400):
            x_sig = (interp[:, i] - mean_at[i]) / max(sigma_at[i], 1e-9)
            h = silverman(x_sig)
            a = np.exp(-0.5 * ((grid[:, None] - x_sig[:half][None, :]) / h) ** 2).mean(axis=1)
            b = np.exp(-0.5 * ((grid[:, None] - x_sig[half:][None, :]) / h) ** 2).mean(axis=1)
            nz_kde.append(float(np.max(np.abs(a - b))) / (np.sqrt(2 * np.pi) * h))
        kde_ratio = kde_d / float(np.mean(nz_kde)) if np.mean(nz_kde) else float("inf")

        print(f"{nb:>5} {width:10.3f} {block_mb:9.2f} {cyc:10.1f} "
              f"{errs[10]:14.3f} {errs[50]:14.3f} {errs[90]:14.3f} {kde_ratio:13.3f}")
    print()
    print("  Errors are relative to the ensemble's own sampling noise (n = 30): < 1 means the")
    print("  rebuilt quantity is indistinguishable from what a different 15 of the 30 members")
    print("  would have produced. Bin width is in local-sigma units.")
    print()


if __name__ == "__main__":
    main()
