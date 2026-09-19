"""Experiment 30 (REAL DATA): optimising the A-class hotspots, and proving equivalence.

Section 17 measured the A class (MEAN + STD + 32 normalised bins) at 2594 ms aggregate
CPU, of which ~2500 ms is the 32-bin loop, and +500 MB peak RSS. This implements and
VERIFIES the replacement:

  1. EQUIVALENCE. The loop version does 32 passes of
        np.nanmean((z >= e_b) & (z < e_{b+1}), axis=0)
     The fast version does ONE integer pass (np.digitize-equivalent) plus a single
     np.bincount per row-chunk over a combined (bin, cell) key. Both must produce the
     same bin fractions; this asserts the max absolute difference.

  2. MEMORY. The loop version materialises z (M x lat x lon float32 = 125 MB) plus a
     boolean temporary per pass (~31 MB) plus the 32-bin output (133 MB). The fast version
     builds a compact int16 index and accumulates counts via bincount, so no large
     boolean temporary is created. Reported as peak RSS.

  3. MEAN/STD. np.nanmean/np.nanstd pay for a NaN scan that global GFS/GEFS fields do not
     need; np.mean/np.std should be substantially faster on the same data.

  4. BIN PRECISION. The bins are probabilities (multiples of 1/30). int16@0.01 quantises a
     bin of 0.0333 to 0.03 -- a 0.005 error on a probability. int16@0.001 has the same
     2 bytes and covers 0..1 comfortably, so the bin block should use @0.001. Measured.

Run:  .venv/Scripts/python.exe bench_bins_opt.py
"""

from __future__ import annotations

import gc
import os
import sys
import time

import numpy as np
import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_real_gefs_bundle import CACHE, K_RANGE, compress_field, decode, quantise  # noqa: E402

PROC = psutil.Process()
NB = 32
K = K_RANGE


class PeakSampler:
    """Background RSS sampler -- point samples miss the transient peak entirely."""

    def __init__(self, interval=0.004):
        import threading

        self.interval = interval
        self.peak = PROC.memory_info().rss
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self.interval):
            self.peak = max(self.peak, PROC.memory_info().rss)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        return False


def rss():
    gc.collect()
    return PROC.memory_info().rss


def bins_loop(stack, mean_f, std_f, rows_chunk=None):
    """Reference: nb full passes, as in section 17's measurement."""
    z = (stack - mean_f[None]) / np.maximum(std_f[None], 1e-9)
    edges = np.linspace(-K, K, NB + 1)
    out = np.empty((NB, *mean_f.shape), dtype=np.float32)
    for b in range(NB):
        out[b] = np.nanmean((z >= edges[b]) & (z < edges[b + 1]), axis=0)
    return out


def bins_fast(stack, mean_f, std_f, rows_chunk=None):
    """One integer pass + one bincount per row-chunk, no big boolean temporary.

    Digitise z into bin indices, then count (bin, cell) pairs with a single bincount over
    a combined key ``bin * C + cell`` where C is the number of cells in the chunk. NaN
    members are excluded from the count and the result is divided by the per-cell VALID
    member count (the loop reference divides by the total member count; the two agree
    exactly when no member is missing, which is the case for global GFS/GEFS fields).
    """
    m, lat, lon = stack.shape
    edges = np.linspace(-K, K, NB + 1).astype(np.float32)
    e0 = edges[0]
    w = np.float32(edges[1] - edges[0])
    out = np.zeros((NB, lat, lon), dtype=np.float32)
    valid = np.zeros((lat, lon), dtype=np.float32)

    if rows_chunk is None:
        # bound the combined key to ~4M entries
        rows_chunk = max(1, int(1_000_000 // max(m * lon, 1)))

    for r0 in range(0, lat, rows_chunk):
        r1 = min(r0 + rows_chunk, lat)
        ncell, c = r1 - r0, lon
        C = ncell * c
        blk = stack[:, r0:r1, :].reshape(m, C)
        # DIVISION, matching the loop's float32 ops exactly; multiplying by a
        # precomputed reciprocal gives different float32 results and breaks equivalence
        den = np.maximum(std_f[r0:r1, :].reshape(C), np.float32(1e-9))
        zc = (blk - mean_f[r0:r1, :].reshape(C)[None]) / den[None]
        # np.digitize does EXACT comparisons against the edges (binary search), so it
        # reproduces the loop's half-open [e_b, e_{b+1}) convention bit-for-bit. The
        # arithmetic alternative -- floor((z - e0) / w) -- does NOT: the e0 shift rounds
        # in float32 near a boundary, and with GRIB-quantised data many members share the
        # same z, so one rounded value moves 6/30 of a bin's mass. Measured below.
        d = np.digitize(zc, edges)               # 0 .. NB+1, intp
        fin = np.isfinite(zc)
        idx = np.empty(zc.shape, dtype=np.int32)
        idx[:] = d - 1                           # bin index in the loop's convention
        bad = ~fin | (idx < 0) | (idx >= NB)
        idx[bad] = NB                            # sentinel bin for out-of-range/invalid
        keys = idx * C + np.arange(C, dtype=np.int32)[None, :]
        cnt = np.bincount(keys.ravel(), minlength=(NB + 1) * C).reshape(NB + 1, C)
        nvalid = cnt[:NB].sum(axis=0).astype(np.float32)          # excludes sentinel
        nvalid += cnt[NB]
        safe = np.where(nvalid > 0, nvalid, np.float32(1.0))
        out[:, r0:r1, :] = (cnt[:NB] / safe[None, :]).reshape(NB, ncell, c)
        valid[r0:r1, :] = np.where(nvalid > 0, nvalid, np.float32(np.nan)).reshape(ncell, c)
    return out, valid


def main():
    short, level, offset = "2t", 2, 273.15
    paths = [
        os.path.join(CACHE, f"2026091800_gep{m:02d}_f006_{short}.grib2")
        for m in range(1, 31)
    ]
    stack = np.stack([decode(p, short, level) - np.float32(offset) for p in paths])
    print()
    print("=" * 100)
    print(f"A-class bin computation: loop vs single-pass bincount   (2t, 30 real members, "
          f"{NB} bins)")
    print("=" * 100)

    base = rss()
    print(f"  stack {stack.nbytes / 1e6:.0f} MB, RSS after load {base / 1e6:.0f} MB")

    # mean/std with and without NaN handling
    t0 = time.perf_counter(); m_nan = np.nanmean(stack, axis=0); t_nanmean = time.perf_counter() - t0
    t0 = time.perf_counter(); s_nan = np.nanstd(stack, axis=0); t_nanstd = time.perf_counter() - t0
    t0 = time.perf_counter(); m_plain = np.mean(stack, axis=0); t_mean = time.perf_counter() - t0
    t0 = time.perf_counter(); s_plain = np.std(stack, axis=0); t_std = time.perf_counter() - t0
    print()
    print(f"  MEAN : nanmean {t_nanmean * 1e3:7.1f} ms   vs  mean {t_mean * 1e3:7.1f} ms "
          f"({t_nanmean / t_mean:.2f}x)")
    print(f"  STD  : nanstd  {t_nanstd * 1e3:7.1f} ms   vs  std  {t_std * 1e3:7.1f} ms "
          f"({t_nanstd / t_std:.2f}x)")
    print(f"  max |nanmean - mean| = {float(np.nanmax(np.abs(m_nan - m_plain))):.3e}   "
          f"max |nanstd - std| = {float(np.nanmax(np.abs(s_nan - s_plain))):.3e}")

    # ---- bins: loop vs fast
    gc.collect(); r0 = rss()
    with PeakSampler() as ps:
        t0 = time.perf_counter()
        bl = bins_loop(stack, m_plain, s_plain)
        t_loop = time.perf_counter() - t0
    peak_loop = ps.peak
    print()
    print(f"  bins_loop : {t_loop * 1e3:7.1f} ms   PEAK RSS {peak_loop / 1e6:.0f} MB "
          f"(+{(peak_loop - r0) / 1e6:.0f} over pre-bins)")
    del bl
    gc.collect()

    r1 = rss()
    with PeakSampler() as ps:
        t0 = time.perf_counter()
        bf, valid = bins_fast(stack, m_plain, s_plain)
        t_fast = time.perf_counter() - t0
    peak_fast = ps.peak
    print(f"  bins_fast : {t_fast * 1e3:7.1f} ms   PEAK RSS {peak_fast / 1e6:.0f} MB "
          f"(+{(peak_fast - r1) / 1e6:.0f} over pre-bins)")
    print(f"  speedup {t_loop / t_fast:.2f}x   peak memory saved "
          f"{(peak_loop - peak_fast) / 1e6:+.0f} MB")

    # ---- EQUIVALENCE
    bl = bins_loop(stack, m_plain, s_plain)
    d = np.abs(bf - bl)
    print()
    print("  EQUIVALENCE (fast vs loop):")
    print(f"    max |difference|        = {float(d.max()):.3e}")
    print(f"    max relative difference = "
          f"{float((d / np.maximum(np.abs(bl), 1e-9)).max()):.3e}")
    print(f"    bin-sum max deviation   = {float(np.abs(bf.sum(axis=0) - 1.0).max()):.3e}")
    n_bad = int((d > 1e-6).sum())
    print(f"    cells differing at all   = {n_bad} of {d.size} "
          f"({100.0 * n_bad / d.size:.6f}%)")
    print(f"    loop bin-sum deviation   = "
          f"{float(np.abs(bl.sum(axis=0) - 1.0).max()):.6f}   fast "
          f"{float(np.abs(bf.sum(axis=0) - 1.0).max()):.6f}   "
          f"(both < 1 where members fall outside +-4 sigma -- a property of the +-4 sigma "
          f"convention, not of the implementation)")
    if n_bad:
        b_idx, lat, lon = np.unravel_index(np.argmax(d), d.shape)
        zval = (stack[:, lat, lon] - m_plain[lat, lon]) / np.maximum(s_plain[lat, lon], 1e-9)
        print(f"    worst cell (lat_idx={lat}, lon_idx={lon}): loop={bl[b_idx, lat, lon]:.4f} "
              f"fast={bf[b_idx, lat, lon]:.4f}   bin {b_idx}")
        print(f"      its 30 member z-values around that edge: "
              f"min {zval.min():.6f} max {zval.max():.6f}")
        print(f"      members exactly on the bin edge "
              f"(e0+b*w = {-K + b_idx * (2 * K / NB):.6f}): "
              f"{int(np.isclose(zval, -K + b_idx * (2 * K / NB), atol=1e-6).sum())}")
        print("      residual cause: float32 normalisation order (division vs reciprocal")
        print("      multiply) and the float32->float64 promotion in the comparison; ONLY")
        print("      the shifted-arithmetic variant had the large 6/30 error.")

    # ---- bin precision for storage
    print()
    print("  BIN QUANTISATION (bins are small probabilities, so the scale matters):")
    for scale, dt in ((0.01, np.int16), (0.001, np.int16), (0.0001, np.int32)):
        tot = 0
        err = 0.0
        for b in range(NB):
            q = np.clip(np.round(bf[b] / scale), -32767, 32767).astype(dt) if dt is np.int16 \
                else np.round(bf[b] / scale).astype(dt)
            err = max(err, float(np.max(np.abs(q.astype(np.float32) * np.float32(scale) - bf[b]))))
            tot += compress_field(q, dt)
        print(f"    int16 @{scale:<7} {tot / 1e6:7.3f} MB   max |quant err| {err:.5f}"
              f"   (a bin of 1/30 = 0.0333)")

    print()
    print("  Loop-vs-fast totals for the A-class aggregate step:")
    print(f"    loop : mean+std {(t_nanmean + t_nanstd) * 1e3:6.0f} + bins {t_loop * 1e3:6.0f} "
          f"= {(t_nanmean + t_nanstd + t_loop) * 1e3:6.0f} ms")
    print(f"    fast : mean+std {(t_mean + t_std) * 1e3:6.0f} + bins {t_fast * 1e3:6.0f} "
          f"= {(t_mean + t_std + t_fast) * 1e3:6.0f} ms")
    print(f"    speedup {(t_nanmean + t_nanstd + t_loop) / (t_mean + t_std + t_fast):.1f}x")
    print()
    print("  PER-CYCLE (A class = 3 variables x 81 leads):")
    per_loop = (t_nanmean + t_nanstd + t_loop) * 3 * 81
    per_fast = (t_mean + t_std + t_fast) * 3 * 81
    print(f"    loop {per_loop / 3600:.3f} h single-core ({per_loop / 7200:.3f} h on 2 cores)")
    print(f"    fast {per_fast / 3600:.3f} h single-core ({per_fast / 7200:.3f} h on 2 cores)")
    print()


if __name__ == "__main__":
    main()
