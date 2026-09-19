"""Experiment 16: representation comparisons.

Part A (review Q2) -- error and bytes side by side for the three representations:
    30 members  vs  MEAN+VAR  vs  MEAN+VAR+4 neighbour covariances.

Part B (review Q1) -- can matrix decomposition help?
    (i) Per-cell eigendecomposition of the local 2x2 covariance block is strictly WORSE:
        4 eigenvalues + 16 eigenvector entries = 20 numbers per cell vs 6 fields, and
        eigenvectors are sign/rotation ambiguous and spatially discontinuous, so they
        compress badly. Not worth measuring beyond noting the count.
    (ii) The viable decomposition is along the MEMBER axis: EOF/SVD of the member matrix.
        If the members' anomalies live in a low-dimensional subspace, storing the leading
        patterns plus the (tiny) coefficient matrix can beat storing all 30 members, and
        it PRESERVES per-member access -- which the aggregate bundle destroys.
        The decisive number is the effective rank, which depends on how the real GEFS
        perturbations were generated, so this measures the tradeoff for a range of ranks.

Part C (review Q3) -- KDE from a stored histogram: how many bins are needed?

Run:  .venv/Scripts/python.exe bench_representation_compare.py
"""

from __future__ import annotations

import numpy as np
from numcodecs import Zstd

from bench_local_packing import LAT, LAT_CHUNKS, LON, LON_CHUNKS
from bench_member_packing import N_MEMBERS, chunk_buffers
from bench_render_outputs import bytes_of
from bench_shard_and_ensemble import grib_pack
from bench_stat_interpolation import smooth

Z5 = Zstd(level=5)
RNG = np.random.default_rng(1618)


def build_members(spread_ratio=0.20, n_modes=None, seed=11):
    """Smooth base + member perturbations.

    n_modes=None -> each member gets an independent perturbation (effective rank 30).
    n_modes=r    -> perturbations are combinations of r basis patterns, which is how an
                    ensemble generated from a finite perturbation set behaves.
    """
    lat = np.linspace(90.0, -90.0, LAT)[:, None]
    base = 12.0 + 9.0 * smooth((LAT, LON), 12.0, seed)
    sigma = np.full((LAT, LON), 9.0 * spread_ratio)

    if n_modes is None:
        perts = [smooth((LAT, LON), 8.0, 9000 + m) for m in range(N_MEMBERS)]
    else:
        basis = [smooth((LAT, LON), 8.0, 9500 + i) for i in range(n_modes)]
        basis = np.stack(basis)
        basis /= basis.reshape(n_modes, -1).std(axis=1)[:, None, None] + 1e-12
        coeffs = RNG.standard_normal((N_MEMBERS, n_modes))
        perts = [np.tensordot(coeffs[m], basis, axes=1) for m in range(N_MEMBERS)]
        # normalise so the ensemble spread matches the independent case
        p = np.stack(perts).reshape(N_MEMBERS, -1)
        p /= p.std(axis=0, keepdims=True) + 1e-12
        perts = [p[m].reshape(LAT, LON) for m in range(N_MEMBERS)]

    out = []
    for m in range(N_MEMBERS):
        f = base + perts[m] * sigma
        out.append(grib_pack(np.asarray(f, dtype=np.float32), 12))
    return out


def neighbours(stack):
    """MEAN, VAR and the four distinct neighbour covariances (NaN-aware)."""
    mean = np.nanmean(stack, axis=0)
    dev = stack - mean[None]
    var = np.nanmean(dev * dev, axis=0)

    def cov(dr, dc):
        a = dev[:, : LAT - dr, : LON - abs(dc)]
        b = dev[:, dr:, abs(dc) :]
        return np.nanmean(a * b, axis=0)

    return mean, var, cov(0, 1), cov(1, 0), cov(1, 1)


def query_points(n=4000):
    rows = RNG.uniform(1.0, LAT - 3.0, n)
    cols = RNG.uniform(2.0, LON - 3.0, n)
    r0, c0 = rows.astype(int), cols.astype(int)
    return r0, c0, rows - r0, cols - c0


def bilinear(field, r0, c0, tr, tc):
    g00, g01 = field[r0, c0], field[r0, c0 + 1]
    g10, g11 = field[r0 + 1, c0], field[r0 + 1, c0 + 1]
    lo = g00 + (g01 - g00) * tc
    up = g10 + (g11 - g10) * tc
    return lo + (up - lo) * tr


def main():
    members = build_members()
    stack = np.stack(members, axis=0)
    r0, c0, tr, tc = query_points()

    # ground truth: interpolate every member, then the statistic
    interp = np.stack([bilinear(m, r0, c0, tr, tc) for m in members])
    true_mean = np.nanmean(interp, axis=0)
    true_std = np.nanstd(interp, axis=0)
    half = N_MEMBERS // 2
    noise_std = float(
        np.max(np.abs(np.nanstd(interp[:half], axis=0) - np.nanstd(interp[half:], axis=0)))
    )

    member_bytes = bytes_of(list(members))
    mean, var, ch, cv, cd = neighbours(stack)
    # anti-diagonal covariance Cov(f(i,j+1), f(i+1,j)) -> shape (LAT-1, LON-1)
    mean_ = mean
    dev = stack - mean_[None]
    ca = np.nanmean(dev[:, 1:, :-1] * dev[:, :-1, 1:], axis=0)

    mv_bytes = bytes_of([mean, var])
    mv4_bytes = bytes_of([mean, var, ch, cv, cd, ca])

    w00, w01 = (1 - tc) * (1 - tr), tc * (1 - tr)
    w10, w11 = (1 - tc) * tr, tc * tr

    # MEAN+VAR path: interpolate the two fields
    std_mv = np.sqrt(np.maximum(bilinear(var, r0, c0, tr, tc), 0.0))
    # MEAN+VAR+4cov path: rebuild Var from the stored local covariances
    V = var
    var_exact = (
        w00**2 * V[r0, c0]
        + w01**2 * V[r0, c0 + 1]
        + w10**2 * V[r0 + 1, c0]
        + w11**2 * V[r0 + 1, c0 + 1]
        + 2 * w00 * w01 * ch[r0, c0]
        + 2 * w00 * w10 * cv[r0, c0]
        + 2 * w00 * w11 * cd[r0, c0]
        + 2 * w01 * w10 * ca[r0, c0]
        + 2 * w01 * w11 * cv[r0, c0 + 1]
        + 2 * w10 * w11 * ch[r0 + 1, c0]
    )
    std_mv4 = np.sqrt(np.maximum(var_exact, 0.0))

    print()
    print("=" * 100)
    print("Part A. Error and bytes: 30 members  vs  MEAN+VAR  vs  MEAN+VAR+4cov")
    print("=" * 100)
    print(f"  (std sampling noise of the 30-member ensemble = {noise_std:.4f})")
    print()
    print(f"{'representation':<26} {'fields':>7} {'MB':>9} {'vs current':>11} "
          f"{'std max err':>12} {'err/noise':>10}")
    print("-" * 100)
    rows_ = [
        ("30 members (current)", 30, member_bytes, 0.0),
        ("MEAN + VAR", 2, mv_bytes, float(np.max(np.abs(std_mv - true_std)))),
        ("MEAN + VAR + 4 covariances", 6, mv4_bytes, float(np.max(np.abs(std_mv4 - true_std)))),
    ]
    for label, nf, nb, err in rows_:
        ratio = err / noise_std if noise_std else 0.0
        print(f"{label:<26} {nf:>7} {nb / 1e6:9.2f} {member_bytes / nb:10.2f}x "
              f"{err:12.2e} {ratio:10.5f}")
    print()
    print(f"  the 4 covariance fields cost {(mv4_bytes - mv_bytes) / 1e6:.2f} MB, i.e. "
          f"{(mv4_bytes - mv_bytes) / member_bytes * 100:.1f}% of the current store,")
    print("  to move the std error from 0.4% of sampling noise to numerical zero.")
    print()

    # ---------------------------------------------------------------- Part B
    print("=" * 100)
    print("Part B. Can matrix decomposition help? EOF along the member axis")
    print("=" * 100)
    for label, modes in (("independent perturbations", None), ("rank-12 perturbation basis", 12)):
        mem = build_members(n_modes=modes)
        M = np.stack(mem, axis=0).reshape(N_MEMBERS, -1).astype(np.float64)
        mmean = M.mean(axis=0, keepdims=True)
        A = M - mmean
        # economy SVD via the 30x30 Gram matrix
        G = A @ A.T
        evals, evecs = np.linalg.eigh(G)
        order = np.argsort(evals)[::-1]
        evals, evecs = evals[order], evecs[:, order]
        total = evals.sum()
        cum = np.cumsum(evals) / total
        eff = int(np.searchsorted(cum, 0.999) + 1)
        print(f"  {label}: variance captured by the first k modes")
        for k in (4, 8, 12, 16, 20, 24, 30):
            print(f"      k={k:>3}  {cum[k - 1] * 100:7.3f}%")
        print(f"    -> effective rank (99.9% variance) = {eff}")
        print()

    # The storage/error tradeoff for truncation, on the independent (worst-case) ensemble.
    mem = build_members()
    M = np.stack(mem, axis=0).reshape(N_MEMBERS, -1).astype(np.float64)
    mmean = M.mean(axis=0)
    A = M - mmean
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    memb_bytes = bytes_of(list(mem))
    print(f"  truncation tradeoff (worst case: independent perturbations, full rank 30)")
    print(f"  {'k':>4} {'patterns MB':>12} {'coeff KB':>10} {'total MB':>10} {'vs current':>11} "
          f"{'std err':>10} {'err/noise':>10}")
    print("-" * 100)
    for k in (4, 8, 12, 16, 20, 24, 30):
        approx = (U[:, :k] * S[:k]) @ Vt[:k] + mmean
        ap = [np.asarray(row.reshape(LAT, LON), dtype=np.float32) for row in approx]
        # patterns are the k right singular vectors (spatial fields); coefficients 30 x k
        pat_bytes = bytes_of([np.asarray(Vt[i].reshape(LAT, LON), np.float32) for i in range(k)])
        coeff_kb = N_MEMBERS * k * 4 / 1024
        tot = pat_bytes + coeff_kb * 1024
        ip = np.stack([bilinear(f, r0, c0, tr, tc) for f in ap])
        err = float(np.max(np.abs(np.nanstd(ip, axis=0) - true_std)))
        print(f"  {k:>4} {pat_bytes / 1e6:12.2f} {coeff_kb:10.1f} {tot / 1e6:10.2f} "
              f"{memb_bytes / tot:10.2f}x {err:10.2e} {err / noise_std:10.5f}")
    print()
    print("  Note: the coefficient matrix (30 x k) is negligible at ~4 bytes each; the cost")
    print("  is the k spatial patterns. EOF keeps PER-MEMBER access (reconstruct on read),")
    print("  which the aggregate bundle gives up. But truncation error grows fast when the")
    print("  members are genuinely independent -- the real question is the true rank of the")
    print("  GEFS perturbation subspace, which must be measured on real members with")
    print("  SVD of the member matrix.")
    print()

    # ---------------------------------------------------------------- Part C
    sample_all = interp  # (30 members, n_pts): the member distribution at each point
    print("=" * 100)
    print("Part C. KDE rebuilt from a stored histogram: how many bins?")
    print("=" * 100)
    # The displayed PDF is a Gaussian KDE (Silverman bandwidth) over the 30 member
    # values at the point (domain/ensemble/pdf.py:87-120). Rebuilding it from a stored
    # histogram means kernel-smoothing the bin masses. This measures the fidelity as a
    # function of the bin count, against the sampling-noise floor.
    grid = np.linspace(sample_all.min(), sample_all.max(), 100)

    def silverman(x):
        n = x.size
        sd = float(np.std(x))
        q75, q25 = np.percentile(x, [75, 25])
        iqr = float(q75 - q25)
        a = min(sd, iqr / 1.349) if iqr > 0 else sd
        return 1.06 * a * n ** (-1 / 5) if a > 0 else None

    def kde_on(x, h, at=None):
        pts = grid if at is None else at
        d = (pts[:, None] - x[None, :]) / h
        return (np.exp(-0.5 * d**2) / np.sqrt(2 * np.pi)).mean(axis=1) / h

    def safe_h(x):
        h = silverman(x)
        return h if h and h > 0 else float(np.std(x)) * 0.5 or 1e-6

    def binned_kde(x, h, nb):
        lo, hi = float(x.min()), float(x.max())
        pad = 0.05 * (hi - lo) if hi > lo else 0.5
        edges = np.linspace(lo - pad, hi + pad, nb + 1)
        counts, _ = np.histogram(x, bins=edges)
        p = counts / counts.sum()
        centers = 0.5 * (edges[:-1] + edges[1:])
        d = (grid[:, None] - centers[None, :]) / h
        return (np.exp(-0.5 * d**2) / np.sqrt(2 * np.pi)) @ p / h

    n_use = 300
    samples = [sample_all[:, i] for i in range(n_use)]

    noise_curves = []
    for x in samples:
        h = safe_h(x)
        noise_curves.append(np.abs(kde_on(x[:half], h) - kde_on(x[half:], h)))
    noise_peak = float(np.max(noise_curves))
    peaks = [float(kde_on(x, safe_h(x)).max()) for x in samples]
    mean_peak = float(np.mean(peaks))

    print(f"  ({n_use} points; mean KDE peak = {mean_peak:.4f}; max sampling-noise |dKDE| = {noise_peak:.4f})")
    print()
    print(f"  {'bins':>5} {'bin width':>11} {'max |dKDE|':>12} {'% of KDE peak':>14} "
          f"{'vs sampling noise':>18} {'verdict':>22}")
    print("-" * 100)
    x_ref = samples[0]
    span = float(x_ref.max() - x_ref.min())
    for nb in (8, 12, 16, 24, 32, 48, 64, 96):
        diffs = []
        for x in samples:
            h = safe_h(x)
            diffs.append(np.abs(binned_kde(x, h, nb) - kde_on(x, h)))
        mx = float(np.max(diffs))
        ratio = mx / noise_peak if noise_peak else float("inf")
        verdict = "hidden in the noise" if ratio < 1 else "VISIBLE above the noise"
        print(f"  {nb:>5} {span / nb:11.3f} {mx:12.5f} {100 * mx / mean_peak:13.2f}% "
              f"{ratio:18.3f} {verdict:>22}")
    print()
    print("  'vs sampling noise' < 1 means the binned reconstruction cannot be told apart")
    print("  from the KDE that a different 15 of the 30 members would have produced.")
    print()


if __name__ == "__main__":
    main()
