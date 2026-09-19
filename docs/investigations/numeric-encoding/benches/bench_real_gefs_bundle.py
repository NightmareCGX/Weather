"""Experiment 20 (REAL DATA): the recommended bundle measured on actual GEFS members.

Everything in sections 3-9.13 was measured on synthetic member fields, with the
synthetic-vs-real compressibility gap corrected analytically in 9.11. This replaces that
with a like-for-like measurement on real GEFS 0.25 deg members.

Method:
  * selective range-fetch of just the wanted GRIB2 message from NOAA's anonymous S3
    bucket, for all 30 perturbation members of one cycle and one lead;
  * decode with cfgrib to 721x1440 float32 -- exactly what ingestion does;
  * baseline = the CURRENT representation: each member's field chunked 100x100 with
    NaN padding and compressed with Zstd(5) per chunk, i.e. sharded_v1;
  * bundle   = MEAN + STD + 16 normalised histogram bins (+ optional 4 covariances),
    encoded with fixed-point int16@0.01, with and without intra-field spatial
    differencing, chunked and compressed the same way;
  * also re-validates the accuracy claim on real data: rebuild P10/P90 from the
    normalised shape and compare against the ensemble's own sampling noise.

Two variables are used on purpose:
  * TMP 2 m -- temperature, close to Gaussian: the FAVOURABLE case;
  * APCP    -- precipitation accumulation, zero-inflated and heavy-tailed: the case
    where the "shape + location + scale" reconstruction is most likely to fail.

Run:  .venv/Scripts/python.exe bench_real_gefs_bundle.py
"""

from __future__ import annotations

import os
import socket
import tempfile
import time
import urllib.request

import numpy as np
import psutil
from numcodecs import Zstd

from bench_local_packing import CHUNK, LAT, LAT_CHUNKS, LON, LON_CHUNKS

Z5 = Zstd(level=5)
N_MEMBERS = 30
N_BINS = 16
K_RANGE = 4.0
SCALE = 0.01

BASE = "https://noaa-gefs-pds.s3.amazonaws.com"
DATE, CYCLE, LEAD = "20260918", "18", "f006"
#: Download cache, kept OUTSIDE the repository: a cycle x 2 variables is ~22 MB of
#: GRIB2 and must not be committed. Override with WEATHER_REALDATA_CACHE.
CACHE = os.environ.get(
    "WEATHER_REALDATA_CACHE", os.path.join(tempfile.gettempdir(), "weather_realdata")
)

# (label, idx variable token, idx level token, cfgrib shortName, level, unit)
VARIABLES = [
    ("2t (temperature)", "TMP", "2 m above ground", "2t", 2, "K -> degC"),
    ("apcp (6h precip)", "APCP", "surface", "tp", 0, "kg m-2 -> mm"),
]

PROC = psutil.Process()
PEAK_RSS = [PROC.memory_info().rss]


def note_rss():
    PEAK_RSS[0] = max(PEAK_RSS[0], PROC.memory_info().rss)
    return PEAK_RSS[0]


def fetch(url, headers=None, timeout=90):
    req = urllib.request.Request(url, headers=headers or {})
    return urllib.request.urlopen(req, timeout=timeout).read()


def fetch_message(member, var_token, level_token, short):
    """Selective range GET of one GRIB2 message, cached on disk."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{DATE}{CYCLE}_gep{member:02d}_{LEAD}_{short}.grib2")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path, os.path.getsize(path)

    name = f"gep{member:02d}.t{CYCLE}z.pgrb2s.0p25.{LEAD}"
    prefix = f"gefs.{DATE}/{CYCLE}/atmos/pgrb2sp25/{name}"
    idx = fetch(f"{BASE}/{prefix}.idx").decode()
    recs = []
    for line in idx.strip().split("\n"):
        p = line.split(":")
        if len(p) >= 5:
            recs.append((int(p[0]), int(p[1]), p[3], p[4]))
    for i, (_, off, var, lvl) in enumerate(recs):
        if var == var_token and lvl == level_token:
            if i + 1 < len(recs):
                nxt = recs[i + 1][1]
            else:
                req = urllib.request.Request(f"{BASE}/{prefix}", method="HEAD")
                nxt = int(urllib.request.urlopen(req, timeout=90).headers["Content-Length"])
            raw = fetch(
                f"{BASE}/{prefix}", headers={"Range": f"bytes={off}-{off + nxt - off - 1}"}
            )
            with open(path, "wb") as fh:
                fh.write(raw)
            return path, len(raw)
    raise KeyError(f"{var_token}:{level_token} not found for member {member}")


def decode(path, short, level):
    import xarray as xr

    keys = {"shortName": short}
    if short == "2t":
        keys.update({"typeOfLevel": "heightAboveGround", "level": level})
    else:
        keys.update({"typeOfLevel": "surface"})
    with xr.open_dataset(path, engine="cfgrib", backend_kwargs={"filter_by_keys": keys}) as ds:
        name = "t2m" if short == "2t" else "tp"
        arr = np.asarray(ds[name].values, dtype=np.float32)
    return arr


def compress_field(arr, dtype=np.float32):
    """sharded_v1 chunking: 8x15 chunks of 100x100, NaN-padded, one Zstd-5 stream each."""
    padded = np.full((LAT, LON), np.nan, dtype=dtype)
    a = arr
    a = a[: min(a.shape[0], LAT), : min(a.shape[1], LON)]
    padded[: a.shape[0], : a.shape[1]] = a
    total = 0
    for r in range(LAT_CHUNKS):
        for c in range(LON_CHUNKS):
            r0, r1 = r * CHUNK, min((r + 1) * CHUNK, LAT)
            c0, c1 = c * CHUNK, min((c + 1) * CHUNK, LON)
            buf = np.full((CHUNK, CHUNK), np.nan, dtype=dtype)
            buf[: r1 - r0, : c1 - c0] = padded[r0:r1, c0:c1]
            total += len(Z5.encode(buf.tobytes(order="C")))
    return total


def quantise(x):
    q = np.round(np.nan_to_num(x, nan=0.0) / SCALE)
    q = np.clip(q, -32767, 32767)
    return np.where(np.isnan(x), np.int16(-32768), q.astype(np.int16)).astype(np.int16)


def diff_lon(q):
    """First difference along longitude exactly invertibly, in the NARROWEST dtype.

    Storing the difference in int32 doubles the raw size (2 -> 4 bytes per value) and that
   膨胀 swamps the compression gain -- on the first real-data run it made the result WORSE
    than not differencing at all. The difference of a smooth field over one grid step is
    tiny (tens of quanta), so int16 normally suffices; this checks the range and only
    widens when it must.
    """
    a = q.astype(np.int32)
    out = np.zeros_like(a)
    out[:, 0] = a[:, 0]
    out[:, 1:] = np.diff(a, axis=1)
    lo, hi = int(out.min()), int(out.max())
    if -32768 <= lo and hi <= 32767:
        return out.astype(np.int16), "int16"
    return out, "int32"


def bundle_fields(stack, with_cov):
    """MEAN, STD, N_BINS normalised bins, optionally the 4 neighbour covariances."""
    mean = np.nanmean(stack, axis=0)
    std = np.nanstd(stack, axis=0)
    z = (stack - mean[None]) / np.maximum(std[None], 1e-9)
    edges = np.linspace(-K_RANGE, K_RANGE, N_BINS + 1)
    bins = [np.nanmean((z >= edges[b]) & (z < edges[b + 1]), axis=0) for b in range(N_BINS)]
    covs = []
    if with_cov:
        dev = stack - mean[None]

        def c(dr, dc):
            a = dev[:, : LAT - dr, : LON - abs(dc)]
            b = dev[:, dr:, abs(dc) :]
            return np.nanmean(a * b, axis=0)

        covs = [c(0, 1), c(1, 0), c(1, 1),
                np.nanmean(dev[:, 1:, :-1] * dev[:, :-1, 1:], axis=0)]
    return mean, std, bins, covs, z


def main():
    socket.setdefaulttimeout(90)
    print()
    print("=" * 108)
    print(f"REAL DATA: GEFS 0.25 deg, {DATE} {CYCLE}Z, {LEAD}, {N_MEMBERS} perturbation members")
    print("=" * 108)

    # ---------------------------------------------------------------- fetch + decode
    t0 = time.perf_counter()
    decoded = {}
    downloaded = 0
    for label, var, lvl_tok, short, level, unit in VARIABLES:
        fields = []
        for m in range(1, N_MEMBERS + 1):
            path, size = fetch_message(m, var, lvl_tok, short)
            downloaded += size
            fields.append(decode(path, short, level))
        decoded[label] = (np.stack(fields), unit)
        print(f"  fetched+decoded {label:<22} {len(fields)} members, shape {fields[0].shape}")
    t1 = time.perf_counter()
    print(f"  wall {t1 - t0:.1f} s, downloaded {downloaded / 1e6:.1f} MB, peak RSS {note_rss() / 1e6:.0f} MB")
    print()

    for label, (stack, unit) in decoded.items():
        finite = stack[np.isfinite(stack)]
        print(f"  {label:<22} range [{finite.min():.3f}, {finite.max():.3f}] {unit}  "
              f"mean {finite.mean():.3f}  zero-fraction {float((finite == 0).mean()):.3f}")

    # ---------------------------------------------------------------- per variable
    for label, (stack, unit) in decoded.items():
        import gc

        gc.collect()
        rss_before = PROC.memory_info().rss
        print()
        print("=" * 108)
        print(f"{label}   ({unit})   stack = {stack.nbytes / 1e6:.0f} MB resident")
        print("=" * 108)

        # baseline: the current representation, one shard per member
        t0 = time.perf_counter()
        member_bytes = sum(compress_field(stack[m]) for m in range(N_MEMBERS))
        t1 = time.perf_counter()
        base_ms = 1000 * (t1 - t0)
        base_mb = member_bytes / 1e6
        print(f"  BASELINE  {N_MEMBERS} member shards (Zstd5, production chunking)")
        print(f"            {base_mb:8.3f} MB per (variable, lead)   encode {base_ms:7.1f} ms")
        print(f"            per shard {base_mb / N_MEMBERS * 1e3:.1f} KB")
        print()

        # per-group breakdown, so each rung of the bundle ladder is priced separately
        mean_f0, std_f0, bins0, covs0, _ = bundle_fields(stack, True)
        grp = {
            "MEAN (1 field)": [mean_f0],
            "STD (1 field)": [std_f0],
            f"{N_BINS} normalised bins": bins0,
            "4 covariances": covs0,
        }
        print("  PER-GROUP COST (int16@0.01, production chunking)")
        run = 0
        for gname, gf in grp.items():
            gb = sum(compress_field(quantise(f), np.int16) for f in gf) / 1e6
            run += gb
            print(f"            {gname:<26} {len(gf):>2} fields  {gb:8.3f} MB  "
                  f"cumulative {run:8.3f} MB")
        print(f"            {'MEAN+STD only':<26} {2:>2} fields  "
              f"{(sum(compress_field(quantise(f), np.int16) for f in grp['MEAN (1 field)'] + grp['STD (1 field)'])) / 1e6:8.3f} MB  "
              f"-> {base_mb / (sum(compress_field(quantise(f), np.int16) for f in grp['MEAN (1 field)'] + grp['STD (1 field)']) / 1e6):5.2f}x")
        print()

        for with_cov in (False, True):
            tag = "with 4 covariances" if with_cov else "without covariances"

            t0 = time.perf_counter()
            mean_f, std_f, bins, covs, z = bundle_fields(stack, with_cov)
            t1 = time.perf_counter()
            compute_ms = 1000 * (t1 - t0)
            peak_after_compute = note_rss()

            plain = [mean_f, std_f] + bins + covs
            t0 = time.perf_counter()
            bytes_plain = sum(compress_field(quantise(f), np.int16) for f in plain)
            t1 = time.perf_counter()
            enc_plain = 1000 * (t1 - t0)

            t0 = time.perf_counter()
            bytes_delta = 0
            dtypes = set()
            for f in plain:
                d, dt = diff_lon(quantise(f))
                dtypes.add(dt)
                bytes_delta += compress_field(d, np.int16 if dt == "int16" else np.int32)
            t1 = time.perf_counter()
            enc_delta = 1000 * (t1 - t0)

            nf = len(plain)
            print(f"  BUNDLE [{tag}]  {nf} fields "
                  f"(MEAN + STD + {N_BINS} bins{(' + 4 cov' if with_cov else '')})")
            print(f"            int16@0.01               {bytes_plain / 1e6:8.3f} MB  "
                  f"({base_mb / (bytes_plain / 1e6):5.2f}x)  encode {enc_plain:7.1f} ms")
            print(f"            int16@0.01 + delta_lon   {bytes_delta / 1e6:8.3f} MB  "
                  f"({base_mb / (bytes_delta / 1e6):5.2f}x)  encode {enc_delta:7.1f} ms  "
                  f"[delta dtype: {','.join(sorted(dtypes))}]")
            print(f"            aggregate compute {compute_ms:7.1f} ms   "
                  f"peak RSS {peak_after_compute / 1e6:.0f} MB")
            print()

        # -------------------------------------------------- accuracy on real data
        rng = np.random.default_rng(2026)
        pts = 2000
        rows = rng.uniform(1, LAT - 3, pts)
        cols = rng.uniform(2, LON - 3, pts)
        r0, c0 = rows.astype(int), cols.astype(int)
        tr, tc = rows - r0, cols - c0

        def bil(f):
            g00, g01 = f[r0, c0], f[r0, c0 + 1]
            g10, g11 = f[r0 + 1, c0], f[r0 + 1, c0 + 1]
            lo = g00 + (g01 - g00) * tc
            up = g10 + (g11 - g10) * tc
            return lo + (up - lo) * tr

        interp = np.stack([bil(stack[m]) for m in range(N_MEMBERS)])
        half = N_MEMBERS // 2

        def noise(fn):
            return float(np.max(np.abs(fn(interp[:half]) - fn(interp[half:]))))

        mean_at = bil(mean_f)
        std_at = bil(std_f)
        edges = np.linspace(-K_RANGE, K_RANGE, N_BINS + 1)
        width = edges[1] - edges[0]
        centers = 0.5 * (edges[:-1] + edges[1:])
        mix = np.stack([bil(b) for b in bins])
        cdf = np.cumsum(mix, axis=0)
        ar = np.arange(pts)

        def zq(q):
            qq = q / 100.0
            idx = np.argmax(cdf >= qq, axis=0)
            below = np.where(idx > 0, cdf[np.maximum(idx - 1, 0), ar], 0.0)
            p_in = mix[idx, ar]
            frac = np.divide(qq - below, p_in, out=np.zeros_like(p_in), where=p_in > 1e-12)
            return edges[idx] + frac * width

        print(f"  ACCURACY on real members (max error over {pts} points, "
              f"relative to the ensemble's own n=30 sampling noise)")
        for q in (10, 50, 90):
            true_q = np.nanpercentile(interp, q, axis=0)
            e = float(np.max(np.abs(mean_at + zq(q) * std_at - true_q)))
            nz = noise(lambda a, q=q: np.nanpercentile(a, q, axis=0))
            print(f"            P{q:<2}  err {e:9.4f}   noise {nz:9.4f}   ratio {e / nz:6.3f}")
        e_sig = float(np.max(np.abs(std_at - np.nanstd(interp, axis=0))))
        nz_sig = noise(lambda a: np.nanstd(a, axis=0))
        print(f"            STD  err {e_sig:9.4f}   noise {nz_sig:9.4f}   ratio {e_sig / nz_sig:6.3f}")

        # what the 4 covariance fields buy: the interpolated variance becomes exact
        mean_c, std_c, bins_c, covs_c, _ = bundle_fields(stack, True)
        V = std_c ** 2
        w00, w01 = (1 - tc) * (1 - tr), tc * (1 - tr)
        w10, w11 = (1 - tc) * tr, tc * tr
        var_exact = (
            w00**2 * V[r0, c0] + w01**2 * V[r0, c0 + 1]
            + w10**2 * V[r0 + 1, c0] + w11**2 * V[r0 + 1, c0 + 1]
            + 2 * w00 * w01 * covs_c[0][r0, c0]
            + 2 * w00 * w10 * covs_c[1][r0, c0]
            + 2 * w00 * w11 * covs_c[2][r0, c0]
            + 2 * w01 * w10 * covs_c[3][r0, c0]
            + 2 * w01 * w11 * covs_c[1][r0, c0 + 1]
            + 2 * w10 * w11 * covs_c[0][r0 + 1, c0]
        )
        e_cov = float(
            np.max(np.abs(np.sqrt(np.maximum(var_exact, 0.0)) - np.nanstd(interp, axis=0)))
        )
        print(f"            STD  err {e_cov:9.4f}   (rebuilt from the 4 covariances)   "
              f"ratio {e_cov / nz_sig:6.4f}")
        print(f"            -> covariances take the STD error from {e_sig / nz_sig:.3f} to "
              f"{e_cov / nz_sig:.4f} of noise, at the byte cost shown above")
        print()
        note_rss()
        print(f"  peak RSS after this variable: {PEAK_RSS[0] / 1e6:.0f} MB "
              f"(+{(PEAK_RSS[0] - rss_before) / 1e6:.0f} MB over pre-variable)")
        print()

    note_rss()
    print("=" * 108)
    print(f"PEAK RSS over the whole experiment: {PEAK_RSS[0] / 1e6:.0f} MB")
    print("=" * 108)
    print()


if __name__ == "__main__":
    main()
