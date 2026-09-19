"""Selective GEFS fetch + cfgrib probe for the C-class variables.

C class = bounded / truncated / bimodal:
    cloud_ceiling       <- HGT at "cloud ceiling"        (capped: UI shows "Unlimited" >= 19.99 km)
    visibility          <- VIS at surface                (quantised, capped)
    cloud_cover_3h      <- TCDC entire atmosphere        (often bimodal at 0 / 100 %)
    relative_humidity_2m<- RH 2 m above ground           (bounded 0..100, near-Gaussian mid-range)

Probing matters here: each of these lives on a different GRIB level type, and cfgrib's
filter_by_keys has to match exactly. This downloads one member per variable and reports
what cfgrib actually finds (shortName, typeOfLevel, units, range, distinct-value count),
so the diagnostic script can use the right filters.

Run:  .venv/Scripts/python.exe gefs_fetch_c.py probe
      .venv/Scripts/python.exe gefs_fetch_c.py fetch
"""

from __future__ import annotations

import os
import socket
import sys
import tempfile
import urllib.request

BASE = "https://noaa-gefs-pds.s3.amazonaws.com"
CACHE = os.environ.get(
    "WEATHER_REALDATA_CACHE", os.path.join(tempfile.gettempdir(), "weather_realdata")
)
DATE, CYCLE = "20260918", "00"

#: (label, idx variable token, idx level token, shortName used in the cache key)
WANTED = (
    ("visibility", "VIS", "surface", "vis"),
    ("relative_humidity_2m", "RH", "2 m above ground", "r2"),
    ("cloud_cover_3h", "TCDC", "entire atmosphere", "tcc"),
    ("cloud_ceiling", "HGT", "cloud ceiling", "gh"),
)

#: cfgrib filters to try, most specific first. Filtering on ``shortName`` returns an
#: EMPTY dataset here (the range-GET file holds a single message, and shortName is not an
#: index key cfgrib will match against), so the strategy is: open with no key filter and
#: pick the variable by name.
FILTERS = {
    "vis": ({},),
    "r2": ({},),
    "tcc": ({},),
    "gh": ({},),
}

#: the variable name cfgrib is expected to expose for each cache key
EXPECTED = {"vis": "vis", "r2": "r2", "tcc": "tcc", "gh": "gh"}


def _get(url, headers=None, timeout=90):
    req = urllib.request.Request(url, headers=headers or {})
    return urllib.request.urlopen(req, timeout=timeout).read()


def fetch(lead: str, member: int, idx_var: str, idx_level: str, short: str) -> str:
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{DATE}{CYCLE}_gep{member:02d}_{lead}_{short}.grib2")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    name = f"gep{member:02d}.t{CYCLE}z.pgrb2s.0p25.{lead}"
    prefix = f"gefs.{DATE}/{CYCLE}/atmos/pgrb2sp25/{name}"
    idx = _get(f"{BASE}/{prefix}.idx").decode()
    recs = []
    for line in idx.strip().split("\n"):
        p = line.split(":")
        if len(p) >= 5:
            recs.append((int(p[1]), p[3], p[4]))
    for i, (off, v, lv) in enumerate(recs):
        if v == idx_var and lv == idx_level:
            if i + 1 < len(recs):
                end = recs[i + 1][0] - 1
            else:
                head = urllib.request.Request(f"{BASE}/{prefix}", method="HEAD")
                end = int(urllib.request.urlopen(head, timeout=90).headers["Content-Length"]) - 1
            raw = _get(f"{BASE}/{prefix}", headers={"Range": f"bytes={off}-{end}"})
            with open(path, "wb") as fh:
                fh.write(raw)
            return path
    raise KeyError(f"{idx_var}:{idx_level}")


def decode(path: str, short: str):
    import numpy as np
    import xarray as xr

    last = None
    for keys in FILTERS[short]:
        try:
            with xr.open_dataset(
                path, engine="cfgrib", backend_kwargs={"filter_by_keys": keys}
            ) as ds:
                vars_ = list(ds.data_vars)
                if not vars_:
                    continue
                name = EXPECTED[short] if EXPECTED[short] in vars_ else vars_[0]
                da = ds[name]
                arr = np.asarray(da.values)
                return name, da.dtype, dict(da.attrs), arr
        except Exception as exc:  # noqa: BLE001 - probing, want to see every failure
            last = exc
    raise RuntimeError(f"no filter matched for {short}: {last}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "probe"
    socket.setdefaulttimeout(90)
    import numpy as np

    if mode == "probe":
        for label, idx_var, idx_level, short in WANTED:
            try:
                p = fetch("f006", 1, idx_var, idx_level, short)
                name, dtype, attrs, arr = decode(p, short)
                finite = arr[np.isfinite(arr)]
                vals = np.unique(finite)
                print(
                    f"{label:<22} idx={idx_var}:{idx_level:<20} -> cfgrib name={name!r} "
                    f"{dtype} shape={arr.shape}"
                )
                print(
                    f"{'':<22} units={attrs.get('units')!r} GRIB_units={attrs.get('GRIB_units')!r} "
                    f"range=[{finite.min():.4f}, {finite.max():.4f}] "
                    f"distinct={vals.size} nan={100 * np.isnan(arr).mean():.1f}%"
                )
                print(
                    f"{'':<22} zero-frac={float((finite == 0).mean()):.4f} "
                    f"at-max-frac={float((finite == finite.max()).mean()):.4f}"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"{label:<22} FAILED: {type(exc).__name__}: {exc}")
        return

    # fetch mode: all 30 members for both leads
    for lead in ("f006", "f240"):
        ok = 0
        for label, idx_var, idx_level, short in WANTED:
            for m in range(1, 31):
                try:
                    fetch(lead, m, idx_var, idx_level, short)
                    ok += 1
                except KeyError:
                    pass
        print(f"{lead}: {ok} messages cached")


if __name__ == "__main__":
    main()
