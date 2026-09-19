"""Selective GEFS downloader for the investigation's real-data experiments.

Fetches only the byte range of one GRIB2 message per (member, lead, variable) using the
upstream `.idx` sidecar, and caches each message under the system temp directory
(override with WEATHER_REALDATA_CACHE). Nothing is written into the repository.

Note on lead availability: the newest cycle is often partially published for the
perturbed members (e.g. 20260918 18Z only had leads 000..213), while 00Z cycles carry the
full 81 leads to f240. The lead comparison therefore uses one 00Z cycle for both leads,
which also gives an independent second cycle for reproducing the f006 numbers.

Run:  .venv/Scripts/python.exe gefs_fetch.py 20260918 00 f006 f240
"""

from __future__ import annotations

import os
import socket
import sys
import tempfile
import urllib.request

BASE = "https://noaa-gefs-pds.s3.amazonaws.com"
CACHE = os.environ.get(
    "WEATHER_REALDATA_CACHE",
    os.path.join(tempfile.gettempdir(), "weather_realdata"),
)

#: (idx variable token, idx level token, cfgrib shortName, cfgrib level)
VARIABLES = (
    ("TMP", "2 m above ground", "2t", 2),
    ("APCP", "surface", "tp", 0),
)


def _get(url: str, headers: dict[str, str] | None = None, timeout: int = 90) -> bytes:
    req = urllib.request.Request(url, headers=headers or {})
    return urllib.request.urlopen(req, timeout=timeout).read()


def fetch(date: str, cycle: str, lead: str, member: int, short: str) -> str:
    """Download just the wanted message; return the cached path."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{date}{cycle}_gep{member:02d}_{lead}_{short}.grib2")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path

    name = f"gep{member:02d}.t{cycle}z.pgrb2s.0p25.{lead}"
    prefix = f"gefs.{date}/{cycle}/atmos/pgrb2sp25/{name}"
    idx = _get(f"{BASE}/{prefix}.idx").decode()

    recs: list[tuple[int, str, str]] = []
    for line in idx.strip().split("\n"):
        p = line.split(":")
        if len(p) >= 5:
            recs.append((int(p[1]), p[3], p[4]))

    for var, lvl, want_short, _ in VARIABLES:
        if want_short != short:
            continue
        for i, (off, var_t, lvl_t) in enumerate(recs):
            if var_t == var and lvl_t == lvl:
                if i + 1 < len(recs):
                    end = recs[i + 1][0] - 1
                else:
                    head = urllib.request.Request(f"{BASE}/{prefix}", method="HEAD")
                    end = int(
                        urllib.request.urlopen(head, timeout=90).headers["Content-Length"]
                    ) - 1
                raw = _get(f"{BASE}/{prefix}", headers={"Range": f"bytes={off}-{end}"})
                with open(path, "wb") as fh:
                    fh.write(raw)
                return path
    raise KeyError(f"{short} not found for {date} {cycle} {lead} member {member}")


def main() -> None:
    if len(sys.argv) < 4:
        print(__doc__)
        return
    date, cycle = sys.argv[1], sys.argv[2]
    leads = sys.argv[3:]
    socket.setdefaulttimeout(90)
    total = 0
    for lead in leads:
        ok = 0
        for m in range(1, 31):
            for _, _, short, _ in VARIABLES:
                try:
                    total += os.path.getsize(fetch(date, cycle, lead, m, short))
                    ok += 1
                except KeyError:
                    print(f"  missing {short} for member {m}")
        print(f"{date} {cycle} {lead}: {ok} messages")
    print(f"cache {CACHE}, new bytes this run {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
