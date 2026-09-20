"""Fetch the real GEFS messages the path-comparison bench needs.

One message per ``(member, lead, variable)``, pulled by byte range from the upstream
``.idx`` sidecar and cached under the system temp directory (``WEATHER_REALDATA_CACHE``
overrides). Nothing is written into the repository.

This is the wider sibling of ``gefs_fetch.py``, which fetches only 2 m temperature and
accumulated precipitation for the size experiments. The comparison bench needs every
variable whose container carries fields, plus the ten planes the precipitation and rose
groups read, which is what ``--set all`` downloads.

Run:  .venv/Scripts/python.exe gefs_fetch_all.py 20260920 00 f006 f003
"""

from __future__ import annotations

import os
import sys
import tempfile
import urllib.request

BASE = "https://noaa-gefs-pds.s3.amazonaws.com"
CACHE = os.environ.get(
    "WEATHER_REALDATA_CACHE",
    os.path.join(tempfile.gettempdir(), "weather_realdata"),
)

#: ``(idx variable, idx level, short name, cache tag)``. The cache tag is what the bench
#: looks files up by, so it is the *variable code*, not the GRIB name.
MESSAGES: tuple[tuple[str, str, str], ...] = (
    ("TMP", "2 m above ground", "temperature_2m"),
    ("APCP", "surface", "precipitation_amount_3h"),
    ("CRAIN", "surface", "crain"),
    ("CSNOW", "surface", "csnow"),
    ("CFRZR", "surface", "cfrzr"),
    ("CICEP", "surface", "cicep"),
    ("RH", "2 m above ground", "relative_humidity_2m"),
    ("GUST", "surface", "wind_gust"),
    ("VIS", "surface", "visibility"),
    ("SNOD", "surface", "snow_depth"),
    ("UGRD", "10 m above ground", "wind_u_10m"),
    ("VGRD", "10 m above ground", "wind_v_10m"),
    ("TCDC", "entire atmosphere", "cloud_cover_3h"),
    ("HGT", "cloud ceiling", "cloud_ceiling"),
)


def _get(url: str, headers: dict[str, str] | None = None, timeout: int = 120) -> bytes:
    request = urllib.request.Request(url, headers=headers or {})
    return urllib.request.urlopen(request, timeout=timeout).read()


def _readings(date: str, cycle: str, lead: str, member: int) -> list[tuple[int, str, str]]:
    """The idx's records as ``(byte offset, variable, level)``, for one member.

    The member matters: GRIB2 messages differ in length between members (the packed values and
    their scale factors differ), so one member's byte offsets are not another's. Reading the wrong
    member's index yields a truncated message, which decodes as a corrupt GRIB2 rather than as an
    obvious error.
    """
    name = f"gep{member:02d}.t{cycle}z.pgrb2s.0p25.{lead}"
    prefix = f"gefs.{date}/{cycle}/atmos/pgrb2sp25/{name}"
    index = _get(f"{BASE}/{prefix}.idx").decode()
    records: list[tuple[int, str, str]] = []
    for line in index.strip().split("\n"):
        parts = line.split(":")
        if len(parts) >= 5:
            records.append((int(parts[1]), parts[3], parts[4]))
    return records


def fetch(date: str, cycle: str, lead: str, member: int, tag: str, level: str) -> str:
    """Download one message; return the cached path."""
    del level  # the tag names the variable; the level comes from MESSAGES
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{date}{cycle}_gep{member:02d}_{lead}_{tag}.grib2")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path

    name = f"gep{member:02d}.t{cycle}z.pgrb2s.0p25.{lead}"
    prefix = f"gefs.{date}/{cycle}/atmos/pgrb2sp25/{name}"
    records = _readings(date, cycle, lead, member)
    wanted_variable, wanted_level = _wanted(tag)

    for position, (offset, variable, level_token) in enumerate(records):
        if variable != wanted_variable or level_token != wanted_level:
            continue
        if position + 1 < len(records):
            end = records[position + 1][0] - 1
        else:
            head = urllib.request.Request(f"{BASE}/{prefix}", method="HEAD")
            end = int(urllib.request.urlopen(head, timeout=120).headers["Content-Length"]) - 1
        raw = _get(f"{BASE}/{prefix}", headers={"Range": f"bytes={offset}-{end}"})
        with open(path, "wb") as handle:
            handle.write(raw)
        return path
    raise KeyError(f"{tag} not found for {date} {cycle} {lead} member {member}")


def _wanted(tag: str) -> tuple[str, str]:
    """The GRIB variable and level a cache tag corresponds to."""
    for variable, level, name in MESSAGES:
        if name == tag:
            return variable, level
    raise KeyError(tag)


def main() -> None:
    if len(sys.argv) < 4:
        print(__doc__)
        return
    date, cycle = sys.argv[1], sys.argv[2]
    leads = sys.argv[3:]
    members = range(1, 31)
    total = len(leads) * len(members) * len(MESSAGES)
    done = 0
    for lead in leads:
        for member in members:
            for variable, level, tag in MESSAGES:
                path = os.path.join(
                    CACHE, f"{date}{cycle}_gep{member:02d}_{lead}_{tag}.grib2"
                )
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    done += 1
                    continue
                try:
                    fetch(date, cycle, lead, member, tag, level)
                except Exception as exc:  # noqa: BLE001 - report and keep going
                    print(f"  FAILED {lead} m{member} {tag}: {type(exc).__name__}: {exc}")
                done += 1
                if done % 25 == 0:
                    print(f"  {done}/{total}", flush=True)


if __name__ == "__main__":
    main()
