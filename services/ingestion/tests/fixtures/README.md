# GRIB2 Test Fixtures

## `gfs.t00z.pgrb2.0p25.f006.grib2`

A tiny, deterministic GRIB2 fixture used by the Milestone 5 parser and
Zarr round-trip tests. It is generated offline with `eccodes` (the same
library `cfgrib` uses under the hood) and committed so tests never touch
the network.

### Contents
| Property | Value |
|---|---|
| Model / product | GFS-like, 2-m temperature (`paramId 167`, `2t`) |
| Grid | Regular lat/lon, `10 x 5` points |
| Latitudes | `40.0, 39.0, 38.0, 37.0, 36.0` (north→south, uniform 1° step) |
| Longitudes | `250.0, 251.0, ..., 259.0` (10 native GRIB points, uniform 1° step, decoded in the `0..360` convention; `250..259` = `-110..-101`) |
| Step | `+6 h` (`stepRange=6`, `stepUnits=h`, instant) |
| Cycle | `20260721 00Z` (`dataDate=20260721`, `dataTime=0`) |
| Level | Surface |
| Data | Linear ramp `280.0..300.0` K (float32) |
| Size | 329 bytes |

> The original committed fixture was internally inconsistent: its grid
> metadata declared `latitudeOfFirstGridPointInDegrees=44.0`,
> `jDirectionIncrementInDegrees=1.0`, `Nj=5` (implying a last latitude of
> `40.0`) but stored `latitudeOfLastGridPointInDegrees=0.0`. `cfgrib`
> decoded that contradiction into a non-uniform latitude axis
> (`[44, 43, 42, 41, 0]`), which the API serving tier correctly rejects
> ("grid must be uniformly spaced"). The corrected fixture sets consistent
> first/last grid points so the decoded axis is uniformly spaced.

### Regeneration
This fixture can be regenerated deterministically with:

```python
from pathlib import Path
import numpy as np
from eccodes import (
    codes_grib_new_from_samples,
    codes_release,
    codes_set,
    codes_set_values,
    codes_write,
)

OUT = Path("gfs.t00z.pgrb2.0p25.f006.grib2")
Ni, Nj = 10, 5
values = np.linspace(280.0, 300.0, Ni * Nj, dtype=np.float32).reshape(Nj, Ni)

with OUT.open("wb") as f:
    msg = codes_grib_new_from_samples("GRIB2")
    codes_set(msg, "dataDate", 20260721)
    codes_set(msg, "dataTime", 0)
    codes_set(msg, "stepType", "instant")
    codes_set(msg, "stepRange", "6")
    codes_set(msg, "stepUnits", "h")
    codes_set(msg, "paramId", 167)
    codes_set(msg, "shortName", "2t")
    codes_set(msg, "typeOfLevel", "surface")
    codes_set(msg, "gridType", "regular_ll")
    codes_set(msg, "Ni", Ni)
    codes_set(msg, "Nj", Nj)
    codes_set(msg, "latitudeOfFirstGridPointInDegrees", 40.0)
    codes_set(msg, "longitudeOfFirstGridPointInDegrees", 250.0)
    codes_set(msg, "latitudeOfLastGridPointInDegrees", 36.0)
    codes_set(msg, "longitudeOfLastGridPointInDegrees", 259.0)
    codes_set(msg, "iDirectionIncrementInDegrees", 1.0)
    codes_set(msg, "jDirectionIncrementInDegrees", 1.0)
    codes_set_values(msg, values)
    codes_write(msg, f)
    codes_release(msg)
```

> Note: eccodes rejects a negative `jDirectionIncrement` on the sample
> grid, so the fixture uses `lat0=40.0` with a positive increment, which
> `cfgrib` decodes as a north→south latitude axis. Both the first and last
> grid points must be set so the decoded axis is uniformly spaced.
