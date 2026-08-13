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
| Level | `heightAboveGround` level `2` |
| Data | Linear ramp `280.0..300.0` K (float32) |
| Size | 329 bytes |

The level was set to `heightAboveGround`/`2` (matching the real operational
GFS `2t` field) so `cfgrib` emits the data variable as **`t2m`** — the same
name the parser's `SURFACE_FIELD_FILTERS` and the pipeline's
`DEFAULT_VARIABLES` mapping expect. A `surface`-level `2t` message decodes
to a `t` variable name, which does not match real GFS output.

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
    codes_set(msg, "typeOfLevel", "heightAboveGround")
    codes_set(msg, "level", 2)
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

## `gfs_multi_typeoflevel.grib2`

A compact deterministic multi-message fixture that reproduces the structural
conditions of a real operational GFS `pgrb2.0p25` file that caused the
Milestone 1-14 production parser to fail. It is built with `eccodes` from
the `"GRIB2"` sample (same as the tiny fixture above) and committed so the
regression tests never touch the network.

### Why it exists
The production parser previously opened an entire GRIB2 file as one
unfiltered cfgrib dataset. A real operational `pgrb2.0p25` file contains
hundreds of messages spanning dozens of `typeOfLevel` values, so cfgrib
raises `DatasetBuildError: multiple values for unique key 'typeOfLevel'`.
This fixture is the minimal set of messages that reproduces that failure
structure without committing the ~545 MB operational file.

### Contents (5 messages, 4 distinct `typeOfLevel`, 2 `prate` stepTypes)

| # | shortName | typeOfLevel | level | stepType | step | Value (all grid cells) |
|---|---|---|---|---|---|---|
| 1 | `2t` | `heightAboveGround` | 2 | instant | 6 | `280.0` K |
| 2 | `prate` | `surface` | 0 | instant | 6 | `0.0003` kg m-2 s-1 |
| 3 | `prate` | `surface` | 0 | **avg** | 0-6 | `0.0001` kg m-2 s-1 |
| 4 | `t` | `isobaricInhPa` | 850 | instant | 6 | `250.0` K |
| 5 | `prmsl` | `meanSea` | 0 | instant | 6 | `101300.0` Pa |

### Properties the fixture reproduces
- **Multiple `typeOfLevel` values** (`heightAboveGround`, `surface`,
  `isobaricInhPa`, `meanSea`) — an unfiltered open raises cfgrib's
  `DatasetBuildError`.
- **Both `prate` `stepType` variants** (instant and avg) — a broad
  `shortName=prate` selection is ambiguous and must be disambiguated to
  `stepType=instant`.
- **Realistic `t2m` naming** — the `2t`/`heightAboveGround`/2 message
  decodes to a `t2m` variable (not `t`), matching real GFS output.
- **At least one unrelated field/level** (`t` at `isobaricInhPa/850`,
  `prmsl` at `meanSea`) so an unfiltered parser would be structurally
  invalid, exactly like the operational file.

### Size
919 bytes.

### Regeneration
```python
from pathlib import Path
import numpy as np
from eccodes import (
    codes_grib_new_from_samples, codes_release, codes_set,
    codes_set_values, codes_write,
)

OUT = Path("gfs_multi_typeoflevel.grib2")
Ni, Nj = 10, 5

def msg(f, sn, param, tol, level, stype, srange, value):
    m = codes_grib_new_from_samples("GRIB2")
    codes_set(m, "dataDate", 20260812)
    codes_set(m, "dataTime", 12)
    codes_set(m, "stepType", stype)
    codes_set(m, "stepRange", srange)
    codes_set(m, "stepUnits", "h")
    codes_set(m, "paramId", param)
    codes_set(m, "shortName", sn)
    codes_set(m, "typeOfLevel", tol)
    codes_set(m, "level", level)
    codes_set(m, "gridType", "regular_ll")
    codes_set(m, "Ni", Ni)
    codes_set(m, "Nj", Nj)
    codes_set(m, "latitudeOfFirstGridPointInDegrees", 40.0)
    codes_set(m, "longitudeOfFirstGridPointInDegrees", 250.0)
    codes_set(m, "latitudeOfLastGridPointInDegrees", 36.0)
    codes_set(m, "longitudeOfLastGridPointInDegrees", 259.0)
    codes_set(m, "iDirectionIncrementInDegrees", 1.0)
    codes_set(m, "jDirectionIncrementInDegrees", 1.0)
    codes_set_values(m, np.full((Nj, Ni), value, dtype=np.float32).ravel())
    codes_write(m, f)
    codes_release(m)

with OUT.open("wb") as f:
    msg(f, "2t", 167, "heightAboveGround", 2, "instant", "6", 280.0)
    msg(f, "prate", 7, "surface", 0, "instant", "6", 0.0003)
    msg(f, "prate", 7, "surface", 0, "avg", "0-6", 0.0001)
    msg(f, "t", 130, "isobaricInhPa", 850, "instant", "6", 250.0)
    msg(f, "prmsl", 151, "meanSea", 0, "instant", "6", 101300.0)
```
