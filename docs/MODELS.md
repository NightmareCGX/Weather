# Operational Weather Models

The Weather Platform ingests, processes, and serves global numerical weather prediction (NWP) model forecasts.

---

## 1. Supported Operational Models

### 1.1 NOAA GFS (Global Forecast System)
* **Status:** Operational
* **Grid Resolution:** 0.25° (~25 km) global regular rectilinear grid ($721 \times 1440$).
* **Cycle Cadence:** 4 times daily (00Z, 06Z, 12Z, 18Z).
* **Canonical Horizon:** 0 to 240 hours at 3-hour cadence (`domain.horizon`). Upstream extends to 384 hours.
* **Storage Layout:** `sharded_v1` single-lead binary shard containers (120 chunks of $100 \times 100$ per variable).
* **Upstream Sources:** AWS Open Data S3 (`noaa-gfs-bdp-pds`) with automated fallback to NOAA NOMADS HTTP.
* **Supported Variables:**
  * `temperature_2m` (2-Meter Temperature, °C)
  * `precipitation_rate` (Surface Precipitation Rate, mm/h)
  * `precipitation_amount_3h` (3-Hour De-accumulated Precipitation, mm)
  * `crain` (Categorical Rain Flag, 0/1)
  * `csnow` (Categorical Snow Flag, 0/1)
  * `wind_10m` (10-Meter Wind Speed & Direction, derived from $u/v$ components)

### 1.2 NOAA GEFS (Global Ensemble Forecast System)
* **Status:** Operational
* **Grid Resolution:** 0.50° (~50 km) global regular rectilinear grid ($361 \times 720$).
* **Cycle Cadence:** 4 times daily (00Z, 06Z, 12Z, 18Z).
* **Canonical Horizon:** 0 to 240 hours at 3-hour cadence (`domain.horizon`).
* **Ensemble Size:** 30 perturbation members (`gep01`–`gep30`).
* **Storage Layout:** `sharded_v1` per-member/per-lead binary shard containers (`{variable}/shard.mem{member:03d}_L{lead:04d}.shard`).
* **Upstream Sources:** AWS Open Data S3 (`noaa-gefs-pds`) with automated fallback to NOAA NOMADS HTTP (`pgrb2sp25`).
* **Ensemble Calculations:** Real-time calculation of ensemble mean, median, standard deviation, spread, interquartile range, percentiles (P10..P90), and empirical probability density functions (PDFs).

---

## 2. Future / Prospective Models (Unimplemented)

The following models are prospective roadmap targets and are **not currently implemented** in the repository:

* **ECMWF IFS & AIFS (European Centre for Medium-Range Weather Forecasts):** High-resolution deterministic (9 km) and machine-learning global forecasts.
* **ECCC GDPS & GEPS (Environment and Climate Change Canada):** Deterministic (15 km) and ensemble (35 km, 20 members) global forecasts.
* **NOAA HRRR (High-Resolution Rapid Refresh):** 3 km convection-allowing regional model over North America.

---

## 3. Data Ingestion & Normalization Standard

All incoming GRIB2 datasets are decoded via `cfgrib` and `ecCodes`, normalized into platform canonical units (temperature in °C, precipitation rate in mm/h, wind speed in km/h), and written into `sharded_v1` Zarr stores. Variable mappings are defined in `ingestion.core.wave_runner.DEFAULT_VARIABLES`.
