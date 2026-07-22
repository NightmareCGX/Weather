# Operational Weather Models

The platform integrates deterministic and ensemble numerical weather prediction (NWP) models from major meteorological centers.

---

## 1. Phase 1 Models: NOAA (USA)

### 1.1 GFS (Global Forecast System)
- **Resolution**: ~25 km (0.25 degree) horizontal grid, 127 vertical levels.
- **Cycle Frequency**: 4 times daily (00Z, 06Z, 12Z, 18Z).
- **Forecast Horizon**: Up to 384 hours (16 days).
- **Format**: GRIB2 files via NOAA NOMADS / AWS S3 buckets.
- **Key Variables**: Temperature, wind speed/direction, geopotential height, relative humidity, precipitation rate, mean sea level pressure.

### 1.2 GEFS (Global Ensemble Forecast System)
- **Resolution**: ~25 km to 50 km horizontal grid.
- **Cycle Frequency**: 4 times daily (00Z, 06Z, 12Z, 18Z).
- **Ensemble Size**: 1 control member + 30 perturbation members.
- **Forecast Horizon**: Up to 384 hours.
- **Purpose**: Powers initial ensemble spread, probability calculations, and confidence intervals.

---

## 2. Phase 2 Models: ECMWF & Canada

### 2.1 ECMWF IFS & AIFS (Europe)
- **Resolution**: ~9 km (IFS deterministic) / AI-driven global models (AIFS).
- **Cycle Frequency**: 2 times daily (00Z, 12Z).
- **Access**: ECMWF Meteorological Archival and Retrieval System (MARS) / Copernicus Climate Data Store (CDS).

### 2.2 Canadian Meteorological Centre (ECCC)
- **GDPS (Global Deterministic Prediction System)**: ~15 km resolution, 2x daily.
- **GEPS (Global Ensemble Prediction System)**: Ensemble forecasts, ~35 km resolution, 20+ members.
- **Access**: ECCC Datamart (HTTPS / AWS Open Data).

---

## 3. Data Ingestion & Normalization Standard
Regardless of the upstream source (NOAA, ECMWF, ECCC), all raw GRIB2 datasets are parsed via `cfgrib` and `xarray`, mapped to standardized CF-conventions variable names, and written to our unified Zarr storage format.
