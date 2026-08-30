# GFS & GEFS Non-Pressure-Level Variable Inventory Contract (0–240h, 3h Cadence)

## Overview

This directory contains the canonical machine-readable variable inventory and contract specifications for operational **NOAA GFS** and **NOAA GEFS** numerical weather prediction models, supporting the **0–240h forecast horizon at 3-hour cadence**.

The inventory covers all **non-pressure-level / surface-oriented** fields (surface, 2 m, 10 m, boundary layer, soil, atmospheric column, cloud layers, and radiation) relevant to the platform's 2D raster and point-forecast architecture.

Isobaric pressure-level / upper-air fields (e.g. 500 hPa HGT, 850 hPa TMP) and native model extensions beyond f240 are explicitly deferred and documented as future expansion context.

---

## Artifact Files

1. **`gfs_non_pressure_inventory.csv`** (189 rows):
   Complete non-pressure-level variable inventory extracted from operational NOAA GFS 0.25° `pgrb2.0p25` GRIB2 files across all 81 forecast leads (`f000, f003, ..., f240`).

2. **`gefs_non_pressure_inventory.csv`** (39 rows):
   Complete non-pressure-level variable inventory extracted from operational NOAA GEFS 0.25° `pgrb2s.0p25` GRIB2 files across all 31 ensemble members (`gec00` control + `gep01`–`gep30` perturbation) and all 81 forecast leads (`f000, f003, ..., f240`).
   *Evidence: Exhaustively verified on 2,511 distinct GRIB index files ($31 \text{ members} \times 81 \text{ leads}$). All 31 members have 100% identical variable, level, and step metadata.*

3. **`variable_contract.csv`** (53 rows):
   Master candidate variable contract linking upstream GFS and GEFS fields to canonical platform identifiers, units, conversion formulas, temporal semantics, and suitability tiers (Tier A: Strong Raw, Tier B: Derived, Tier C: Semantics/Interval Work, Tier D: Specialized).

---

## Schema Column Definitions

| Column | Description |
|---|---|
| `model` | Model identifier (`gfs` or `gefs`). |
| `upstream_product` | Upstream GRIB2 product file family (e.g. `pgrb2.0p25`, `pgrb2s.0p25`). |
| `upstream_identifier` | GRIB parameter mnemonic (e.g. `TMP`, `DPT`, `UGRD`, `PRATE`, `APCP`). |
| `upstream_long_name` | Full meteorological parameter name. |
| `canonical_candidate_name` | Proposed platform variable identifier (e.g. `temperature_2m`, `dewpoint_2m`, `wind_gust`). |
| `category` | Meteorological domain (Thermodynamic, Kinematic, Mass, Precipitation, Radiation, Soil, Convective). |
| `level_type` | Level type classification (`heightAboveGround`, `surface`, `meanSeaLevel`, `atmosphere`, `depthBelowGround`). |
| `level` | Upstream vertical level string (e.g. `2 m above ground`, `10 m above ground`, `surface`). |
| `units` | Upstream physical units. |
| `temporal_semantics` | Temporal character (`instant`, `accumulation`, `average`, `maximum`, `minimum`). |
| `step_type` | GRIB `stepType` attribute (`instant`, `accum`, `average`, `max`, `min`). |
| `step_range_pattern` | Description of step ranges across forecast leads (e.g. `0-6h resetting`, `3h/6h alternating`). |
| `accumulation_or_interval_window` | Nominal interval length over which the metric is computed. |
| `required_lead_min` | Earliest product lead (`0`). |
| `required_lead_max` | Latest product lead (`240`). |
| `required_lead_cadence` | Product lead cadence (`3h`). |
| `applicable_at_f000` | Boolean indicating whether field exists at analysis time `f000` (False for interval-dependent fields). |
| `lead_completeness` | Completeness description across the 81 product leads. |
| `member_scope` | Ensemble member scope (`deterministic` or `31 members (gec00, gep01..gep30)`). |
| `member_completeness` | Verification status across ensemble members. |
| `raw_upstream` | Boolean: True if direct upstream GRIB field. |
| `directly_ingestible` | Boolean: True if compatible with current 2D storage and single-message decoding. |
| `derived` | Boolean: True if computed from multiple upstream fields (e.g. wind speed from U and V). |
| `derivation_dependencies` | Upstream source fields required for derivation. |
| `ingestion_constraints` | Transformation or filtering constraints required for safe ingestion. |
| `current_repo_support` | Boolean: True if currently supported in repository code. |
| `current_repo_locations` | Source code files implementing the variable. |
| `product_usefulness` | Utility rating for weather platform users (`High`, `Medium`, `Low`). |
| `tier` | Implementation suitability tier (Tier A, Tier B, Tier C, Tier D). |
| `evidence_type` | Evidence classification: `OBSERVED_EXHAUSTIVE`, `OBSERVED_SAMPLED`, `DOCUMENTED`, `INFERRED`, `UNKNOWN`. |
| `evidence_reference` | Source URL / S3 key / index line supporting the entry. |
| `confidence` | Confidence level (`HIGH`). |
| `notes` | Meteorological, architectural, and operational notes. |

---

## Evidence Classification Summary

- **`OBSERVED_EXHAUSTIVE`**: 
  - GFS: Verified on all 81 leads (`f000, f003, ..., f240`) of `gfs.t00z.pgrb2.0p25`.
  - GEFS: Verified on all 31 members $\times$ all 81 leads ($2,511 \text{ index files}$) of `gepNN.t00z.pgrb2s.0p25`.
- **`INFERRED`**: Mathematical derivations (e.g. `wind_speed_10m` from `UGRD` + `VGRD`).
- **`DOCUMENTED`**: NCEP WMO parameter tables and GRIB2 documentation.
