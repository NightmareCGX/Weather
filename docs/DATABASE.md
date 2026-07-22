# Final Database Architecture Review & Specification: Global Probabilistic Weather Platform

---

## 1. Why `lead_time_hours` is Preferred Over `valid_time` in NWP Systems

In numerical weather prediction (NWP) and ensemble forecasting systems, storing forecasts relative to **`lead_time_hours`** (offset hours from the model run's `cycle_time`) rather than absolute **`valid_time`** timestamps provides critical engineering and scientific advantages:

1. **Inherent Scientific Alignment**: NWP models are initialized at precise cycle times (e.g., 00Z, 06Z, 12Z, 18Z) and generate outputs indexed by forecast hour steps (e.g., `+00`, `+03`, `+06`, `+12`, ..., `+240`). Storing `lead_time_hours` directly mirrors the native GRIB2 message headers and Zarr dataset coordinate structure (`step`), eliminating complex time-conversion overhead during ingestion.
2. **Re-run & Backfill Idempotency**: If a model run cycle is re-ingested or corrected, its absolute valid timestamps would shift if keyed by `valid_time`, whereas its relative `lead_time_hours` remain invariant. This guarantees absolute data integrity and clean upserts.
3. **Deterministic Derivation**: Absolute valid time can always be computed deterministically on-the-fly (`valid_time = cycle_time + lead_time_hours`), preventing data redundancy and time zone synchronization bugs.

---

## 2. Final Schema Changes

- **`forecast_products` Table**: Replaced `valid_time` with **`lead_time_hours`** (INTEGER).
- **Unique Constraint Update**: 
  ```sql
  UNIQUE (run_id, variable_id, grid_id, product_type, lead_time_hours)
  ```
- **`point_query_fallback_audit` Table**: Explicitly designated as a Redis-fallback and point-query audit ledger. Optimized with `cache_key` as primary key and a dedicated `BTREE` index on `expires_at` for background TTL cleanup workers.

---

## 3. Final Relationship Map

```
[forecast_centers] 1 ──< [models] 1 ──< [model_versions] 1 ──< [model_runs]
                                                                  │
                                           ┌──────────────────────┴──────────────────────┐
                                           │ (1:N)                                       │ (1:N)
                                           v                                             v
                                  [ensemble_members]                            [forecast_products]
                                                                                         │
                                           ┌─────────────────────────────────────────────┘
                                           │ (N:1)                                       │ (N:1)
                                           v                                             v
                                  [forecast_variables]                           [forecast_grids]
```

---

## 4. Final Unique Constraints

1. **`forecast_centers`**: `UNIQUE (center_id)`
2. **`models`**: `UNIQUE (model_id)`
3. **`model_versions`**: `UNIQUE (model_id, version_string)`
4. **`model_runs`**: `UNIQUE (model_version_id, cycle_time)`
5. **`ensemble_members`**: `UNIQUE (run_id, member_index)`
6. **`forecast_variables`**: `UNIQUE (variable_code)`
7. **`forecast_grids`**: `UNIQUE (grid_code)`
8. **`forecast_products`**: `UNIQUE (run_id, variable_id, grid_id, product_type, lead_time_hours)`

---

## 5. Final Indexing Recommendations

1. **Spatial Indexes (`GIST`)**:
   - `stations(geom)`
   - `cities(geom)`
   - `ski_resorts(geom)`
2. **Temporal & Relational Indexes (`BTREE`)**:
   - `model_runs(model_version_id, cycle_time DESC)` — Fast retrieval of the latest model cycle.
   - `forecast_products(run_id, variable_id, grid_id)` — Rapid catalog filtering.
   - `point_query_fallback_audit(expires_at)` — Dedicated index for efficient TTL cleanup of stale fallback cache records.
   - `verification_observations(station_id, valid_time DESC)` — Efficient historical observation joins for calibration.

---

## 6. Remaining Architectural Concerns & Mitigation

1. **Zarr vs. Relational Storage Boundary**: 
   * *Concern*: Storing heavy multidimensional raster arrays directly in PostgreSQL would degrade performance.
   * *Mitigation*: PostgreSQL stores *metadata, catalogs, and pointers* (`model_runs.zarr_store_path`), while heavy numerical grids reside in cloud-optimized Zarr stores on object storage (MinIO/S3).
2. **Time-Series Growth**:
   * *Concern*: As verification observations and forecast product catalogs grow across multiple models (NOAA, ECMWF, Canada) and AI downscaling, table sizes will expand rapidly.
   * *Mitigation*: The schema is partitioned with future-ready table boundaries, and time-range partitioning on `verification_observations` and `model_runs` can be applied without architectural refactoring.
