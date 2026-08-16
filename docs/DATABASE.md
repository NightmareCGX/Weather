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

### Catalog write path

Ingestion populates the PostgreSQL catalog automatically. The production orchestration entrypoint is the `weather-ingest` CLI (`services/ingestion/src/ingestion/cli.py`, exposing `ingestion.core.pipeline.ingest_grib_file`). It downloads a GRIB2 file (NOAA NOMADS), parses it, writes the dataset to a Zarr store, and then upserts the center, model, model version, model run, grid, forecast variables, forecast products (one per variable × lead time), and ensemble members (one per `member` index) via `services/ingestion/src/ingestion/core/catalog.py`. The model run is recorded as `status='ready'` with its `zarr_store_path`, so the API serving tier (`_resolve_run`) discovers and serves it. No manual database seeding is required.

### Same-cycle re-ingestion semantics (PATCH)

A same-cycle per-lead/member ingestion is a **PATCH**, not a cycle replacement:

- Re-ingesting one lead/member replaces only that region's data and catalog rows; **all other valid leads/members are preserved**.
- Absence of a lead/member from the current invocation is **not** deletion intent.
- After a successful write, the catalog is **reconciled to the actual committed Zarr state** (the source of truth): rows whose lead/member is genuinely absent from the store (stale rows) are deleted; rows present in the store are preserved.

### Catalog↔Zarr consistency invariant

For a run to be `ready`, the catalog's committed contents must equal the actual **committed** Zarr state — not the preallocated coordinate axis (a cycle store is pre-allocated with the full expected lead/member axis NaN-filled; a lead/member is "committed" only when its region actually holds non-NaN forecast data):

```
deterministic:
    committed forecast_products lead set == actual committed Zarr lead set

ensemble:
    committed ensemble_member_products (member, lead) pairs == actual committed Zarr (member, lead) pairs
```

### READY meaning

`ready` requires **both**:

1. **Invocation completeness** — every expected lead (and, for ensembles, every expected `(member, lead)` Cartesian pair) has a committed catalog row (a **subset** check, PATCH-safe).
2. **Store↔catalog consistency** — the catalog equals the actual committed Zarr state (above).

`ready` does **not** mean "the model's full theoretical horizon is complete" (that full-horizon completeness is not persisted). A run can be internally consistent while not covering the entire model horizon, and that is acceptable.

If the store cannot be read reliably, the run is **not** `ready` (`partial`); a stale catalog can never be reported `ready`.

### Full-overwrite safety

The low-level Zarr helpers (`write_dataset`, `write_dataset_atomic`, `prepare_run_store` with `mode="w"`) can rebuild a store's coordinate axis and would silently shrink/replace a live run's store. They are guarded at the orchestration/pipeline boundary: a full overwrite of a store referenced by a `model_runs` row is rejected (`LiveStoreOverwriteError`) unless the caller explicitly enters a coordinated path. New/non-live store creation is unaffected.

### Existing stale rows

Existing `ready` rows whose catalog diverged from the store (created before this invariant) are **not retroactively repaired** by ingestion. They self-heal on the next successful re-ingestion of that run, which reconciles the catalog to the store and re-derives `ready` under the store↔catalog gate.

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
4. **`model_runs`**: `UNIQUE (model_version_id, cycle_time)`. The run primary key is **version-scoped** (`run_{version_id}_{cycle_time:%Y%m%d%H%M}_{model_id}`) so two versions of the same model at the same cycle produce distinct run ids (the schema's uniqueness allows both rows; the id must distinguish them).
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
   * *Mitigation*: The schema is partition-ready with future table boundaries, and time-range partitioning on `verification_observations` and `model_runs` can be applied without architectural refactoring.

---

## 7. Zarr region-write concurrency protocol

The ingestion pipeline uses a PostgreSQL-advisory-lock coordination protocol so
multiple processes can safely write disjoint physical Zarr regions of the same
forecast run concurrently.

### Lock order (deadlock-free)

```
admission turnstile
    -> store gate (SHARED for writers/readers, EXCLUSIVE for init/finalize)
    -> sorted unique physical-region locks
```

- The store gate uses **native** `pg_advisory_lock_shared`/`pg_advisory_lock`
  on one key (SHARED/EXCLUSIVE on the same key).
- Region locks are exclusive per physical conflict group. Under the current
  ensemble chunk layout (`member` full-extent), different members at the same
  lead share a physical chunk and serialize; different leads are disjoint and
  may proceed concurrently.

### Data commit order

```
UPDATING marker declared
    -> region data objects
    -> COMPLETE marker
    -> committed manifest
    -> catalog reconciliation/status commit
```

Stable per-region markers live at `<store>/__commit__/v1/regions/<region>.json`
with state+generation in the body; the committed manifest at
`<store>/__commit__/v1/manifest.json`.

### Protocol modes

- `legacy`: no markers (pre-upgrade behavior).
- `hybrid_marker_v1`: a touched region's marker overrides the legacy rule;
  marker-less regions use the legacy rule.
- `marker_v1`: strict — every committed region must have a COMPLETE marker.

### API reader gate

The API serving tier participates in the SHARED store gate when reading a
forecast Zarr store, so it never observes a store mid-re-ingest. Cache keys
include the committed-manifest generation, so a same-set same-cycle data
replacement makes old cache entries unreachable.

### Local-filesystem concurrent serving

Compliant API readers and ingestion writers coordinate via the PostgreSQL
gate even on local filesystems. Arbitrary non-gated external processes opening
local Zarr files directly are **unsupported** during an in-place chunk
replacement (local writes are not atomic); MinIO/S3 is the supported
concurrent-serving backend.

### Accepted split-brain limitation

If PostgreSQL connectivity is lost while an old ingestion writer can still
mutate MinIO/S3, the writer is **not fenced** (no fencing tokens). Under the
ensemble shared-chunk layout the RMW lost-update race can reappear in that
excluded split-brain failure model.
