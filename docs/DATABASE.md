# Database Architecture Specification: Global Probabilistic Weather Platform

---

## 1. Scientific & Engineering Rationale for `lead_time_hours`

In numerical weather prediction (NWP) systems, forecasts are indexed relative to **`lead_time_hours`** (offset hours from the model run's initialization `cycle_time`) rather than absolute `valid_time` timestamps:

1. **Scientific Alignment:** Global NWP models initialize at fixed synoptic cycle hours (00Z, 06Z, 12Z, 18Z) and publish forecast steps (e.g. `+00`, `+03`, `+06`, ..., `+240`). Storing `lead_time_hours` directly mirrors the native GRIB2 message headers and Zarr coordinate arrays (`lead_time_hours`).
2. **Re-run & Backfill Idempotency:** If a forecast cycle is re-ingested or corrected, relative `lead_time_hours` coordinates remain invariant, ensuring deterministic upsert behavior.
3. **Deterministic Derivation:** Valid time is derived on the fly:
   $$\text{valid\_time} = \text{cycle\_time} + \text{lead\_time\_hours}$$

---

## 2. Relational Schema & Migration History

Schema migrations are managed by Alembic (`services/api/alembic/versions/`):

### Migration 001: Initial Schema (`001_initial_schema.py`)
* `forecast_centers`: Numerical weather prediction modeling centers (e.g. `noaa`).
* `models`: Weather models (`gfs`, `gefs`) with resolution and ensemble flags.
* `model_versions`: Version strings tied to models (`v16`).
* `model_runs`: Model forecast cycles (`cycle_time`, `status`, `zarr_store_path`).
* `ensemble_members`: Ensemble member catalog (`member_index`, `member_name`).
* `forecast_variables`: Variable dictionary (`temperature_2m`, `precipitation_rate`, `precipitation_amount_3h`, `crain`, `csnow`).
* `forecast_grids`: Spatial grid specifications (e.g. `global_0p25`, `global_0p50`).
* `forecast_products`: Committed lead times per variable and grid.
* `stations`, `cities`, `ski_resorts`: Spatial reference tables with PostGIS `GEOMETRY(Point, 4326)` and `GIST` indexes.
* `point_query_fallback_audit`: Audit ledger and Redis fallback table with BTREE index on `expires_at`.

### Migration 002: Ensemble Member Products (`002_ensemble_member_products.py`)
* `ensemble_member_products`: Tracks individual committed `(member_index, lead_time_hours)` pairs per run to support progressive Cartesian product verification for GEFS.

### Migration 003: Cycle Lifecycle & Retirement (`003_cycle_lifecycle.py`)
* `forecast_cycle_lifecycle`: Tracks durable cycle supersession and retirement timestamps (`retired_at`, `retired_by_cycle_time`, `deleted_at`). Survives physical store deletion as an audit tombstone.

### Migration 004: Deletion Fencing (`004_deletion_fence.py`)
* `forecast_cycle_lifecycle.deletion_started_at`: Crash-safe deletion fence column and index (`idx_cycle_lifecycle_claimed`). Fences off retired cycles so ingestion writers cannot resurrect a cycle during or after GC deletion.

---

## 3. Table Ownership & Mutability Matrix

| Table | Writer Service | Reader Service | Primary Key / Constraints | Mutability |
| :--- | :--- | :--- | :--- | :--- |
| `forecast_centers` | Ingestion catalog init | API catalog router | `id` PK, `UNIQUE (center_id)` | Static |
| `models` | Ingestion catalog init | API catalog router | `id` PK, `UNIQUE (model_id)` | Static |
| `model_versions` | Ingestion catalog init | API catalog router | `id` PK, `UNIQUE (model_id, version_string)` | Static |
| `model_runs` | Ingestion wave runner & GC | API serving, Ingestion GC | `id` PK, `UNIQUE (model_version_id, cycle_time)` | Mutable (`status`, `zarr_store_path`) |
| `ensemble_members` | Ingestion catalog init | API ensemble router | `id` PK, `UNIQUE (run_id, member_index)` | Static per run |
| `forecast_products` | Ingestion wave runner | API availability service | `id` PK, `UNIQUE (run_id, variable_id, grid_id, product_type, lead_time_hours)` | Append-only per wave |
| `ensemble_member_products`| Ingestion wave runner | Ingestion wave runner | `id` PK, `UNIQUE (run_id, member_index, lead_time_hours)` | Append-only per wave |
| `forecast_cycle_lifecycle`| Ingestion GC reconciler | Ingestion GC, API lifecycle | `cycle_time` PK | Mutable timestamps |
| `forecast_variables` | Ingestion catalog init | API catalog router | `id` PK, `UNIQUE (variable_code)` | Static |
| `forecast_grids` | Ingestion catalog init | API catalog router | `id` PK, `UNIQUE (grid_code)` | Static |
| `stations`, `cities`, `ski_resorts` | Seed scripts | API search & places | `id` PK, PostGIS `GIST` on `geom` | Static reference |
| `point_query_fallback_audit` | API response cache | API response cache | `cache_key` PK, `BTREE (expires_at)` | Ephemeral |

---

## 4. Lifecycle & Availability Semantics

### 4.1 Run Status Transitions
```text
[discovered] ──► [processing] ──► [partial] ──► [ready]
                                     │             │
                                     ▼             ▼
                                  [failed]     [retired] ──► [deleted]
```

* `processing`: Run record created in PostgreSQL; initial wave download/write in progress.
* `partial`: One or more lead waves have committed and published to `forecast_products`. Serving tier can serve available leads.
* `ready`: **The complete canonical horizon (`domain.horizon.canonical_lead_time_hours`, e.g. 0–240h at 3h cadence for GFS, and 30 members × full leads for GEFS) is fully committed.**
* `failed`: Wave unrecoverably failed or aborted.
* `retired`: Cycle has been superseded by a newer model cycle (tracked in `forecast_cycle_lifecycle.retired_at`).
* `deleted`: Storage files purged by GC engine (`forecast_cycle_lifecycle.deleted_at` set).

### 4.2 Same-Cycle Re-Ingestion (PATCH Semantics)
* Ingestion of a lead or member wave acts as a **PATCH** on the cycle store.
* Writing a lead updates the corresponding region in the `sharded_v1` Zarr store and upserts `forecast_products` / `ensemble_member_products`.
* Other leads and members in the store are preserved.
* After writing, the catalog reconciles with committed Zarr markers.

---

## 5. Entity Relationship Diagram

```text
[forecast_centers] 1 ──< [models] 1 ──< [model_versions] 1 ──< [model_runs]
                                                                  │
                                           ┌──────────────────────┴──────────────────────┐
                                           │ (1:N)                                       │ (1:N)
                                           ▼                                             ▼
                                  [ensemble_members]                            [forecast_products]
                                           │                                             │
                                           │ (1:N)                                       │ (N:1)
                                           ▼                                             ▼
                             [ensemble_member_products]                         [forecast_variables]
                                                                                         │
                                                                                         │ (N:1)
                                                                                         ▼
                                                                                [forecast_grids]

[forecast_cycle_lifecycle]
  ├── cycle_time (PK, TIMESTAMP WITH TIME ZONE)
  ├── retired_at (TIMESTAMP WITH TIME ZONE)
  ├── retired_by_cycle_time (TIMESTAMP WITH TIME ZONE)
  ├── deletion_started_at (TIMESTAMP WITH TIME ZONE)
  ├── deleted_at (TIMESTAMP WITH TIME ZONE)
  └── created_at / updated_at

[cities], [stations], [ski_resorts] (PostGIS geometry tables)
```

---

## 6. PostgreSQL Advisory Lock Coordination

The database coordinates reader/writer concurrency using session-level 64-bit advisory locks (`domain.locks`, `ingestion.core.locks`, `api.core.reader_gate`):

```text
Lock Hierarchy:
1. Admission Turnstile (`0x2000000000000000 | hash`)  — Brief exclusive hold by writers to drain readers.
2. Store Gate (`0x0000000000000000 | hash`)            — SHARED for readers & region writers; EXCLUSIVE for finalizer & GC.
3. Region Conflict (`0x1000000000000000 | hash`)       — EXCLUSIVE per (lead, member) region during write.
4. Leader Election (`0x3000...` / `0x4000...`)         — Realtime Scheduler & GC Reconciler singletons.
```

* **Acquisition Semantics:** Executed using `SET LOCAL lock_timeout = :ms` inside a transaction block, entering the PostgreSQL lock queue.
* **Session Persistence:** Locks remain attached to the physical connection across transaction commits.
* **Safe Invalidation:** On any query or unlock error, the connection is invalidated (`conn.invalidate()`), ensuring PostgreSQL immediately drops all associated session locks.
