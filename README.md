# Global Probabilistic Weather Platform

A production-grade, global probabilistic weather forecasting platform combining operational numerical weather prediction (NWP) models (NOAA, ECMWF, ECCC), a custom neural AI downscaling pipeline (25km to 3km/1km), Multi-Model Ensemble (MME) calibration, and high-performance point and spatial APIs.

---

## Architecture Overview

```
+-----------------------------------------------------------------------------------------+
|                                1. DATA INGESTION ENGINE                                 |
|  [NOAA NOMADS / AWS S3]    [ECMWF MARS / CDS]    [ECCC Datamart]                        |
+-----------------------------------------------------------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------------------------+
|                              2. STANDARDIZED ZARR STORE                                 |
|                 (Cloud-Optimized Zarr Datasets on Object Storage / S3)                  |
+-----------------------------------------------------------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------------------------+
|                          3. CUSTOM AI DOWNSCALING PIPELINE                              |
|           (Coarse Global Grids ~25km + Topography/DEM ──► High-Res 3km / 1km)           |
+-----------------------------------------------------------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------------------------+
|                         4. MULTI-MODEL ENSEMBLE (MME) CALIBRATION                       |
|         (Bayesian Model Averaging, Bias Correction, Probabilistic Moments)              |
+-----------------------------------------------------------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------------------------+
|                               5. API & SERVING TIER                                     |
|                        (FastAPI + PostGIS + Redis Caching)                              |
+-----------------------------------------------------------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------------------------+
|                               6. FRONTEND TIER                                          |
|                     (Next.js / React + MapLibre GL JS + Charts)                         |
+-----------------------------------------------------------------------------------------+
```

---

## Documentation Index

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — Complete system architecture, component breakdown, and data flow.
- [ROADMAP.md](docs/ROADMAP.md) — Phased implementation roadmap from Phase 1 to Phase 6.
- [API.md](docs/API.md) — REST API specifications, endpoints, request/response models, and error handling.
- [DATABASE.md](docs/DATABASE.md) — PostgreSQL + PostGIS schema design, spatial indexing, and caching strategy.
- [MODELS.md](docs/MODELS.md) — Operational weather model integrations (NOAA GFS/GEFS, ECMWF IFS/AIFS, ECCC GDPS/GEPS).
- [AI_PLAN.md](docs/AI_PLAN.md) — Custom AI downscaling architecture, training data pipeline, and inference strategy.
- [DEPLOYMENT.md](docs/DEPLOYMENT.md) — Cloud-native deployment architecture, Docker Compose, Kubernetes, and CI/CD pipelines.
- [CONTRIBUTING.md](docs/CONTRIBUTING.md) — Development guidelines, code style, testing standards, and pull request workflows.

---

## Repository Structure

```text
weather-platform/
├── .github/
│   └── workflows/                # CI/CD pipelines (Lint, Test, Docker Build, Deploy)
├── docker/                       # Dockerfiles & compose setups for dev/prod
├── docs/                         # Comprehensive project documentation
│   ├── ARCHITECTURE.md
│   ├── ROADMAP.md
│   ├── API.md
│   ├── DATABASE.md
│   ├── MODELS.md
│   ├── AI_PLAN.md
│   ├── DEPLOYMENT.md
│   └── CONTRIBUTING.md
├── infra/                        # Terraform / Kubernetes manifests (Helm charts)
├── packages/                     # Shared packages / TypeScript types / UI components
├── services/
│   ├── ingestion/                # GRIB2 downloaders, workers, schedulers
│   ├── processing/               # AI downscaling, MME calibration, spatial interpolation
│   ├── api/                      # FastAPI core service
│   └── frontend/                 # Next.js / React client application
├── pyproject.toml                # Python workspace configuration
└── README.md
```

---

## License

Proprietary & Confidential. All rights reserved.
