# Global Probabilistic Weather Platform

A production-grade, global probabilistic weather forecasting platform combining operational numerical weather prediction (NWP) models (NOAA, ECMWF, ECCC), a custom neural AI downscaling pipeline (25km to 3km/1km), Multi-Model Ensemble (MME) calibration, and high-performance point and spatial APIs.

> **Implementation status**: Milestones 1–10 are complete and approved (see [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)). The live surface today is: NOAA GFS/GEFS ingestion (GRIB2 → Zarr → PostgreSQL catalog), the FastAPI `/v1` catalog / points / search / probabilities / maps / ensembles surface, PostGIS + Redis caching, and the domain ensemble/geo math engine. The AI downscaling, MME calibration, ECMWF/Canada providers, and the frontend map viewer are future milestones (11–18) and are **not yet implemented**.

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
- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — Master milestone roadmap (Milestones 1–18).
- [API.md](docs/API.md) — REST API specifications, endpoints, request/response models, and error handling.
- [DATABASE.md](docs/DATABASE.md) — PostgreSQL + PostGIS schema design, spatial indexing, and caching strategy.
- [TESTING.md](docs/TESTING.md) — Testing strategy, layers, and fixtures.
- [MODELS.md](docs/MODELS.md) — Operational weather model integrations (NOAA GFS/GEFS, ECMWF IFS/AIFS, ECCC GDPS/GEPS).
- [AI_PLAN.md](docs/AI_PLAN.md) — Custom AI downscaling architecture, training data pipeline, and inference strategy.
- [DEPLOYMENT.md](docs/DEPLOYMENT.md) — Cloud-native deployment architecture, Docker Compose, Kubernetes, and CI/CD pipelines.
- [CONTRIBUTING.md](docs/CONTRIBUTING.md) — Development guidelines, code style, testing standards, and pull request workflows.

---

## Repository Structure

```text
weather-platform/
├── CLAUDE.md                     # Project instructions for AI-assisted development
├── ENGINEERING_CONTRACT.md       # Engineering standards and development rules
├── IMPLEMENTATION_PLAN.md        # Master milestone roadmap (Milestones 1–18)
├── docs/                         # Authoritative project documentation
│   ├── ARCHITECTURE.md
│   ├── API.md
│   ├── DATABASE.md
│   ├── TESTING.md
│   ├── MODELS.md
│   ├── AI_PLAN.md
│   ├── DEPLOYMENT.md
│   └── CONTRIBUTING.md
├── packages/                     # Shared Python packages
│   ├── domain/                   # Core business logic, ensemble math, spatial interpolation
│   ├── contracts/                # Shared Pydantic contracts (placeholder)
│   └── config/                   # Centralized configuration (placeholder)
├── services/
│   ├── api/                      # FastAPI core service (v1 catalog/points/search/probabilities/maps/ensembles, PostGIS, Redis cache)
│   ├── ingestion/                # GRIB2 downloaders, decoders, Zarr storage, PostgreSQL catalog writer (weather-ingest CLI)
│   └── frontend/                 # Next.js / React client application (scaffold)
├── docker-compose.yml            # Local PostgreSQL + PostGIS, Redis, MinIO
├── .env.example                  # Environment variable template
├── pyproject.toml                # Python workspace configuration
└── README.md
```

---

## License

Proprietary & Confidential. All rights reserved.
