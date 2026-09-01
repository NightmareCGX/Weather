# Deployment Architecture & Operations

The platform is designed for cloud-native containerized deployment across multi-region cloud environments (AWS / GCP / Azure).

---

## 1. Local Development (`Docker Compose`)
For local development, all auxiliary services are spun up via Docker Compose:
- **PostgreSQL 16 + PostGIS**: Spatial relational database.
- **Redis**: Caching and Celery message broker.
- **MinIO**: S3-compatible object storage for local Zarr and GRIB2 testing.

---

## 2. Production Cloud Architecture
- **Container Orchestration**: Kubernetes (EKS / GKE) running stateless microservices (`services/ingestion`, `services/processing`, `services/api`).
- **GPU Inference Node Pools**: Dedicated GPU node groups (NVIDIA T4 / A10G) running PyTorch inference workers for AI downscaling.
- **Object Storage**: AWS S3 with lifecycle policies for hot/cold Zarr raster storage.
- **Database**: Managed PostgreSQL (AWS RDS / GCP Cloud SQL) with PostGIS extension enabled and read replicas for serving.
- **CDN & Edge**: Cloudflare / AWS CloudFront caching API responses and map tiles at the edge.

---

## 3. CI/CD Pipelines (`GitHub Actions`)
- **Linting & Type Checking**: Ruff, Black, MyPy, ESLint.
- **Automated Testing**: Pytest for Python services, Jest/Playwright for Next.js frontend.
- **Container Builds**: Automated multi-architecture Docker image builds pushed to container registries on semantic release tags.
- **Infrastructure as Code**: Terraform for provisioning cloud networking, S3 buckets, RDS instances, and Kubernetes clusters.

---

## 4. Operational prerequisites (post-M14 remediation)

### Google Places API (New) — location autocomplete
- Enable the **Places API (New)** in the Google Cloud project.
- Create a **server API key** restricted by IP address and restricted to the
  Places API service. The key is read by the FastAPI service from
  `GOOGLE_PLACES_API_KEY` and **never reaches the browser** (the frontend
  proxies `/v1/*` to FastAPI via Next.js rewrites).
- `SEARCH_PROVIDER` selects the backend (`google` default, or `mapbox` with
  `MAPBOX_TOKEN`).

### DEM (terrain elevation)
- The API reads elevation from `DEM_DATA_PATH` — a global xarray-readable DEM
  store (Zarr or NetCDF) with `latitude`/`longitude` coordinates and an
  `elevation` data variable in meters. The supported source is **local
  Copernicus 30m DEM data**: the operator downloads the Copernicus DEM
  (GLO-30 / Copernicus 30m) and prepares it into a Zarr/NetCDF store on the
  API's object storage or a mounted volume, then points `DEM_DATA_PATH` at it.
  No Sentinel Hub credentials or online DEM retrieval are used.
- Expected layout: a global grid with `latitude` (descending or ascending) and
  `longitude` coordinates and an `elevation` data variable in meters. The
  provider interpolates bilinearly; no-data (NaN/ocean) cells yield `null`
  (rendered `unavailable`), never a fabricated value.
- `ELEVATION_PROVIDER` selects the backend (`dem` default, `google`, or
  `none`). The elevation provider and its rounded-coordinate cache are created
  **once per process** and reused across requests (the DEM store is not
  re-opened per request). No DEM configured → elevation renders `unavailable`.

### Environment variables
New optional env vars (defaults in `services/api/src/api/core/config.py`):
`SEARCH_PROVIDER`, `GOOGLE_PLACES_API_KEY`, `GOOGLE_PLACES_API_BASE`,
`GOOGLE_PLACES_REGION`, `GOOGLE_PLACES_TIMEOUT`, `MAPBOX_TOKEN`,
`ELEVATION_PROVIDER`, `DEM_DATA_PATH`, `ELEVATION_CACHE_MAX`,
`ELEVATION_CACHE_DISABLED`, `ENSEMBLE_MIN_COVERAGE_RATIO`. See `.env.example`.

The ingestion worker reads `NOMADS_BASE_URL`, `NOAA_USER_AGENT`,
`REQUEST_TIMEOUT_SECONDS`, `DOWNLOAD_RETRIES`, `RETRY_BACKOFF_SECONDS` (see
`services/ingestion/src/ingestion/core/config.py`). GEFS ingestion fetches the
30 perturbation members (`gep01`..`gep30`) from the NOMADS `pgrb2sp25` layout,
preserving each member's real identity; the run becomes `READY` only when every
requested lead and member is committed. Source `.grib2`/`.idx` files are deleted
after durable per-file commit.

---

