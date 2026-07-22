# System Architecture Specification

## 1. Executive Summary
The Global Probabilistic Weather Platform is designed as a cloud-native, high-throughput spatiotemporal processing and serving engine. It ingests raw numerical weather prediction (NWP) datasets from global meteorological centers, processes them through a custom neural AI downscaling pipeline, performs multi-model ensemble (MME) calibration, and serves low-latency probabilistic forecasts through high-performance APIs and interactive map interfaces.

---

## 2. Core Architectural Principles
- **High-Resolution First**: Raw global forecasts (~25km) are translated into high-resolution local fields (3km / 1km) via neural downscaling prior to spatial serving and ensemble calibration.
- **Cloud-Optimized Storage**: Multidimensional raster data is stored in chunked, compressed Zarr formats on object storage (S3/MinIO), enabling sub-second slicing across spatial and temporal dimensions without memory exhaustion.
- **Asynchronous Ingestion & Processing**: Data downloads, GRIB2 decoding, and AI inference run asynchronously via Celery worker pools orchestrated around weather model release cycles (00Z, 06Z, 12Z, 18Z).
- **Decoupled API & Serving Tier**: FastAPI provides asynchronous REST and spatial query capabilities backed by PostGIS spatial indexing and Redis hot-cache layers.

---

## 3. Detailed Component Architecture

### 3.1 Data Ingestion Engine (`services/ingestion`)
- **Connectors**: Pluggable protocol adapters for NOAA NOMADS/S3, ECMWF CDS/MARS, and ECCC Datamart.
- **Schedulers**: Cron/Celery beat tasks that monitor upstream release schedules and trigger ingestion workflows.
- **Decoders**: Leverages `cfgrib` and `xarray` to parse GRIB2 binary streams into standardized internal NetCDF/Zarr arrays.

### 3.2 Storage Tier (`Zarr` + `PostgreSQL/PostGIS` + `Redis`)
- **Raster Store (Zarr)**: Chunked by `(time, lead_time, vertical_level, latitude, longitude)` for optimized spatial bounding-box queries and temporal slicing.
- **Relational Store (PostgreSQL + PostGIS)**: Stores station metadata, administrative boundaries, city points, ski resorts, and user configurations. Uses PostGIS spatial types (`GEOMETRY(Point, 4326)`) and R-tree spatial indexes (`GIST`).
- **Cache Layer (Redis)**: Caches point-query responses and interpolated time-series with TTLs aligned to model update intervals.

### 3.3 Custom AI Downscaling Pipeline (`services/processing/downscale`)
- **Inputs**: Coarse global model prognostic fields (temperature, winds, humidity, geopotential height) + static high-resolution terrain tensors (DEM, slope, aspect, land-use, distance-to-coast).
- **Inference Engine**: PyTorch-based neural downscaling models (conditional UNet / Super-Resolution CNN) operating on regional tiles.
- **Output**: 3km / 1km high-resolution calibrated atmospheric fields.

### 3.4 Multi-Model Ensemble & Calibration Engine (`services/processing/ensemble`)
- **Bias Correction**: Quantile mapping and empirical calibration against station observation histories.
- **MME Blending**: Bayesian Model Averaging (BMA) weighting NOAA, ECMWF, and Canadian ensemble members dynamically based on recent verification skill scores.
- **Probabilistic Calculators**: Vectorized NumPy/Xarray operators computing ensemble mean, spread, standard deviation, interquartile range, P10/P50/P90 percentiles, and exceedance probabilities.

### 3.5 API Serving Tier (`services/api`)
- **FastAPI Framework**: Asynchronous endpoints supporting high concurrency.
- **Spatial Resolver**: Maps any arbitrary latitude/longitude, city, ski resort, or street address to grid indices via PostGIS spatial lookups.

### 3.6 Frontend Tier (`services/frontend`)
- **Next.js / React**: Modern SSR/CSR application framework.
- **MapLibre GL JS**: High-performance vector and raster tile rendering for meteorological overlays (wind streamlines, temperature contours, precipitation radar).
- **Analytics**: Interactive meteograms, ensemble spaghetti plots, probability distribution curves, and Skew-T thermodynamic diagrams.
