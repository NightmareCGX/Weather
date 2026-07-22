# Global Probabilistic Weather Platform: API Design Specification

**Principal API Architect Specification**  
*Version: 1.1.0 (Production-Ready, Final Consistency Pass)*

---

## 1. Overall API Philosophy

Our API design is modeled after industry standards set by Stripe, GitHub, and OpenAI. We treat our API as a **first-class product**.

### Core Tenets
1. **Resource-Oriented & Domain-Driven**: Endpoints represent domain entities (`/v1/points`, `/v1/models`, `/v1/search`) rather than mirroring internal database tables or internal microservice topologies.
2. **Predictable & Consistent**: Every endpoint follows uniform patterns for pagination, filtering, sorting, error handling, rate limiting, and response envelopes.
3. **Backward Compatible**: Breaking changes are strictly prohibited on active version paths (`/v1/`). Additions are non-breaking (adding new optional query parameters or response fields).
4. **Developer Experience (DX)**: Clear error messages, request IDs for tracing, idempotency support, and comprehensive OpenAPI 3.1 specifications.

---

## 2. Cross-Cutting API Standards

### 2.1 API Versioning
- **URL Path Versioning**: All production endpoints are prefixed with `/v1/` (e.g., `https://api.weatherplatform.com/v1/points`).
- **Breaking Changes**: Managed by introducing a new major version prefix (`/v2/`) with a minimum 12-month deprecation notice for legacy versions.

### 2.2 Authentication & Authorization
- **Bearer Token Auth**: All requests must include an API key or OAuth2 bearer token in the HTTP header:
  `Authorization: Bearer wp_live_xxxxxxxxxxxxxxxxxxxx`
- **Scopes & Roles**: API keys are bound to commercial subscription tiers, enforcing rate limits and access privileges (e.g., `points:read`, `maps:read`, `ensemble:read`, `admin:write`).

### 2.3 Response Envelope Policy
- **Universal Envelope**: **All** successful API responses (including single objects, lists, and metadata) strictly wrap their payload data inside the standard envelope:
  ```json
  {
    "object": "list" | "point_forecast" | "spatial_layer" | "verification" | "model" | "center" | "run" | "variable" | "grid",
    "data": { ... },
    "has_more": false,
    "next_cursor": null
  }
  ```
  *(Lists return an array under `data`, while single resource retrievals return an object under `data`).*

### 2.4 Error Model (RFC 7807 compliant)
Errors return standard HTTP status codes along with a structured machine-readable error payload:
```json
{
  "error": {
    "code": "invalid_request_error",
    "type": "validation_error",
    "message": "The latitude parameter must be between -90 and 90 degrees.",
    "param": "lat",
    "request_id": "req_9f8d7c6b5a4321"
  }
}
```

### 2.5 Pagination, Filtering, Sorting
- **Cursor-Based Pagination**: Used for list endpoints where data changes frequently. Parameters: `limit` (default 20, max 100), `starting_after`, `ending_before`.
- **Filtering**: Query parameters use dot-notation for nested attributes (e.g., `filter[model]=gfs&filter[variable]=temperature_2m`).
- **Sorting**: Controlled via `sort` parameter (e.g., `sort=-valid_time` for descending order).

### 2.6 Datetime, Units, & Coordinates
- **Datetimes & Forecast Concepts**: Absolute timestamps use ISO 8601 UTC strings (`2026-07-21T12:00:00Z`). However, the core numerical weather prediction (NWP) concept for forecast products is **`lead_time_hours`** (offset hours from cycle time), ensuring ingestion idempotency and multi-model alignment. `valid_time` may be returned as a convenience property derived from `cycle_time + lead_time_hours`.
- **Units**: International System of Units (SI) by default (`°C`, `mm`, `km/h`, `hPa`), with query param support for imperial overrides (`?units=imperial` for `°F`, `in`, `mph`).
- **Coordinate System**: WGS 84 (`EPSG:4326`) for latitude (-90 to 90) and longitude (-180 to 180).

### 2.7 Caching, ETags, & Request IDs
- **Caching**: Endpoints return `Cache-Control` headers reflecting model update intervals (e.g., `public, max-age=1800` for 30 minutes).
- **ETags**: Responses include `ETag` headers for conditional GET requests (`If-None-Match`).
- **Request IDs**: Every response includes `X-Request-Id` for debugging and customer support tracing.
- **Idempotency**: State-changing endpoints support `Idempotency-Key` headers.

---

## 3. Resource Hierarchy & Domain Groups

1. **Catalog**: `/v1/centers`, `/v1/models`, `/v1/runs`, `/v1/variables`, `/v1/grids`
2. **Point Forecast**: `/v1/points`
3. **Probability**: `/v1/probabilities`
4. **Maps**: `/v1/maps`
5. **Ensemble**: `/v1/ensembles`
6. **Search**: `/v1/search`
7. **Verification**: `/v1/verifications`
8. **Administration**: `/v1/health`

---

## 4. Comprehensive Endpoint Specifications

---

### DOMAIN 1: CATALOG
*Purpose: Model discovery, forecast run status, variables, grid definitions, and center metadata.*

#### 1.1 List Forecast Centers
- **HTTP Method**: `GET`
- **Endpoint**: `/v1/centers`
- **Purpose**: Retrieve all supported meteorological forecast centers.
- **Required Parameters**: None.
- **Optional Parameters**: `limit`, `starting_after`.
- **Example Request**: `GET /v1/centers`
- **Example Response**:
  ```json
  {
    "object": "list",
    "data": [
      {
        "id": "noaa",
        "object": "center",
        "name": "National Oceanic and Atmospheric Administration",
        "country": "USA"
      }
    ],
    "has_more": false,
    "next_cursor": null
  }
  ```
- **HTTP Status Codes**: `200 OK`, `401 Unauthorized`.
- **Cache Policy**: `public, max-age=86400` (24 hours).

#### 1.2 List Models
- **HTTP Method**: `GET`
- **Endpoint**: `/v1/models`
- **Purpose**: Retrieve all supported operational and AI weather models.
- **Required Parameters**: None.
- **Optional Parameters**: `center_id` (string), `is_ensemble` (boolean).
- **Example Request**: `GET /v1/models?center_id=noaa`
- **Example Response**:
  ```json
  {
    "object": "list",
    "data": [
      {
        "id": "gfs",
        "object": "model",
        "name": "Global Forecast System",
        "center_id": "noaa",
        "is_ensemble": false,
        "resolution_km": 25.0
      },
      {
        "id": "gefs",
        "object": "model",
        "name": "Global Ensemble Forecast System",
        "center_id": "noaa",
        "is_ensemble": true,
        "resolution_km": 25.0
      }
    ],
    "has_more": false,
    "next_cursor": null
  }
  ```
- **HTTP Status Codes**: `200 OK`, `401 Unauthorized`, `429 Too Many Requests`.
- **Cache Policy**: `public, max-age=86400` (24 hours).

#### 1.3 List Model Runs
- **HTTP Method**: `GET`
- **Endpoint**: `/v1/runs`
- **Purpose**: Retrieve ingested model execution cycles and their Zarr storage statuses.
- **Required Parameters**: None.
- **Optional Parameters**: `model_id` (string), `status` (string: `ready`, `processing`), `limit`.
- **Example Request**: `GET /v1/runs?model_id=gfs&status=ready`
- **Example Response**:
  ```json
  {
    "object": "list",
    "data": [
      {
        "id": "run_2026072100_gfs",
        "object": "run",
        "model_id": "gfs",
        "cycle_time": "2026-07-21T00:00:00Z",
        "status": "ready"
      }
    ],
    "has_more": false,
    "next_cursor": null
  }
  ```
- **HTTP Status Codes**: `200 OK`, `401 Unauthorized`.
- **Cache Policy**: `public, max-age=300` (5 minutes).

#### 1.4 List Forecast Variables
- **HTTP Method**: `GET`
- **Endpoint**: `/v1/variables`
- **Purpose**: Retrieve standardized physical meteorological variables available in the platform.
- **Required Parameters**: None.
- **Optional Parameters**: `limit`.
- **Example Request**: `GET /v1/variables`
- **Example Response**:
  ```json
  {
    "object": "list",
    "data": [
      {
        "id": "temperature_2m",
        "object": "variable",
        "name": "2-Meter Temperature",
        "unit": "°C"
      },
      {
        "id": "precipitation_rate",
        "object": "variable",
        "name": "Precipitation Rate",
        "unit": "mm/h"
      }
    ],
    "has_more": false,
    "next_cursor": null
  }
  ```
- **HTTP Status Codes**: `200 OK`, `401 Unauthorized`.
- **Cache Policy**: `public, max-age=86400` (24 hours).

#### 1.5 List Forecast Grids
- **HTTP Method**: `GET`
- **Endpoint**: `/v1/grids`
- **Purpose**: Retrieve supported spatial grid definitions (coarse global vs. high-res AI downscaled).
- **Required Parameters**: None.
- **Optional Parameters**: `limit`.
- **Example Request**: `GET /v1/grids`
- **Example Response**:
  ```json
  {
    "object": "list",
    "data": [
      {
        "id": "global_025deg",
        "object": "grid",
        "name": "Global 0.25 Degree Grid",
        "resolution_km": 25.0
      },
      {
        "id": "downscaled_3km",
        "object": "grid",
        "name": "AI Downscaled Local Grid",
        "resolution_km": 3.0
      }
    ],
    "has_more": false,
    "next_cursor": null
  }
  ```
- **HTTP Status Codes**: `200 OK`, `401 Unauthorized`.
- **Cache Policy**: `public, max-age=86400` (24 hours).

---

### DOMAIN 2: POINT FORECAST
*Purpose: Retrieve forecasts for any coordinate, city, ski resort, or address.*

#### 2.1 Get Point Forecast
- **HTTP Method**: `GET`
- **Endpoint**: `/v1/points`
- **Purpose**: Return hourly aggregated forecasts indexed by `lead_time_hours` for a specific geographic location.
- **Required Parameters**: Exactly one spatial specifier (`lat` & `lon`, `city_id`, `resort_id`, or `address`).
- **Optional Parameters**: 
  - `models` (comma-separated string; **MVP Default**: `gfs,gefs`)
  - `variables` (comma-separated string, e.g., `temperature_2m,precipitation_rate`)
  - `units` (`metric` or `imperial`, default `metric`)
  - `start_lead_time_hours` / `end_lead_time_hours` (integer offsets)
- **Example Request**: `GET /v1/points?lat=39.1911&lon=-106.8175&models=gfs`
- **Example Response**:
  ```json
  {
    "object": "point_forecast",
    "data": {
      "location": {
        "latitude": 39.1911,
        "longitude": -106.8175,
        "elevation_m": 3417,
        "resolved_via": "coordinates"
      },
      "generated_at": "2026-07-21T06:00:00Z",
      "model": "gfs",
      "forecasts": [
        {
          "lead_time_hours": 6,
          "valid_time": "2026-07-21T06:00:00Z",
          "temperature_2m": 15.0,
          "precipitation_mm": 0.0,
          "wind_speed_kmh": 14.2
        }
      ]
    },
    "has_more": false,
    "next_cursor": null
  }
  ```
- **HTTP Status Codes**: `200 OK`, `400 Bad Request`, `404 Not Found`, `429 Too Many Requests`.
- **Cache Policy**: `public, max-age=1800` (30 minutes).

---

### DOMAIN 3: PROBABILITY
*Purpose: Query probabilistic exceedance odds, thresholds, and confidence intervals.*

#### 3.1 Get Exceedance Probability
- **HTTP Method**: `GET`
- **Endpoint**: `/v1/probabilities`
- **Purpose**: Calculate the statistical probability that a variable will exceed a given threshold based on ensemble spread.
- **Required Parameters**: `lat`, `lon`, `variable`, `threshold`, `operator` (`gt`, `lt`, `between`), `lead_time_hours`.
- **Optional Parameters**: `model` (default `gefs`).
- **Example Request**: `GET /v1/probabilities?lat=45.0&lon=-122.0&variable=precipitation_rate&threshold=10.0&operator=gt&lead_time_hours=24`
- **Example Response**:
  ```json
  {
    "object": "probability_forecast",
    "data": {
      "location": { "latitude": 45.0, "longitude": -122.0 },
      "variable": "precipitation_rate",
      "threshold": 10.0,
      "operator": "gt",
      "lead_time_hours": 24,
      "probability": 0.42,
      "confidence_interval_95": [0.38, 0.46]
    },
    "has_more": false,
    "next_cursor": null
  }
  ```
- **HTTP Status Codes**: `200 OK`, `400 Bad Request`, `422 Unprocessable Entity`.
- **Cache Policy**: `public, max-age=3600`.

---

### DOMAIN 4: MAPS
*Purpose: Raster and vector tile endpoints for MapLibre GL UI rendering.*

#### 4.1 Get Map Tile Metadata
- **HTTP Method**: `GET`
- **Endpoint**: `/v1/maps`
- **Purpose**: Return tile URL templates and legend configuration for weather map visualization indexed by `lead_time_hours`.
- **Required Parameters**: `model`, `variable`, `level`, `lead_time_hours`.
- **Example Request**: `GET /v1/maps?model=gfs&variable=temperature_2m&level=surface&lead_time_hours=12`
- **Example Response**:
  ```json
  {
    "object": "spatial_layer",
    "data": {
      "tile_url_template": "https://tiles.weatherplatform.com/v1/gfs/temperature_2m/surface/{z}/{x}/{y}.png?lead_time_hours=12",
      "min_zoom": 0,
      "max_zoom": 9,
      "lead_time_hours": 12,
      "legend": {
        "unit": "°C",
        "stops": [[-40, "#0000ff"], [0, "#00ff00"], [40, "#ff0000"]]
      }
    },
    "has_more": false,
    "next_cursor": null
  }
  ```
- **HTTP Status Codes**: `200 OK`, `400 Bad Request`.
- **Cache Policy**: `public, max-age=3600`.

---

### DOMAIN 5: ENSEMBLE
*Purpose: Member inspection, statistical spread, mean, median, and percentiles.*

#### 5.1 Get Ensemble Statistics & Spread
- **HTTP Method**: `GET`
- **Endpoint**: `/v1/ensembles`
- **Purpose**: Return statistical dispersion (mean, spread, P10, P25, P50, P75, P90) across ensemble perturbation members for a given `lead_time_hours`.
- **Required Parameters**: `lat`, `lon`, `variable`, `model` (default `gefs`), `lead_time_hours`.
- **Example Request**: `GET /v1/ensembles?lat=39.19&lon=-106.81&variable=temperature_2m&model=gefs&lead_time_hours=18`
- **Example Response**:
  ```json
  {
    "object": "ensemble_statistics",
    "data": {
      "model": "gefs",
      "lead_time_hours": 18,
      "member_count": 31,
      "statistics": {
        "mean": 15.1,
        "median": 15.0,
        "spread": 2.7,
        "p10": 12.4,
        "p25": 13.8,
        "p50": 15.0,
        "p75": 16.5,
        "p90": 17.8
      }
    },
    "has_more": false,
    "next_cursor": null
  }
  ```
- **HTTP Status Codes**: `200 OK`, `400 Bad Request`.
- **Cache Policy**: `public, max-age=1800`.

---

### DOMAIN 6: SEARCH
*Purpose: Geocoding cities, stations, ski resorts, and coordinates.*

#### 6.1 Search Locations
- **HTTP Method**: `GET`
- **Endpoint**: `/v1/search`
- **Purpose**: Full-text search across cities, ski resorts, and observation stations.
- **Required Parameters**: `q` (search query string).
- **Optional Parameters**: `type` (`city`, `resort`, `station`, `all`), `limit`.
- **Example Request**: `GET /v1/search?q=Aspen&type=resort`
- **Example Response**:
  ```json
  {
    "object": "list",
    "data": [
      {
        "id": "resort_aspen_mountain",
        "object": "ski_resort",
        "name": "Aspen Mountain",
        "region": "Colorado",
        "country": "USA",
        "latitude": 39.1911,
        "longitude": -106.8175
      }
    ],
    "has_more": false,
    "next_cursor": null
  }
  ```
- **HTTP Status Codes**: `200 OK`, `400 Bad Request`.
- **Cache Policy**: `public, max-age=86400`.

---

### DOMAIN 7: VERIFICATION
*Purpose: Historical observations, forecast verification, and skill scores.*

#### 7.1 Get Verification Metrics
- **HTTP Method**: `GET`
- **Endpoint**: `/v1/verifications`
- **Purpose**: Retrieve historical error metrics (RMSE, bias, MAE) and skill scores for specific models.
- **Required Parameters**: `model`, `start_date`, `end_date`.
- **Example Request**: `GET /v1/verifications?model=gfs&start_date=2026-06-01&end_date=2026-07-01`
- **Example Response**:
  ```json
  {
    "object": "verification_report",
    "data": {
      "model": "gfs",
      "period": { "start": "2026-06-01", "end": "2026-07-01" },
      "metrics": {
        "temperature_2m_rmse": 1.42,
        "temperature_2m_bias": 0.15,
        "wind_speed_mae": 2.10
      }
    },
    "has_more": false,
    "next_cursor": null
  }
  ```
- **HTTP Status Codes**: `200 OK`, `400 Bad Request`.
- **Cache Policy**: `public, max-age=86400`.

---

### DOMAIN 8: ADMINISTRATION
*Purpose: System health, operational status, and API metadata.*

#### 8.1 Get System Health
- **HTTP Method**: `GET`
- **Endpoint**: `/v1/health`
- **Purpose**: Check database, Redis, and storage system connectivity.
- **Example Response**:
  ```json
  {
    "object": "health_check",
    "data": {
      "status": "healthy",
      "version": "1.1.0",
      "database": "connected",
      "redis": "connected",
      "object_storage": "connected"
    },
    "has_more": false,
    "next_cursor": null
  }
  ```
- **HTTP Status Codes**: `200 OK`, `503 Service Unavailable`.
- **Cache Policy**: `no-store`.

---

## 5. Future Expansion Strategy

The API is architected to absorb future platform expansions without breaking existing clients:
1. **ECMWF & Canada**: Seamlessly integrated by adding new model identifiers to `GET /v1/models` and accepting `?models=ecmwf,gdps` in point queries.
2. **AI Downscaling**: Exposed as high-resolution model variants or grid specifiers (e.g., `?grid=downscaled_3km`), seamlessly flowing through existing point and map endpoints.
3. **Multi-Model Ensemble (MME)**: Served as a first-class model option (`?models=mme`) utilizing the exact same schema structure as deterministic models.
4. **Weather Alerts**: Future alert polygons and warnings accessible via `/v1/alerts`.
5. **Streaming / WebSockets**: Designed to support future real-time radar and alert streams via parallel WebSocket namespaces (`wss://api.weatherplatform.com/v1/stream`).
6. **Batch Requests**: Future high-throughput batch query support via `/v1/batches`.
7. **Developer SDKs**: Strongly-typed Python, TypeScript, and Go SDKs auto-generated directly from our OpenAPI 3.1 specification.
