/**
 * Shared types for the frontend API client.
 *
 * These mirror the FastAPI response schemas in
 * `services/api/src/api/schemas.py` (universal envelope, model catalog,
 * spatial-layer metadata, location search, point forecast, ensemble
 * statistics, and forecast-variable catalog resources).
 */

export interface Model {
  id: string;
  object: "model";
  name: string;
  center_id: string;
  is_ensemble: boolean;
  resolution_km: number;
}

/**
 * A catalog forecast variable (API.md section 1.4). `unit` carries the
 * registered SI unit string (e.g. "°C", "mm/h") used to label chart axes.
 */
export interface VariableResource {
  id: string;
  object: "variable";
  name: string;
  unit: string;
}

/**
 * A location search result (API.md section 6.1). `object` distinguishes the
 * source table (`city`, `ski_resort`, `station`) or the place-autocomplete
 * provider (`place`). Type-specific fields are optional because not every
 * source table defines them. A `place` result carries a `place_id` (the
 * provider's canonical place id) but no resolved coordinates yet — they are
 * populated only after the user selects it and the canonical place is
 * resolved.
 */
export interface SearchResult {
  id: string;
  object: "city" | "ski_resort" | "station" | "place";
  name: string;
  region: string | null;
  country: string | null;
  elevation_m: number | null;
  latitude: number;
  longitude: number;
  place_id?: string | null;
}

/**
 * The selected location shared by search, map click, and the forecast
 * dashboard. This is a frontend model, not a backend response shape: it
 * unifies a `/v1/search` result, a raw map-click coordinate, and a station
 * (which has no `/v1/points` platform specifier, so it resolves by
 * coordinates).
 */
export interface SelectedLocation {
  /** Display label (a place name, or "lat, lon" for raw coordinates). */
  name: string;
  object: "city" | "ski_resort" | "station" | "coordinates";
  latitude: number;
  longitude: number;
  elevation_m: number | null;
  region: string | null;
  country: string | null;
  /** Platform id used as a `/v1/points` spatial specifier (city/resort). */
  id: string | null;
  /** How `/v1/points` resolves this location (drives `resolved_via`). */
  resolvedVia: "city" | "resort" | "coordinates";
}

/** A single forecast entry indexed by lead time (API.md section 2.1). */
export interface ForecastEntry {
  lead_time_hours: number;
  valid_time: string;
  /** Requested forecast variables are attached as dynamic top-level keys. */
  [variableCode: string]: number | string;
}

/** The resolved location of a point forecast (API.md section 2.1). */
export interface PointForecastLocation {
  latitude: number;
  longitude: number;
  elevation_m: number | null;
  resolved_via: string;
}

/** The payload of a point forecast (API.md section 2.1). */
export interface PointForecast {
  location: PointForecastLocation;
  generated_at: string;
  model: string;
  forecasts: ForecastEntry[];
}

/** Ensemble dispersion statistics (API.md section 5.1). */
export interface EnsembleStatistics {
  mean: number;
  median: number;
  spread: number;
  p10: number;
  p25: number;
  p50: number;
  p75: number;
  p90: number;
}

/**
 * The payload of ensemble statistics (API.md section 5.1).
 *
 * `members` is an opt-in additive field (request `include_members=true`): raw
 * ensemble-member forecast values in dataset `member`-coordinate order, for
 * the requested model, location, variable, and lead time. It is absent on
 * statistics-only responses, so the Distribution View must treat absence as
 * "not yet available" rather than fabricating a distribution from the
 * aggregate statistics.
 */
export interface EnsembleStatisticsData {
  model: string;
  lead_time_hours: number;
  member_count: number;
  statistics: EnsembleStatistics;
  members?: number[];
}

/**
 * One available forecast initialization (cycle time) of a variable.
 *
 * `valid_time` for a given lead is derived as `value + lead_time_hours`
 * (DATABASE.md section 1).
 */
export interface InitialTimeAvailability {
  value: string;
  lead_time_hours: number[];
}

/** A forecast variable and the initial times available for it. */
export interface VariableAvailability {
  id: string;
  name: string;
  unit: string;
  initial_times: InitialTimeAvailability[];
}

/** A forecast model and the variables available for it. */
export interface ModelAvailability {
  id: string;
  name: string;
  is_ensemble: boolean;
  variables: VariableAvailability[];
}

/** The payload of the forecast availability endpoint. */
export interface ForecastAvailability {
  models: ModelAvailability[];
}

export type LegendStop = readonly [number, string];

export interface SpatialLayerLegend {
  unit: string;
  stops: LegendStop[];
}

export interface SpatialLayer {
  tile_url_template: string;
  min_zoom: number;
  max_zoom: number;
  lead_time_hours: number;
  legend: SpatialLayerLegend;
}

/** Universal list/single-response envelope (docs/API.md section 2.3). */
export interface Envelope<T> {
  object: string;
  data: T;
  has_more: boolean;
  next_cursor: string | null;
}

/** RFC 7807 problem-detail payload (docs/API.md section 2.4). */
export interface ErrorDetail {
  code: string;
  type: string;
  message: string;
  param: string | null;
  request_id: string | null;
}

export interface ErrorEnvelope {
  error: ErrorDetail;
}
