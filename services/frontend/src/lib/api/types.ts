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
  /**
   * The source forecast run's cycle time. The point forecast is a cross-cycle
   * deterministic time series: entries may come from different cycles (the
   * minimum-lead record for each valid_time), so the source cycle is exposed
   * to make provenance unambiguous.
   */
  cycle_time?: string;
  /** Requested forecast variables are attached as dynamic top-level keys. */
  [variableCode: string]: number | string | undefined;
}

/**
 * Structural/metadata keys carried by a {@link ForecastEntry} that are NOT
 * forecast data variables (API.md section 2.1: the point forecast is a
 * cross-cycle series where each entry exposes the source ``cycle_time``).
 *
 * These fields describe the entry itself (its lead offset, its valid time,
 * the run that produced it) rather than rendering as a meteorological
 * variable, so every consumer that derives the "real" forecast-variable set
 * from a series MUST treat them as excluded. Because the backend attaches
 * variable codes as dynamic keys, the only sound way to separate the two is
 * this explicit structural-key set — an allow-by-schema convention. Keeping
 * the set adjacent to {@link ForecastEntry} ensures a future additive
 * metadata field is captured here, in the same review, instead of surfacing
 * later as a bogus chart/ensemble/map request.
 */
export const FORECAST_ENTRY_METADATA_FIELDS: ReadonlySet<string> = new Set([
  "lead_time_hours",
  "valid_time",
  "cycle_time",
  "wind_direction_10m",
  "wind_cardinal_10m",
  "precipitation_type",
  "precipitation_transition",
  "precipitation_start_type",
  "precipitation_end_type",
  "precipitation_evidence",
]);

/**
 * Whether a forecast-entry key is a structural/metadata field rather than a
 * forecast data variable. Shared by the point-forecast and map-variable
 * extractors so the metadata boundary is governed by one source of truth.
 */
export function isForecastEntryMetadataField(key: string): boolean {
  return FORECAST_ENTRY_METADATA_FIELDS.has(key);
}

/** Whether a forecast-entry key is a candidate forecast data variable. */
export function isForecastDataVariable(key: string): boolean {
  return !isForecastEntryMetadataField(key);
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
 * Canonical ensemble probability density function evaluation.
 */
export interface EnsemblePDF {
  x: number[];
  density: number[];
}

export interface ConsensusVector {
  speed: number;
  direction: number | null;
  cardinal: string;
  coherence: number;
}

export interface WindRoseSector {
  sector: string;
  count: number;
  probability: number;
  bins: Record<string, number>;
}

export interface WindRose {
  calm_percentage: number;
  calm_count: number;
  sectors: WindRoseSector[];
  member_count?: number;
}

export type PrecipitationType =
  "none" | "rain" | "snow" | "freezing_rain" | "ice_pellets" | "mixed" | "unknown";

export type PhysicalPhase = "dry" | "rain" | "snow" | "freezing_rain" | "ice_pellets" | "unknown";

export type PrecipitationTransition =
  | "none"
  | "persistent_rain"
  | "persistent_snow"
  | "persistent_freezing_rain"
  | "persistent_ice_pellets"
  | "dry_to_rain"
  | "dry_to_snow"
  | "dry_to_freezing_rain"
  | "dry_to_ice_pellets"
  | "wet_to_dry"
  | "rain_to_snow"
  | "snow_to_rain"
  | "rain_to_freezing_rain"
  | "freezing_rain_to_rain"
  | "snow_to_freezing_rain"
  | "freezing_rain_to_snow"
  | "snow_to_ice_pellets"
  | "ice_pellets_to_snow"
  | "mixed_transition"
  | "unknown";

export type EvidenceState = "exact" | "strongly_inferred" | "ambiguous";

/**
 * The payload of ensemble statistics (API.md section 5.1).
 *
 * `members` is an opt-in additive field (request `include_members=true`): raw
 * ensemble-member forecast values in dataset `member`-coordinate order, for
 * the requested model, location, variable, and lead time. `pdf` carries the
 * canonical 1-D Gaussian Kernel Density Estimate (or `null` when variation is
 * degenerate). Both are absent on statistics-only responses, so the
 * Distribution View must treat absence as "not yet available" rather than
 * fabricating a distribution from aggregate statistics.
 */
export interface EnsembleStatisticsData {
  model: string;
  lead_time_hours: number;
  member_count: number;
  statistics: EnsembleStatistics;
  members?: number[];
  pdf?: EnsemblePDF | null;
  consensus_vector?: ConsensusVector | null;
  wind_rose?: WindRose | null;
  phase_support?: Record<string, number> | null;
  transition_frequency?: Record<string, number> | null;
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

export type LegendStop = readonly [number, string];

export interface SpatialLayerLegend {
  unit: string;
  stops: LegendStop[];
}

export interface LayerDescriptor {
  tile_url_template: string;
  min_zoom: number;
  max_zoom: number;
  legend: SpatialLayerLegend;
  vector_field_url_template?: string | null;
}

/** A forecast variable and the initial times available for it. */
export interface VariableAvailability {
  id: string;
  name: string;
  unit: string;
  initial_times: InitialTimeAvailability[];
  layer?: LayerDescriptor | null;
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

export interface SpatialLayer {
  tile_url_template: string;
  min_zoom: number;
  max_zoom: number;
  lead_time_hours: number;
  legend: SpatialLayerLegend;
  vector_field_url_template?: string | null;
}

export interface VectorGridMetadata {
  lat_start: number;
  lat_step: number;
  lat_count: number;
  lon_start: number;
  lon_step: number;
  lon_count: number;
  scale: number;
}

export interface VectorFieldData {
  meta: VectorGridMetadata;
  u: Float32Array;
  v: Float32Array;
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
