import type {
  EnsembleStatisticsData,
  Envelope,
  ErrorDetail,
  ErrorEnvelope,
  ForecastAvailability,
  Model,
  PointForecast,
  SearchResult,
  SpatialLayer,
  VariableResource,
} from "./types";

/**
 * Typed API client for the weather platform backend.
 *
 * Requests are same-origin: the Next.js rewrite in `next.config.mjs` proxies
 * `/v1/*` to the FastAPI service, so there is no base URL and no CORS.
 * Every success response uses the universal envelope and exposes its `data`;
 * every error is surfaced as an {@link ApiError} carrying the RFC 7807 fields.
 * Every method accepts an optional `AbortSignal` for request cancellation and
 * stale-response protection.
 */

export const API_PREFIX = "/v1";

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly type: string;
  readonly param: string | null;
  readonly requestId: string | null;

  constructor(
    message: string,
    status: number,
    code: string,
    type: string,
    param: string | null,
    requestId: string | null
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.type = type;
    this.param = param;
    this.requestId = requestId;
  }
}

/**
 * Raised when a request is cancelled via its `AbortSignal`. Hooks use this to
 * ignore cancelled requests silently (the user changed selection mid-flight),
 * distinct from a genuine network failure.
 */
export class RequestAbortedError extends Error {
  constructor() {
    super("Request aborted.");
    this.name = "RequestAbortedError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_PREFIX}${path}`, {
      ...init,
      headers: { Accept: "application/json", ...(init?.headers ?? {}) },
    });
  } catch (err) {
    if (err instanceof Error && err.name === "AbortError") {
      throw new RequestAbortedError();
    }
    throw new ApiError("Network request failed.", 0, "network_error", "network_error", null, null);
  }

  if (!response.ok) {
    throw await toApiError(response);
  }

  const envelope = (await response.json()) as Envelope<T>;
  return envelope.data;
}

async function toApiError(response: Response): Promise<ApiError> {
  let detail: ErrorDetail | null = null;
  try {
    const body = (await response.json()) as ErrorEnvelope;
    detail = body.error;
  } catch {
    // Response body was not JSON; fall through to a generic error.
  }
  return new ApiError(
    detail?.message ?? `Request failed with status ${response.status}.`,
    response.status,
    detail?.code ?? "invalid_request_error",
    detail?.type ?? "invalid_request_error",
    detail?.param ?? null,
    detail?.request_id ?? null
  );
}

export interface ListModelsOptions {
  center_id?: string;
  is_ensemble?: boolean;
}

export async function listModels(options: ListModelsOptions = {}): Promise<Model[]> {
  const params = new URLSearchParams();
  if (options.center_id !== undefined) {
    params.set("center_id", options.center_id);
  }
  if (options.is_ensemble !== undefined) {
    params.set("is_ensemble", String(options.is_ensemble));
  }
  const query = params.toString();
  return request<Model[]>(`/models${query ? `?${query}` : ""}`);
}

export interface GetMapLayerParams {
  model: string;
  variable: string;
  level?: "surface";
  leadTimeHours: number;
  /** ISO 8601 UTC cycle time pinning the model run (optional). */
  initialTime?: string;
  signal?: AbortSignal;
}

export async function getMapLayer({
  model,
  variable,
  level = "surface",
  leadTimeHours,
  initialTime,
  signal,
}: GetMapLayerParams): Promise<SpatialLayer> {
  const params = new URLSearchParams({
    model,
    variable,
    level,
    lead_time_hours: String(leadTimeHours),
  });
  if (initialTime !== undefined) {
    params.set("initial_time", initialTime);
  }
  return request<SpatialLayer>(`/maps?${params.toString()}`, { signal });
}

export async function getForecastAvailability(signal?: AbortSignal): Promise<ForecastAvailability> {
  return request<ForecastAvailability>(`/forecast/availability`, { signal });
}

export interface SearchLocationsOptions {
  q: string;
  type?: "city" | "resort" | "station" | "all";
  limit?: number;
  signal?: AbortSignal;
}

export async function searchLocations({
  q,
  type = "all",
  limit,
  signal,
}: SearchLocationsOptions): Promise<SearchResult[]> {
  const params = new URLSearchParams({ q, type });
  if (limit !== undefined) {
    params.set("limit", String(limit));
  }
  return request<SearchResult[]>(`/search?${params.toString()}`, { signal });
}

export type PointLocationSpecifier =
  | { type: "coordinates"; latitude: number; longitude: number }
  | { type: "city"; cityId: string }
  | { type: "resort"; resortId: string };

export interface GetPointForecastParams {
  location: PointLocationSpecifier;
  model?: string;
  units?: "metric" | "imperial";
  variables?: string[];
  startLeadTimeHours?: number;
  endLeadTimeHours?: number;
  signal?: AbortSignal;
}

export async function getPointForecast({
  location,
  model = "gfs",
  units = "metric",
  variables,
  startLeadTimeHours,
  endLeadTimeHours,
  signal,
}: GetPointForecastParams): Promise<PointForecast> {
  const params = new URLSearchParams({ models: model, units });
  if (location.type === "coordinates") {
    params.set("lat", String(location.latitude));
    params.set("lon", String(location.longitude));
  } else if (location.type === "city") {
    params.set("city_id", location.cityId);
  } else {
    params.set("resort_id", location.resortId);
  }
  if (variables !== undefined && variables.length > 0) {
    params.set("variables", variables.join(","));
  }
  if (startLeadTimeHours !== undefined) {
    params.set("start_lead_time_hours", String(startLeadTimeHours));
  }
  if (endLeadTimeHours !== undefined) {
    params.set("end_lead_time_hours", String(endLeadTimeHours));
  }
  return request<PointForecast>(`/points?${params.toString()}`, { signal });
}

export interface GetEnsembleStatisticsParams {
  latitude: number;
  longitude: number;
  variable: string;
  model?: string;
  leadTimeHours?: number;
  /** Opt into the raw ensemble-member values (Ensemble Distribution View). */
  includeMembers?: boolean;
  signal?: AbortSignal;
}

export async function getEnsembleStatistics({
  latitude,
  longitude,
  variable,
  model = "gefs",
  leadTimeHours = 0,
  includeMembers = false,
  signal,
}: GetEnsembleStatisticsParams): Promise<EnsembleStatisticsData> {
  const params = new URLSearchParams({
    lat: String(latitude),
    lon: String(longitude),
    variable,
    model,
    lead_time_hours: String(leadTimeHours),
  });
  if (includeMembers) {
    params.set("include_members", "true");
  }
  return request<EnsembleStatisticsData>(`/ensembles?${params.toString()}`, { signal });
}

export async function listVariables(signal?: AbortSignal): Promise<VariableResource[]> {
  return request<VariableResource[]>(`/variables`, { signal });
}
