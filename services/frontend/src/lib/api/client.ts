import type { Envelope, ErrorDetail, ErrorEnvelope, Model, SpatialLayer } from "./types";

/**
 * Typed API client for the weather platform backend.
 *
 * Requests are same-origin: the Next.js rewrite in `next.config.mjs` proxies
 * `/v1/*` to the FastAPI service, so there is no base URL and no CORS.
 * Every success response uses the universal envelope and exposes its `data`;
 * every error is surfaced as an {@link ApiError} carrying the RFC 7807 fields.
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

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_PREFIX}${path}`, {
      ...init,
      headers: { Accept: "application/json", ...(init?.headers ?? {}) },
    });
  } catch {
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
}

export async function getMapLayer({
  model,
  variable,
  level = "surface",
  leadTimeHours,
}: GetMapLayerParams): Promise<SpatialLayer> {
  const params = new URLSearchParams({
    model,
    variable,
    level,
    lead_time_hours: String(leadTimeHours),
  });
  return request<SpatialLayer>(`/maps?${params.toString()}`);
}
