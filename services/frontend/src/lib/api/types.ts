/**
 * Shared types for the frontend API client.
 *
 * These mirror the FastAPI response schemas in
 * `services/api/src/api/schemas.py` (universal envelope, model catalog,
 * and spatial-layer metadata).
 */

export interface Model {
  id: string;
  object: "model";
  name: string;
  center_id: string;
  is_ensemble: boolean;
  resolution_km: number;
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
