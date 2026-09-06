import type {
  ForecastAvailability,
  InitialTimeAvailability,
  ModelAvailability,
  SpatialLayer,
  VariableAvailability,
} from "@/lib/api/types";

/**
 * Pure, deterministic helpers for the data-driven forecast selection workflow (Lifecycle V2).
 *
 * The selection model is:
 *   Model -> Variable -> Valid Time
 *
 * User-facing Initial Time and Lead Time controls are removed. Valid times are
 * derived from the backend availability response and filtered by the 3-hour UI
 * grace window:
 *   visible if: valid_time >= now - 3h
 *   hidden if:  valid_time < now - 3h (strictly <)
 */

/** A full forecast selection under Lifecycle V2. */
export interface ForecastSelection {
  model: string;
  variable: string;
  validTime?: string;
  // Legacy optional fields for backward compatibility during transition
  initialTime?: string;
  leadTimeHours?: number;
}

/** The options currently available at each level of the selection. */
export interface ForecastOptions {
  models: ModelAvailability[];
  model: ModelAvailability | null;
  variables: VariableAvailability[];
  variable: VariableAvailability | null;
  /** Selectable valid times within the 3-hour grace window, ascending. */
  validTimes?: string[];
  /** Legacy initial times (for backward compatibility with existing tests). */
  initialTimes: InitialTimeAvailability[];
  initialTime: InitialTimeAvailability | null;
  /** Legacy lead times (for backward compatibility with existing tests). */
  leadTimes: number[];
}

export const GRACE_WINDOW_HOURS = 3;

/**
 * Check whether a valid time falls within the 3-hour UI grace window:
 *   visible if: valid_time >= now - 3h
 *   hidden if:  valid_time < now - 3h (strictly <)
 */
export function isWithinGraceWindow(validTime: string | null, nowMs: number = Date.now()): boolean {
  if (validTime === null) return false;
  const t = new Date(validTime).getTime();
  if (Number.isNaN(t)) return false;
  const threshold = nowMs - GRACE_WINDOW_HOURS * 3600 * 1000;
  return t >= threshold;
}

/**
 * Filter an array of valid times by the 3-hour UI grace window.
 */
export function filterGraceWindowValidTimes(
  validTimes: string[],
  nowMs: number = Date.now()
): string[] {
  return validTimes.filter((vt) => isWithinGraceWindow(vt, nowMs));
}

/** Pick the first model in availability (or null when empty). */
export function defaultModel(availability: ForecastAvailability | null): string | null {
  const models = availability?.models ?? [];
  return models.length > 0 ? models[0].id : null;
}

/** Resolve a model id to its availability entry (or null). */
export function findModel(
  availability: ForecastAvailability | null,
  modelId: string | null
): ModelAvailability | null {
  if (availability === null || modelId === null) {
    return null;
  }
  return availability.models.find((model) => model.id === modelId) ?? null;
}

/** Pick the first variable of a model that has available data. */
export function defaultVariable(model: ModelAvailability | null): string | null {
  if (model === null || model.variables.length === 0) {
    return null;
  }
  return model.variables[0].id;
}

/** Resolve a variable code within a model to its availability entry (or null). */
export function findVariable(
  model: ModelAvailability | null,
  variableId: string | null
): VariableAvailability | null {
  if (model === null || variableId === null) {
    return null;
  }
  return model.variables.find((variable) => variable.id === variableId) ?? null;
}

/**
 * Extract all unique valid times for a variable (from valid_times or synthesized
 * from legacy initial_times), sorted chronologically ascending.
 */
export function extractVariableValidTimes(variable: VariableAvailability | null): string[] {
  if (variable === null) {
    return [];
  }
  if (variable.valid_times && variable.valid_times.length > 0) {
    const times = new Set(variable.valid_times.map((vt) => vt.valid_time));
    return Array.from(times).sort((a, b) => new Date(a).getTime() - new Date(b).getTime());
  }

  // Fallback synthesis from legacy initial_times: cycle_time + lead_time_hours
  const times = new Set<string>();
  for (const it of variable.initial_times) {
    const base = new Date(it.value);
    if (Number.isNaN(base.getTime())) continue;
    for (const lead of it.lead_time_hours) {
      const vt = new Date(base);
      vt.setUTCHours(vt.getUTCHours() + lead);
      times.add(vt.toISOString());
    }
  }
  return Array.from(times).sort((a, b) => new Date(a).getTime() - new Date(b).getTime());
}

/** Pick the default valid time (first available valid time within grace window). */
export function defaultValidTime(validTimes: string[], nowMs: number = Date.now()): string | null {
  const filtered = filterGraceWindowValidTimes(validTimes, nowMs);
  return filtered.length > 0 ? filtered[0] : validTimes.length > 0 ? validTimes[0] : null;
}

/** Legacy helper: default initial time */
export function defaultInitialTime(variable: VariableAvailability | null): string | null {
  if (variable === null || variable.initial_times.length === 0) {
    return null;
  }
  return variable.initial_times[0].value;
}

/** Legacy helper: find initial time */
export function findInitialTime(
  variable: VariableAvailability | null,
  initialTime: string | null
): InitialTimeAvailability | null {
  if (variable === null || initialTime === null) {
    return null;
  }
  return variable.initial_times.find((entry) => entry.value === initialTime) ?? null;
}

/** Legacy helper: default lead time */
export function defaultLeadTime(initialTime: InitialTimeAvailability | null): number | null {
  const leads = initialTime?.lead_time_hours ?? [];
  return leads.length > 0 ? leads[0] : null;
}

/**
 * Build the options for a selection from availability, filtered by grace window.
 */
export function buildForecastOptions(
  availability: ForecastAvailability | null,
  selection: ForecastSelection | null,
  nowMs: number = Date.now()
): ForecastOptions {
  if (availability === null) {
    return emptyOptions();
  }

  const model = selection !== null ? findModel(availability, selection.model) : null;
  const variable =
    model !== null && selection !== null ? findVariable(model, selection.variable) : null;

  const allValidTimes = extractVariableValidTimes(variable);
  const selectableValidTimes = filterGraceWindowValidTimes(allValidTimes, nowMs);

  const initialTime =
    variable !== null && selection !== null && selection.initialTime
      ? findInitialTime(variable, selection.initialTime)
      : (variable?.initial_times[0] ?? null);

  return {
    models: availability.models,
    model,
    variables: model?.variables ?? [],
    variable,
    validTimes: selectableValidTimes.length > 0 ? selectableValidTimes : allValidTimes,
    initialTimes: variable?.initial_times ?? [],
    initialTime,
    leadTimes: initialTime?.lead_time_hours ?? [],
  };
}

function emptyOptions(): ForecastOptions {
  return {
    models: [],
    model: null,
    variables: [],
    variable: null,
    validTimes: [],
    initialTimes: [],
    initialTime: null,
    leadTimes: [],
  };
}

/**
 * Compute valid time string from (initialTime, leadTimeHours) if provided.
 */
export function resolveValidTime(
  initialTime: string | null,
  leadTimeHours: number | null
): string | null {
  if (initialTime === null || leadTimeHours === null) {
    return null;
  }
  const parsed = new Date(initialTime);
  if (Number.isNaN(parsed.getTime())) {
    return null;
  }
  parsed.setUTCHours(parsed.getUTCHours() + leadTimeHours);
  return parsed.toISOString();
}

/**
 * Synchronously construct the authoritative SpatialLayer for a valid selection
 * under Lifecycle V2 using the backend-provided layer descriptor.
 */
export function resolveSpatialLayer(
  availability: ForecastAvailability | null,
  selection: ForecastSelection | null
): SpatialLayer | null {
  if (availability === null || selection === null) {
    return null;
  }
  const model = findModel(availability, selection.model);
  const variable = findVariable(model, selection.variable);
  if (variable === null || !variable.layer) {
    return null;
  }

  // Validate availability before constructing layer
  if (selection.validTime) {
    const selMs = new Date(selection.validTime).getTime();
    const validTimes = extractVariableValidTimes(variable);
    const hasValid = validTimes.some((vt) => new Date(vt).getTime() === selMs);
    if (!hasValid) {
      return null;
    }
  } else if (selection.initialTime && selection.leadTimeHours !== undefined) {
    const initial = findInitialTime(variable, selection.initialTime);
    if (initial === null || !initial.lead_time_hours.includes(selection.leadTimeHours)) {
      return null;
    }
  } else {
    return null;
  }

  const {
    tile_url_template,
    valid_time_tile_url_template,
    valid_time_vector_field_url_template,
    min_zoom,
    max_zoom,
    legend,
    vector_field_url_template,
  } = variable.layer;

  let tileUrl: string;
  let vectorFieldUrl: string | null = null;

  if (selection.validTime) {
    const encodedVt = encodeURIComponent(selection.validTime);
    if (valid_time_tile_url_template) {
      tileUrl = valid_time_tile_url_template.replace("{valid_time}", encodedVt);
    } else if (tile_url_template.includes("{valid_time}")) {
      tileUrl = tile_url_template.replace("{valid_time}", encodedVt);
    } else {
      tileUrl = `/v1/maps/${selection.model}/${selection.variable}/surface/{z}/{x}/{y}.png?valid_time=${encodedVt}`;
    }
    if (valid_time_vector_field_url_template) {
      vectorFieldUrl = valid_time_vector_field_url_template.replace("{valid_time}", encodedVt);
    } else if (vector_field_url_template) {
      vectorFieldUrl = `/v1/maps/${selection.model}/wind_10m/vector-field?valid_time=${encodedVt}`;
    }
  } else {
    // Legacy fallback
    tileUrl = tile_url_template
      .replace("{lead_time_hours}", String(selection.leadTimeHours ?? 0))
      .replace("{initial_time}", encodeURIComponent(selection.initialTime ?? ""));
    if (vector_field_url_template) {
      vectorFieldUrl = vector_field_url_template
        .replace("{lead_time_hours}", String(selection.leadTimeHours ?? 0))
        .replace("{initial_time}", encodeURIComponent(selection.initialTime ?? ""));
    }
  }

  const layerResult: SpatialLayer = {
    tile_url_template: tileUrl,
    min_zoom,
    max_zoom,
    lead_time_hours: selection.leadTimeHours ?? 0,
    legend,
  };
  if (selection.validTime) {
    layerResult.valid_time = selection.validTime;
  }
  if (vectorFieldUrl !== null) {
    layerResult.vector_field_url_template = vectorFieldUrl;
  }
  return layerResult;
}
