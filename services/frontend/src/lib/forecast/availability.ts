import type {
  ForecastAvailability,
  InitialTimeAvailability,
  ModelAvailability,
  VariableAvailability,
} from "@/lib/api/types";

/**
 * Pure, deterministic helpers for the data-driven forecast selection workflow.
 *
 * Every option in the Model / Variable / Initial Time / Lead Time selectors
 * is derived from the backend's forecast availability response (which is in
 * turn generated from the database), so the UI never hard-codes what forecast
 * data exists. Changing an upstream selection narrows the downstream options
 * to exactly what the availability data allows.
 */

/** A full forecast selection (mirrors the map configuration). */
export interface ForecastSelection {
  model: string;
  variable: string;
  initialTime: string;
  leadTimeHours: number;
}

/** The options currently available at each level of the selection cascade. */
export interface ForecastOptions {
  models: ModelAvailability[];
  /** The selected model's availability, or null when none selected. */
  model: ModelAvailability | null;
  /** The selected model's variables, or empty. */
  variables: VariableAvailability[];
  /** The selected variable's initial times, or empty. */
  initialTimes: InitialTimeAvailability[];
  /** The selected variable's availability, or null when none selected. */
  variable: VariableAvailability | null;
  /** The selected initial time's availability, or null when none selected. */
  initialTime: InitialTimeAvailability | null;
  /** The available lead times for the current selection, ascending. */
  leadTimes: number[];
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

/**
 * Pick the first variable of a model that has at least one initial time.
 * Returns null when the model has no available variables.
 */
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
 * Pick the first initial time of a variable (newest first), or null when the
 * variable has no initial times.
 */
export function defaultInitialTime(variable: VariableAvailability | null): string | null {
  if (variable === null || variable.initial_times.length === 0) {
    return null;
  }
  return variable.initial_times[0].value;
}

/** Resolve an initial time within a variable to its availability entry (or null). */
export function findInitialTime(
  variable: VariableAvailability | null,
  initialTime: string | null
): InitialTimeAvailability | null {
  if (variable === null || initialTime === null) {
    return null;
  }
  return variable.initial_times.find((entry) => entry.value === initialTime) ?? null;
}

/**
 * Pick the first lead time of an initial time, or null when none exists.
 */
export function defaultLeadTime(initialTime: InitialTimeAvailability | null): number | null {
  const leads = initialTime?.lead_time_hours ?? [];
  return leads.length > 0 ? leads[0] : null;
}

/**
 * Build the cascading options for a selection from availability.
 *
 * The returned structure contains exactly the options the current selection
 * makes available at each downstream level, plus the resolved availability
 * entries. Callers use this to render the selectors and to reset stale
 * downstream selections when an upstream selection changes.
 */
export function buildForecastOptions(
  availability: ForecastAvailability | null,
  selection: ForecastSelection | null
): ForecastOptions {
  if (availability === null) {
    return emptyOptions();
  }

  const model = selection !== null ? findModel(availability, selection.model) : null;
  const variable =
    model !== null && selection !== null ? findVariable(model, selection.variable) : null;
  const initialTime =
    variable !== null && selection !== null
      ? findInitialTime(variable, selection.initialTime)
      : null;

  return {
    models: availability.models,
    model,
    variables: model?.variables ?? [],
    initialTimes: variable?.initial_times ?? [],
    variable,
    initialTime,
    leadTimes: initialTime?.lead_time_hours ?? [],
  };
}

function emptyOptions(): ForecastOptions {
  return {
    models: [],
    model: null,
    variables: [],
    initialTimes: [],
    variable: null,
    initialTime: null,
    leadTimes: [],
  };
}

/**
 * Compute the valid time for a selection: `initial_time + lead_time_hours`
 * (DATABASE.md section 1). Returns null when either part is missing.
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
