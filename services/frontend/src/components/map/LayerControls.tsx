"use client";

import { useForecastSelection } from "@/context/forecast-selection";
import { formatInitialTimeLabel } from "@/lib/forecast/labels";
import { formatLeadTimeHours } from "@/lib/forecast/time";

/**
 * Presentation-only forecast selection controls for the map layer.
 *
 * All options are derived from the shared forecast-selection context (which is
 * database-driven via `/v1/forecast/availability`): the Model dropdown lists
 * exactly the models in the database, Variable lists the selected model's
 * variables, Initial Time lists the selected variable's initial times, and
 * Lead Time lists the selected initial time's lead times. Nothing here is
 * hard-coded.
 */
export function LayerControls() {
  const {
    availability,
    status,
    error,
    selection,
    options,
    validTime,
    setModel,
    setVariable,
    setInitialTime,
    setLeadTimeHours,
    retry,
  } = useForecastSelection();

  if (status === "loading" || status === "idle") {
    return (
      <div
        className="border-b border-slate-200 bg-white px-4 py-2 text-sm text-slate-500"
        role="status"
      >
        Loading forecast options…
      </div>
    );
  }

  if (status === "error" || availability === null) {
    return (
      <div
        className="flex flex-wrap items-center gap-2 border-b border-red-200 bg-red-50 px-4 py-2 text-sm text-red-700"
        role="alert"
      >
        <span>Unable to load forecast data.</span>
        <button
          type="button"
          onClick={retry}
          className="rounded border border-red-300 px-2 py-0.5 text-xs font-medium hover:bg-red-100"
        >
          Retry
        </button>
      </div>
    );
  }

  if (availability.models.length === 0 || selection === null) {
    return (
      <div className="border-b border-slate-200 bg-white px-4 py-2 text-sm text-slate-500">
        No forecast data available.
      </div>
    );
  }

  return (
    <div className="flex flex-wrap items-center gap-4 border-b border-slate-200 bg-white px-4 py-2">
      <label className="flex items-center gap-2 text-sm text-slate-700">
        Model
        <select
          className="rounded border border-slate-300 px-2 py-1"
          value={selection.model}
          onChange={(event) => setModel(event.target.value)}
          aria-label="Model"
        >
          {options.models.map((model) => (
            <option key={model.id} value={model.id}>
              {model.name}
            </option>
          ))}
        </select>
      </label>

      <label className="flex items-center gap-2 text-sm text-slate-700">
        Variable
        <select
          className="rounded border border-slate-300 px-2 py-1"
          value={selection.variable}
          onChange={(event) => setVariable(event.target.value)}
          aria-label="Variable"
        >
          {options.variables.map((variable) => (
            <option key={variable.id} value={variable.id}>
              {variable.name}
            </option>
          ))}
        </select>
      </label>

      <label className="flex items-center gap-2 text-sm text-slate-700">
        Initial Time
        <select
          className="rounded border border-slate-300 px-2 py-1"
          value={selection.initialTime}
          onChange={(event) => setInitialTime(event.target.value)}
          aria-label="Initial time"
        >
          {options.initialTimes.map((entry) => (
            <option key={entry.value} value={entry.value}>
              {formatInitialTimeLabel(entry.value)}
            </option>
          ))}
        </select>
      </label>

      <label className="flex items-center gap-2 text-sm text-slate-700">
        Lead time
        <select
          className="rounded border border-slate-300 px-2 py-1"
          value={selection.leadTimeHours}
          onChange={(event) => setLeadTimeHours(Number(event.target.value))}
          aria-label="Lead time"
        >
          {options.leadTimes.map((lead) => (
            <option key={lead} value={lead}>
              {formatLeadTimeHours(lead)}
            </option>
          ))}
        </select>
      </label>

      {validTime !== null && (
        <span className="text-sm text-slate-500" data-testid="valid-time">
          Valid {formatInitialTimeLabel(validTime)}
        </span>
      )}
    </div>
  );
}
