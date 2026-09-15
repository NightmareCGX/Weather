"use client";

import { useForecastSelection } from "@/context/forecast-selection";
import { useDisplayTimezone } from "@/context/selected-location";
import { formatDayHourUtc, formatDayHourWithTimeZone } from "@/lib/forecast/time";

/**
 * Presentation-only forecast selection controls for the map layer (Lifecycle V2).
 *
 * All options are derived from the shared forecast-selection context:
 * Model, Variable, and Valid Time dropdowns. Initial Time and Lead Time controls
 * are removed as primary selectors.
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
    setValidTime,
    retry,
  } = useForecastSelection();
  const displayTimezone = useDisplayTimezone();

  if (status === "loading" || status === "idle") {
    return (
      <div
        className="border-b border-slate-800 bg-slate-900/90 px-4 py-2.5 text-sm text-slate-400"
        role="status"
      >
        Loading forecast options…
      </div>
    );
  }

  if (status === "error" || availability === null) {
    return (
      <div
        className="flex flex-wrap items-center gap-2 border-b border-red-900/50 bg-red-950/40 px-4 py-2.5 text-sm text-red-300"
        role="alert"
      >
        <span>Unable to load forecast data.</span>
        <button
          type="button"
          onClick={retry}
          className="rounded border border-red-800 bg-red-900/40 px-2 py-0.5 text-xs font-medium text-red-200 hover:bg-red-800/50"
        >
          Retry
        </button>
      </div>
    );
  }

  if (availability.models.length === 0 || selection === null) {
    return (
      <div className="border-b border-slate-800 bg-slate-900/90 px-4 py-2.5 text-sm text-slate-400">
        No forecast data available.
      </div>
    );
  }

  return (
    <div className="flex flex-wrap items-center gap-4 border-b border-slate-800 bg-slate-900/90 px-4 py-2">
      <label className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-400">
        Model
        <select
          className="rounded-md border border-slate-700 bg-slate-800/90 px-2.5 py-1 text-sm font-medium text-slate-200 outline-none transition focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500"
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

      <label className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-400">
        Variable
        <select
          className="rounded-md border border-slate-700 bg-slate-800/90 px-2.5 py-1 text-sm font-medium text-slate-200 outline-none transition focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500"
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

      <label className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-400">
        Valid Time
        <select
          className="rounded-md border border-slate-700 bg-slate-800/90 px-2.5 py-1 text-sm font-medium text-slate-200 font-mono outline-none transition focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500"
          value={selection.validTime ?? ""}
          onChange={(event) => setValidTime?.(event.target.value)}
          aria-label="Valid time"
        >
          {(options.validTimes ?? []).map((vt) => (
            <option key={vt} value={vt}>
              {formatDayHourUtc(vt)} UTC
            </option>
          ))}
        </select>
      </label>

      {validTime !== null && (
        <span className="text-sm text-slate-500" data-testid="valid-time">
          Valid {formatDayHourWithTimeZone(validTime, displayTimezone)}
        </span>
      )}
    </div>
  );
}
