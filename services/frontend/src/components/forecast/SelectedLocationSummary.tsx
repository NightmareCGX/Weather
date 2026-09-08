"use client";

import { locationTypeLabel } from "@/lib/forecast/selection";
import type { SelectedLocation } from "@/lib/api/types";

interface SelectedLocationSummaryProps {
  location: SelectedLocation;
  onClose?: () => void;
}

/**
 * Presentation-only summary of the selected location: type badge, name,
 * region/country, coordinates, and elevation (when the resolved record defines
 * one). Raw coordinate selections show the coordinate label and no elevation.
 *
 * Includes an optional Close button to deselect the location.
 */
export function SelectedLocationSummary({ location, onClose }: SelectedLocationSummaryProps) {
  return (
    <section aria-label="Selected location" className="border-b border-slate-200 px-4 py-3">
      <div className="flex items-center justify-between gap-2">
        <div className="flex min-w-0 items-center gap-2">
          <span className="shrink-0 rounded bg-slate-100 px-1.5 py-0.5 text-xs font-medium text-slate-600">
            {locationTypeLabel(location)}
          </span>
          <h2 className="truncate text-sm font-semibold text-slate-900">{location.name}</h2>
        </div>
        {onClose && (
          <button
            type="button"
            aria-label="Close forecast panel"
            onClick={onClose}
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded text-slate-400 hover:bg-slate-100 hover:text-slate-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-500"
          >
            <svg
              className="h-4 w-4"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
              focusable="false"
            >
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        )}
      </div>
      {(location.region !== null || location.country !== null) && (
        <p className="mt-0.5 text-xs text-slate-500">
          {[location.region, location.country].filter(Boolean).join(", ")}
        </p>
      )}
      <dl className="mt-1.5 grid grid-cols-2 gap-x-4 gap-y-0.5 text-xs text-slate-600">
        <div className="flex justify-between">
          <dt>Latitude</dt>
          <dd className="tabular-nums">{location.latitude.toFixed(4)}</dd>
        </div>
        <div className="flex justify-between">
          <dt>Longitude</dt>
          <dd className="tabular-nums">{location.longitude.toFixed(4)}</dd>
        </div>
        <div className="flex justify-between">
          <dt>Elevation</dt>
          <dd className="tabular-nums">
            {location.elevation_m !== null
              ? `${Math.round(location.elevation_m).toLocaleString()} m`
              : "unavailable"}
          </dd>
        </div>
      </dl>
    </section>
  );
}
