"use client";

import { locationTypeLabel } from "@/lib/forecast/selection";
import type { SelectedLocation } from "@/lib/api/types";

interface SelectedLocationSummaryProps {
  location: SelectedLocation;
}

/**
 * Presentation-only summary of the selected location: type badge, name,
 * region/country, coordinates, and elevation (when the resolved record defines
 * one). Raw coordinate selections show the coordinate label and no elevation.
 */
export function SelectedLocationSummary({ location }: SelectedLocationSummaryProps) {
  return (
    <section aria-label="Selected location" className="border-b border-slate-200 px-4 py-3">
      <div className="flex items-center gap-2">
        <span className="rounded bg-slate-100 px-1.5 py-0.5 text-xs font-medium text-slate-600">
          {locationTypeLabel(location)}
        </span>
        <h2 className="truncate text-sm font-semibold text-slate-900">{location.name}</h2>
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
