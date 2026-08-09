"use client";

import { buildLegendGradient } from "@/lib/map/legend";
import type { SpatialLayer } from "@/lib/api/types";

interface LegendProps {
  layer: SpatialLayer | null;
}

/**
 * Presentation-only legend bar rendered from `/v1/maps` metadata. The legend
 * reflects the layer's registered unit and color stops.
 */
export function Legend({ layer }: LegendProps) {
  if (layer === null) {
    return null;
  }

  const { unit, stops } = layer.legend;
  return (
    <div className="pointer-events-none absolute bottom-4 left-4 z-10 rounded border border-slate-200 bg-white/95 px-3 py-2 shadow">
      <div className="mb-1 text-xs font-medium text-slate-700">{unit}</div>
      <div
        className="h-3 w-48 rounded"
        data-testid="legend-gradient"
        style={{ backgroundImage: buildLegendGradient(stops) }}
      />
    </div>
  );
}
