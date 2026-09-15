"use client";

import { buildLegendGradient, getLegendTicks } from "@/lib/map/legend";
import type { SpatialLayer } from "@/lib/api/types";

interface LegendProps {
  layer: SpatialLayer | null;
  variableName?: string | null;
  variableCode?: string | null;
}

/**
 * Presentation-only legend bar rendered from `/v1/maps` metadata. The legend
 * reflects the layer's registered unit and color stops, with dynamic physical ticks (Scheme B).
 */
export function Legend({ layer, variableName, variableCode }: LegendProps) {
  if (layer === null) {
    return null;
  }

  const { unit, stops } = layer.legend;
  const label = variableName
    ? variableName.includes("(")
      ? variableName
      : `${variableName} (${unit})`
    : unit;

  // Infer variableCode from tile_url_template if not directly supplied
  let inferredCode = variableCode;
  if (!inferredCode && layer.tile_url_template) {
    const match = layer.tile_url_template.match(/\/v1\/maps\/[^/]+\/([^/]+)\//);
    if (match) {
      inferredCode = match[1];
    }
  }

  const ticks = getLegendTicks(stops, inferredCode);

  return (
    <div className="pointer-events-none absolute bottom-4 left-4 z-10 rounded border border-slate-200 bg-white/95 px-3 py-2 shadow">
      <div className="mb-1 text-xs font-medium text-slate-700">{label}</div>
      <div
        className="h-3 w-56 rounded"
        data-testid="legend-gradient"
        style={{ backgroundImage: buildLegendGradient(stops, inferredCode) }}
      />
      {ticks.length > 0 && (
        <div className="relative mt-1 h-3.5 w-56 text-[10px] text-slate-500 tabular-nums">
          {ticks.map((tick, index) => {
            const isFirst = index === 0;
            const isLast = index === ticks.length - 1;
            const alignClass = isFirst
              ? "left-0 text-left"
              : isLast
                ? "right-0 text-right"
                : "-translate-x-1/2 text-center";

            return (
              <span
                key={`${tick.value}-${index}`}
                data-testid={`legend-tick-${index}`}
                style={{
                  left: isLast ? undefined : `${tick.positionPercent}%`,
                }}
                className={`absolute top-0 leading-none ${alignClass}`}
              >
                {tick.label}
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
}
