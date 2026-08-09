import type { LegendStop } from "@/lib/api/types";

/**
 * Build a CSS `linear-gradient(...)` value from a legend's color stops.
 *
 * Stops are `[value, color]` pairs ordered ascending by value. The gradient
 * is normalized to the stop values' range so the CSS gradient spans the full
 * 0..100% width regardless of the underlying units.
 */
export function buildLegendGradient(stops: readonly LegendStop[]): string {
  if (stops.length === 0) {
    return "linear-gradient(to right, transparent, transparent)";
  }

  const first = stops[0][0];
  const last = stops[stops.length - 1][0];
  const range = last - first;

  const parts = stops.map(([value, color]) => {
    const position = range === 0 ? 0 : ((value - first) / range) * 100;
    return `${color} ${position}%`;
  });

  return `linear-gradient(to right, ${parts.join(", ")})`;
}
