import type { LegendStop } from "@/lib/api/types";

export interface LegendTick {
  value: number;
  label: string;
  positionPercent: number;
}

/** Custom non-linear visual position percentage map for specific stretched variables. */
export const STRETCH_POSITION_PERCENT: Record<string, Record<number, number>> = {
  cloud_ceiling: {
    0.0: 0,
    3.0: 30,
    9.0: 60,
    20.0: 100,
  },
  visibility: {
    0.0: 0,
    1.0: 20,
    3.0: 45,
    6.0: 65,
    10.0: 82,
    24.0: 100,
  },
};

/**
 * Build a CSS `linear-gradient(...)` value from a legend's color stops.
 *
 * If a variable has a defined stretch map, the gradient stops are mapped to
 * their dedicated visual percentages instead of linear physical spacing.
 */
export function buildLegendGradient(
  stops: readonly LegendStop[],
  variableCode?: string | null
): string {
  if (stops.length === 0) {
    return "linear-gradient(to right, transparent, transparent)";
  }

  const stretchMap = variableCode ? STRETCH_POSITION_PERCENT[variableCode] : undefined;
  const first = stops[0][0];
  const last = stops[stops.length - 1][0];
  const range = last - first;

  const parts = stops.map(([value, color]) => {
    let position: number;
    if (stretchMap && stretchMap[value] !== undefined) {
      position = stretchMap[value];
    } else {
      position = range === 0 ? 0 : ((value - first) / range) * 100;
    }
    return `${color} ${position}%`;
  });

  return `linear-gradient(to right, ${parts.join(", ")})`;
}

/**
 * Compute key tick marks and labels across the physical range of legend stops.
 */
export function getLegendTicks(
  stops: readonly LegendStop[],
  variableCode?: string | null,
  minDistancePercent = 14
): LegendTick[] {
  if (stops.length === 0) {
    return [];
  }
  if (stops.length === 1) {
    return [{ value: stops[0][0], label: `${stops[0][0]}`, positionPercent: 0 }];
  }

  const stretchMap = variableCode ? STRETCH_POSITION_PERCENT[variableCode] : undefined;
  const first = stops[0][0];
  const last = stops[stops.length - 1][0];
  const range = last - first;

  const toLabel = (val: number) =>
    Number.isInteger(val) || Math.abs(val - Math.round(val)) < 1e-4
      ? `${Math.round(val)}`
      : `${parseFloat(val.toFixed(1))}`;

  // If stretchMap is defined, all configured stops have explicitly designed non-overlapping positions
  if (stretchMap) {
    return stops.map(([val]) => ({
      value: val,
      label: toLabel(val),
      positionPercent: stretchMap[val] ?? (range === 0 ? 0 : ((val - first) / range) * 100),
    }));
  }

  // Always include the first and last stops
  const firstTick: LegendTick = {
    value: first,
    label: toLabel(first),
    positionPercent: 0,
  };
  const lastTick: LegendTick = {
    value: last,
    label: toLabel(last),
    positionPercent: 100,
  };

  if (range === 0 || stops.length === 2) {
    return [firstTick, lastTick];
  }

  // Calculate percentage positions for all intermediate stops
  const candidates = stops.slice(1, -1).map(([val]) => {
    const pos = ((val - first) / range) * 100;
    return {
      value: val,
      label: toLabel(val),
      positionPercent: pos,
    };
  });

  const accepted: LegendTick[] = [firstTick];
  for (const cand of candidates) {
    const distFromLastAccepted = cand.positionPercent - accepted[accepted.length - 1].positionPercent;
    const distToEnd = 100 - cand.positionPercent;
    if (distFromLastAccepted >= minDistancePercent && distToEnd >= minDistancePercent) {
      accepted.push(cand);
    }
  }
  accepted.push(lastTick);

  return accepted;
}
