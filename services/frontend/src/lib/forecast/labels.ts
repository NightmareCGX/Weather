import type { VariableResource } from "@/lib/api/types";

/** Format an ISO 8601 UTC time as a compact "2026-08-13 00Z" label. */
export function formatInitialTimeLabel(iso: string): string {
  const parsed = new Date(iso);
  if (Number.isNaN(parsed.getTime())) {
    return iso;
  }
  const year = parsed.getUTCFullYear();
  const month = String(parsed.getUTCMonth() + 1).padStart(2, "0");
  const day = String(parsed.getUTCDate()).padStart(2, "0");
  const hour = String(parsed.getUTCHours()).padStart(2, "0");
  return `${year}-${month}-${day} ${hour}Z`;
}

/**
 * Variable display metadata and value formatting.
 *
 * The `/v1/variables` catalog is the authoritative source for variable names
 * and units. This module merges it with a fallback map so the dashboard can
 * label the default variables even before the catalog request resolves, and
 * so unknown future variables still render a stable label.
 */

export interface VariableMeta {
  name: string;
  unit: string;
}

/** Fallback metadata for the documented default variables (API.md examples). */
export const FALLBACK_VARIABLE_META: Record<string, VariableMeta> = {
  temperature_2m: { name: "Temperature (2 m)", unit: "°C" },
  precipitation_rate: { name: "Precipitation Rate", unit: "mm/h" },
  precipitation_amount_3h: { name: "3-Hour Precipitation", unit: "mm" },
  relative_humidity_2m: { name: "Relative Humidity", unit: "%" },
  wind_gust: { name: "Wind Gust", unit: "km/h" },
  visibility: { name: "Visibility", unit: "m" },
  snow_depth: { name: "Snow Depth", unit: "m" },
  wind_10m: { name: "Wind (10 m)", unit: "km/h" },
  cloud_cover_3h: { name: "Cloud Cover", unit: "%" },
  cloud_ceiling: { name: "Cloud Ceiling", unit: "m" },
};

/**
 * Build a variable-code → metadata map from the catalog, falling back to
 * {@link FALLBACK_VARIABLE_META} for known defaults when the catalog is not
 * loaded or lacks an entry.
 */
export function buildVariableMeta(
  variables: readonly VariableResource[] | null | undefined
): Record<string, VariableMeta> {
  const meta: Record<string, VariableMeta> = {};
  for (const variable of variables ?? []) {
    meta[variable.id] = { name: variable.name, unit: variable.unit };
  }
  for (const [code, fallback] of Object.entries(FALLBACK_VARIABLE_META)) {
    if (meta[code] === undefined) {
      meta[code] = fallback;
    }
  }
  return meta;
}

/**
 * Format a numeric value with its unit, dropping trailing zeros.
 *
 * @example
 *   formatValue(15.0, "°C")       // "15 °C"
 *   formatValue(0.424, "", {style:"percent"}) // "42%"
 */
export function formatValue(
  value: number,
  unit: string,
  options: Intl.NumberFormatOptions = {},
  maximumFractionDigits = 2
): string {
  const formatted = new Intl.NumberFormat(undefined, {
    maximumFractionDigits,
    ...options,
  }).format(value);
  if (unit === "") {
    return formatted;
  }
  return `${formatted} ${unit}`;
}

/** Format a fraction in [0, 1] as a percentage string (e.g. `0.42` → `"42%"`). */
export function formatPercent(value: number): string {
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 0, style: "percent" }).format(
    value
  );
}

/** Format a probability with its 95% confidence interval. */
export function formatProbabilityRange(probability: number, ci: readonly [number, number]): string {
  return `${formatPercent(probability)} (95% CI ${formatPercent(ci[0])}–${formatPercent(ci[1])})`;
}

/**
 * Format cloud ceiling height with unlimited sentinel handling.
 *
 * @example
 *   formatCloudCeiling(1200, "m") // "1,200 m"
 *   formatCloudCeiling(null, "m", true) // "Unlimited"
 *   formatCloudCeiling(20000, "m") // "Unlimited"
 */
export function formatCloudCeiling(
  value: number | null | undefined,
  unit: string = "m",
  isUnlimited: boolean = false
): string {
  if (isUnlimited || value === null || value === undefined || value >= 19990) {
    return "Unlimited";
  }
  return `${Math.round(value).toLocaleString()} ${unit}`;
}

/**
 * Format meteorological wind direction with optional cardinal label.
 *
 * @example
 *   formatWindDirection(225.0, "SW") // "225° SW"
 *   formatWindDirection(null, "CALM") // "Calm"
 */
export function formatWindDirection(
  directionDeg: number | null | undefined,
  cardinal?: string | null
): string {
  if (directionDeg == null || cardinal === "CALM") {
    return "Calm";
  }
  const rounded = Math.round(directionDeg);
  if (cardinal) {
    return `${rounded}° ${cardinal}`;
  }
  return `${rounded}°`;
}
