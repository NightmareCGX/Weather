import type { PhysicalPhase, PrecipitationTransition, PrecipitationType } from "@/lib/api/types";
import { formatValue } from "./labels";

/**
 * Authoritative frontend metadata, formatting, and color tokens for precipitation
 * amount, phase classification, transitions, and GEFS ensemble phase support.
 *
 * Phase classification is owned by the backend/domain layer and rendered here.
 * The frontend never derives or re-infers precipitation phase from temperature.
 */

export interface PhaseMeta {
  code: string;
  name: string;
  color: string;
  badgeBg: string;
  badgeText: string;
  ariaLabel: string;
}

/**
 * Visual design system tokens for physical and interval precipitation phases.
 */
export const PRECIPITATION_PHASE_TOKENS: Record<string, PhaseMeta> = {
  rain: {
    code: "rain",
    name: "Rain",
    color: "#0284c7", // sky-600
    badgeBg: "#e0f2fe",
    badgeText: "#0369a1",
    ariaLabel: "Rain",
  },
  snow: {
    code: "snow",
    name: "Snow",
    color: "#6366f1", // indigo-500
    badgeBg: "#e0e7ff",
    badgeText: "#4338ca",
    ariaLabel: "Snow",
  },
  freezing_rain: {
    code: "freezing_rain",
    name: "Freezing Rain",
    color: "#e11d48", // rose-600
    badgeBg: "#ffe4e6",
    badgeText: "#be123c",
    ariaLabel: "Freezing Rain (Icing Hazard)",
  },
  ice_pellets: {
    code: "ice_pellets",
    name: "Ice Pellets",
    color: "#0d9488", // teal-600
    badgeBg: "#ccfbf1",
    badgeText: "#0f766e",
    ariaLabel: "Ice Pellets",
  },
  mixed: {
    code: "mixed",
    name: "Mixed",
    color: "#8b5cf6", // violet-500
    badgeBg: "#ede9fe",
    badgeText: "#6d28d9",
    ariaLabel: "Mixed Precipitation",
  },
  dry: {
    code: "dry",
    name: "Dry",
    color: "#94a3b8", // slate-400
    badgeBg: "#f1f5f9",
    badgeText: "#475569",
    ariaLabel: "Dry (No Precipitation)",
  },
  none: {
    code: "none",
    name: "Dry",
    color: "#94a3b8",
    badgeBg: "#f1f5f9",
    badgeText: "#475569",
    ariaLabel: "Dry (No Precipitation)",
  },
  unknown: {
    code: "unknown",
    name: "Unclassified",
    color: "#64748b", // slate-500
    badgeBg: "#f8fafc",
    badgeText: "#475569",
    ariaLabel: "Unclassified Precipitation",
  },
};

/**
 * Ordered list of physical phase buckets used for 100% stacked GEFS phase support.
 *
 * Invariant: GEFS phase support has NO 'mixed' bucket (members distribute support
 * across physical phases). All 6 buckets sum to 100% including Unknown.
 */
export const GEFS_PHYSICAL_PHASES: readonly PhysicalPhase[] = [
  "dry",
  "rain",
  "snow",
  "freezing_rain",
  "ice_pellets",
  "unknown",
] as const;

/**
 * User-facing labels for GEFS physical phase support segments.
 */
export const GEFS_PHASE_LABELS: Record<PhysicalPhase, string> = {
  dry: "Dry",
  rain: "Rain",
  snow: "Snow",
  freezing_rain: "Freezing Rain",
  ice_pellets: "Ice Pellets",
  unknown: "Unknown",
};

/**
 * Map of transition codes to user-facing descriptions.
 */
const TRANSITION_DISPLAY_NAMES: Record<string, string> = {
  none: "Dry",
  persistent_rain: "Rain",
  persistent_snow: "Snow",
  persistent_freezing_rain: "Freezing Rain",
  persistent_ice_pellets: "Ice Pellets",
  rain_to_snow: "Rain → Snow",
  snow_to_rain: "Snow → Rain",
  rain_to_freezing_rain: "Rain → Freezing Rain",
  freezing_rain_to_rain: "Freezing Rain → Rain",
  snow_to_freezing_rain: "Snow → Freezing Rain",
  freezing_rain_to_snow: "Freezing Rain → Snow",
  snow_to_ice_pellets: "Snow → Ice Pellets",
  ice_pellets_to_snow: "Ice Pellets → Snow",
  rain_to_ice_pellets: "Rain → Ice Pellets",
  ice_pellets_to_rain: "Ice Pellets → Rain",
  dry_to_rain: "Dry → Rain",
  dry_to_snow: "Dry → Snow",
  dry_to_freezing_rain: "Dry → Freezing Rain",
  dry_to_ice_pellets: "Dry → Ice Pellets",
  wet_to_dry: "Wet → Dry",
  mixed_transition: "Mixed",
  unknown: "Unclassified",
};

/**
 * Retrieve metadata and styling tokens for a precipitation phase.
 */
export function getPrecipitationPhaseMeta(phase: string | null | undefined): PhaseMeta {
  if (!phase || !(phase in PRECIPITATION_PHASE_TOKENS)) {
    return PRECIPITATION_PHASE_TOKENS.unknown;
  }
  return PRECIPITATION_PHASE_TOKENS[phase];
}

/**
 * Format a discrete transition code into its human-readable title.
 */
export function formatTransitionName(transition: string | null | undefined): string {
  if (!transition) return "Dry";
  return TRANSITION_DISPLAY_NAMES[transition] ?? "Unclassified";
}

export interface PointForecastPrecipEntry {
  precipitation_amount_3h?: number | null;
  precipitation_type?: PrecipitationType | string;
  precipitation_transition?: PrecipitationTransition | string;
  precipitation_start_type?: PrecipitationType | string;
  precipitation_end_type?: PrecipitationType | string;
  precipitation_evidence?: string;
  lead_time_hours?: number;
}

/**
 * Format the user-facing phase/transition description for a point forecast interval.
 *
 * Examples:
 * - Persistent rain: "Rain"
 * - Rain to snow transition: "Rain → Snow"
 * - Multi-phase ambiguous: "Mixed"
 * - Dry: "Dry"
 * - Unclassified: "Unclassified"
 */
export function getPointForecastPhaseLabel(
  entry: PointForecastPrecipEntry | null | undefined
): string {
  if (!entry) return "Dry";

  const amount = entry.precipitation_amount_3h;
  // Lead 0 (null amount) has no accumulation interval
  if (amount === null || amount === undefined) {
    return "—";
  }

  // Dry interval
  if (amount <= 0.05 || entry.precipitation_type === "none") {
    return "Dry";
  }

  const transition = entry.precipitation_transition;
  const pType = entry.precipitation_type;

  // Discrete two-phase transitions
  if (transition && transition !== "none") {
    if (
      transition === "rain_to_snow" ||
      transition === "snow_to_rain" ||
      transition === "rain_to_freezing_rain" ||
      transition === "freezing_rain_to_rain" ||
      transition === "snow_to_freezing_rain" ||
      transition === "freezing_rain_to_snow" ||
      transition === "snow_to_ice_pellets" ||
      transition === "ice_pellets_to_snow" ||
      transition === "rain_to_ice_pellets" ||
      transition === "ice_pellets_to_rain"
    ) {
      return TRANSITION_DISPLAY_NAMES[transition];
    }

    if (transition === "persistent_rain" || transition === "dry_to_rain") return "Rain";
    if (transition === "persistent_snow" || transition === "dry_to_snow") return "Snow";
    if (transition === "persistent_freezing_rain" || transition === "dry_to_freezing_rain") {
      return "Freezing Rain";
    }
    if (transition === "persistent_ice_pellets" || transition === "dry_to_ice_pellets") {
      return "Ice Pellets";
    }
    if (transition === "mixed_transition") return "Mixed";
    if (transition === "unknown") return "Unclassified";
  }

  // Fallback to precipitation_type
  if (pType === "rain") return "Rain";
  if (pType === "snow") return "Snow";
  if (pType === "freezing_rain") return "Freezing Rain";
  if (pType === "ice_pellets") return "Ice Pellets";
  if (pType === "mixed") return "Mixed";
  if (pType === "unknown") return "Unclassified";

  return "Dry";
}

/**
 * Format a point forecast precipitation entry as:
 * `<amount> <unit> · <phase>`
 *
 * Examples:
 * - `4.2 mm · Rain`
 * - `5.1 mm · Rain → Snow`
 * - `0.0 mm · Dry`
 * - `—` at f000
 */
export function formatPointPrecipitationDisplay(
  entry: PointForecastPrecipEntry | null | undefined,
  unit: string = "mm"
): string {
  if (!entry) return "—";
  const amount = entry.precipitation_amount_3h;
  if (amount === null || amount === undefined) {
    return "—";
  }

  const phaseLabel = getPointForecastPhaseLabel(entry);
  const formattedAmount = formatValue(amount, unit);
  return `${formattedAmount} · ${phaseLabel}`;
}

export interface TransitionPhases {
  start: string;
  end: string;
}

/**
 * Parse start and end phases from a transition identifier.
 */
export function getTransitionPhases(
  transition: string | null | undefined
): TransitionPhases | null {
  if (!transition) return null;
  const parts = transition.split("_to_");
  if (parts.length === 2) {
    return { start: parts[0], end: parts[1] };
  }
  return null;
}

/**
 * Get the color representation for a point forecast interval.
 * Returns either a single hex color or null (for gradients).
 */
export function getBarColorForEntry(entry: PointForecastPrecipEntry | null | undefined): string {
  if (
    !entry ||
    entry.precipitation_amount_3h === null ||
    entry.precipitation_amount_3h === undefined
  ) {
    return "transparent";
  }

  if (entry.precipitation_amount_3h <= 0.05 || entry.precipitation_type === "none") {
    return PRECIPITATION_PHASE_TOKENS.dry.color;
  }

  const label = getPointForecastPhaseLabel(entry);
  if (label === "Rain") return PRECIPITATION_PHASE_TOKENS.rain.color;
  if (label === "Snow") return PRECIPITATION_PHASE_TOKENS.snow.color;
  if (label === "Freezing Rain") return PRECIPITATION_PHASE_TOKENS.freezing_rain.color;
  if (label === "Ice Pellets") return PRECIPITATION_PHASE_TOKENS.ice_pellets.color;
  if (label === "Mixed") return PRECIPITATION_PHASE_TOKENS.mixed.color;
  if (label === "Unclassified") return PRECIPITATION_PHASE_TOKENS.unknown.color;

  // Transitions: default to mixed or transition start color
  const trPhases = getTransitionPhases(entry.precipitation_transition);
  if (trPhases && trPhases.start in PRECIPITATION_PHASE_TOKENS) {
    return PRECIPITATION_PHASE_TOKENS[trPhases.start].color;
  }

  return PRECIPITATION_PHASE_TOKENS.rain.color;
}
