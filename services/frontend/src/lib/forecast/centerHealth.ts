import type { ForecastAvailability, ModelAvailability } from "@/lib/api/types";

/**
 * Derive per-center and per-model feed health from the already-fetched
 * availability payload.
 *
 * The header's status badge is a read-only view of what the platform can
 * currently serve. Everything it needs is already in the availability
 * response the app fetches at load, at each 3-hour cadence boundary, and on
 * return to the foreground — so this module adds no request, no query, and no
 * polling of its own.
 *
 * Status vocabulary:
 *   ready   - the newest cycle is committed and at least one valid time is
 *             servable.
 *   syncing - the newest cycle is still being ingested (``processing`` or
 *             ``partial``): data is arriving, nothing is wrong.
 *   stale   - the newest cycle is more than {@link STALE_CYCLE_MULTIPLIER}
 *             cadence intervals behind the clock, i.e. the feed missed at
 *             least one full publication. Checked BEFORE ``ready``: a cycle
 *             that old is a stalled feed even if it was once marked ready.
 *   down    - no cycle information at all, so nothing can be claimed.
 */

/** How many cadence intervals behind the clock a cycle must be to read as stale. */
export const STALE_CYCLE_MULTIPLIER = 2;

/** Fallback cadence when a model predates the ``cycle_cadence_hours`` field. */
const DEFAULT_CADENCE_HOURS = 6;

export type ModelHealthStatus = "ready" | "syncing" | "stale" | "down";
export type CenterHealthStatus = "healthy" | "degraded" | "down";

export interface ModelHealth {
  id: string;
  name: string;
  isEnsemble: boolean;
  status: ModelHealthStatus;
  /** Newest cycle time held for this model, ISO 8601 UTC, or null. */
  latestCycle: string | null;
  /** True when the payload reports at least one servable valid time. */
  servable: boolean;
}

export interface CenterHealth {
  id: string;
  name: string;
  status: CenterHealthStatus;
  models: ModelHealth[];
  /** Models whose status is ``ready``. */
  readyCount: number;
  totalCount: number;
}

interface CycleSummary {
  latestCycle: string | null;
  latestStatus: string | null;
  anyServable: boolean;
}

/**
 * Reduce a model's variables to one cycle summary.
 *
 * Every variable repeats the same cycle set, so the newest initial time is
 * taken across all of them; `servable` is satisfied by any variable, because
 * the badge reports whether the feed is delivering at all, not whether a
 * particular variable is complete.
 */
function summarizeCycles(model: ModelAvailability): CycleSummary {
  let latestCycle: string | null = null;
  let latestStatus: string | null = null;
  let anyServable = false;

  for (const variable of model.variables) {
    for (const initial of variable.initial_times ?? []) {
      if (latestCycle === null || initial.value > latestCycle) {
        latestCycle = initial.value;
        latestStatus = initial.status ?? null;
      }
    }
    for (const valid of variable.valid_times ?? []) {
      if (valid.servable) {
        anyServable = true;
      }
    }
  }

  return { latestCycle, latestStatus, anyServable };
}

function classifyModel(model: ModelAvailability, nowMs: number): ModelHealth {
  const { latestCycle, latestStatus, anyServable } = summarizeCycles(model);
  const cadenceHours = model.cycle_cadence_hours ?? DEFAULT_CADENCE_HOURS;

  const base = {
    id: model.id,
    name: model.name,
    isEnsemble: model.is_ensemble,
    latestCycle,
    servable: anyServable,
  };

  if (latestCycle === null) {
    return { ...base, status: "down" };
  }

  const ageMs = nowMs - Date.parse(latestCycle);
  const staleAfterMs = cadenceHours * STALE_CYCLE_MULTIPLIER * 3_600_000;
  if (Number.isFinite(ageMs) && ageMs > staleAfterMs) {
    return { ...base, status: "stale" };
  }

  if (latestStatus === "ready" && anyServable) {
    return { ...base, status: "ready" };
  }
  if (latestStatus === "processing" || latestStatus === "partial") {
    return { ...base, status: "syncing" };
  }

  return { ...base, status: "down" };
}

function rollupCenter(models: ModelHealth[]): CenterHealthStatus {
  if (models.length === 0) return "down";
  if (models.every((model) => model.status === "down")) return "down";
  if (models.every((model) => model.status === "ready")) return "healthy";
  return "degraded";
}

/**
 * Group availability models by issuing center and classify each.
 *
 * Models without a `center_id` are grouped under a synthetic center so they
 * stay visible rather than disappearing from the badge.
 */
export function deriveCenterHealth(
  availability: ForecastAvailability | null,
  nowMs: number
): CenterHealth[] {
  if (availability === null) return [];

  const byCenter = new Map<string, { name: string; models: ModelHealth[] }>();

  for (const model of availability.models) {
    const id = model.center_id ?? "unknown";
    const entry = byCenter.get(id) ?? {
      name: model.center_name ?? id.toUpperCase(),
      models: [],
    };
    entry.models.push(classifyModel(model, nowMs));
    byCenter.set(id, entry);
  }

  return Array.from(byCenter.entries())
    .map(([id, entry]) => {
      const models = [...entry.models].sort((a, b) => a.id.localeCompare(b.id));
      return {
        id,
        name: entry.name,
        models,
        status: rollupCenter(models),
        readyCount: models.filter((model) => model.status === "ready").length,
        totalCount: models.length,
      };
    })
    .sort((a, b) => a.id.localeCompare(b.id));
}
