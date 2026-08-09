import type { EnsembleStatistics, EnsembleStatisticsData, ForecastEntry } from "@/lib/api/types";

/**
 * Pure data transformations from API envelopes to chart-ready structures.
 *
 * Everything here is deterministic and side-effect free so it can be unit
 * tested without a DOM. No weather calculations live in the API layer; these
 * are formatting/normalization helpers only (ENGINEERING_CONTRACT section 2).
 */

/** A single plottable point of a meteogram series. */
export interface MeteogramPoint {
  lead_time_hours: number;
  /** ISO 8601 UTC valid time (``cycle_time + lead_time_hours``). */
  valid_time: string;
  /** The variable value, or `null` when missing/non-finite. */
  value: number | null;
}

/**
 * Extract the plottable variable codes from a forecast series, excluding the
 * structural keys `lead_time_hours` and `valid_time`.
 */
export function forecastVariableCodes(forecasts: ForecastEntry[]): string[] {
  const codes = new Set<string>();
  for (const entry of forecasts) {
    for (const key of Object.keys(entry)) {
      if (key !== "lead_time_hours" && key !== "valid_time") {
        codes.add(key);
      }
    }
  }
  return Array.from(codes);
}

/**
 * Build the meteogram series for one variable code from `/v1/points` entries.
 *
 * Entries that lack the code or carry a non-finite value contribute a `null`
 * value so charts can skip or annotate them; the lead/time metadata is always
 * preserved.
 */
export function toMeteogramSeries(
  forecasts: ForecastEntry[],
  variableCode: string
): MeteogramPoint[] {
  return forecasts.map((entry) => {
    const raw = entry[variableCode];
    const value = typeof raw === "number" && Number.isFinite(raw) ? raw : null;
    return {
      lead_time_hours: entry.lead_time_hours,
      valid_time: entry.valid_time,
      value,
    };
  });
}

/** The full set of lead times present in a point forecast, ascending. */
export function forecastLeadTimes(forecasts: ForecastEntry[]): number[] {
  const leads = forecasts.map((entry) => entry.lead_time_hours);
  return Array.from(new Set(leads)).sort((a, b) => a - b);
}

/** A single point of the ensemble-over-time fan chart. */
export interface EnsembleChartPoint {
  lead_time_hours: number;
  mean: number;
  median: number;
  spread: number;
  p10: number;
  p25: number;
  p50: number;
  p75: number;
  p90: number;
}

/**
 * Build the ensemble-over-time fan chart data from one `/v1/ensembles`
 * response per lead time.
 *
 * `ensembleByLead` maps a lead time to its statistics payload; `leads` defines
 * the ordering (typically the point forecast's lead set). Leads without data
 * are omitted so the chart simply has no point there.
 */
export function toEnsembleChartData(
  ensembleByLead: ReadonlyMap<number, EnsembleStatisticsData>
): EnsembleChartPoint[] {
  const leads = Array.from(ensembleByLead.keys()).sort((a, b) => a - b);
  return leads.map((lead) => {
    const data = ensembleByLead.get(lead);
    if (data === undefined) {
      throw new Error(`Missing ensemble data for lead ${lead}.`);
    }
    return { lead_time_hours: lead, ...data.statistics };
  });
}

/**
 * A single point of the ensemble fan chart, prepared for a stacked-area
 * rendering of the P10–P90 and P25–P75 bands.
 *
 * Recharts renders a percentile band as a stacked pair of areas: a transparent
 * bottom stack equal to the band's lower edge, and a second stack equal to the
 * band's height. The resulting union spans exactly `[lower, upper]`.
 */
export interface EnsembleFanPoint {
  lead_time_hours: number;
  /** Lower edge of the P10–P90 band (stack base). */
  p10Base: number;
  /** Height of the P10–P90 band. */
  p90Height: number;
  /** Lower edge of the P25–P75 central band (stack base). */
  p25Base: number;
  /** Height of the P25–P75 central band. */
  p75Height: number;
  median: number;
  mean: number;
  spread: number;
}

/** Build the stacked fan-band points for one `/v1/ensembles` response per lead. */
export function toEnsembleFanData(
  ensembleByLead: ReadonlyMap<number, EnsembleStatisticsData>
): EnsembleFanPoint[] {
  return toEnsembleChartData(ensembleByLead).map((point) => ({
    lead_time_hours: point.lead_time_hours,
    p10Base: point.p10,
    p90Height: point.p90 - point.p10,
    p25Base: point.p25,
    p75Height: point.p75 - point.p25,
    median: point.median,
    mean: point.mean,
    spread: point.spread,
  }));
}

/** A single histogram bin. */
export interface HistogramBin {
  /** Lower edge (inclusive). */
  start: number;
  /** Upper edge (exclusive; the last bin is inclusive). */
  end: number;
  /** Number of members in the bin. */
  count: number;
  /** Midpoint used as the bin's category label. */
  mid: number;
}

/**
 * Bin raw ensemble member values into a histogram.
 *
 * Uses Sturges' rule (`ceil(log2(n)) + 1`) so the bin count grows gently with
 * ensemble size and remains deterministic. Degenerate inputs are handled
 * explicitly: an empty array yields no bins, and identical members yield a
 * single bin containing all of them.
 */
export function histogramBins(members: number[]): HistogramBin[] {
  if (members.length === 0) {
    return [];
  }
  const finite = members.filter((value) => Number.isFinite(value));
  if (finite.length === 0) {
    return [];
  }
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  if (min === max) {
    return [{ start: min, end: max, count: finite.length, mid: min }];
  }
  const binCount = Math.max(1, Math.ceil(Math.log2(finite.length)) + 1);
  const width = (max - min) / binCount;
  const bins: HistogramBin[] = [];
  for (let i = 0; i < binCount; i += 1) {
    const start = min + i * width;
    const end = start + width;
    bins.push({ start, end, count: 0, mid: (start + end) / 2 });
  }
  for (const value of finite) {
    let index = Math.min(binCount - 1, Math.floor((value - min) / width));
    if (index < 0) {
      index = 0;
    }
    bins[index].count += 1;
  }
  return bins;
}

/** A member-dot plot point (one dot per raw member value). */
export interface MemberDot {
  /** Member index in dataset `member`-coordinate order. */
  index: number;
  value: number;
}

/** Flatten raw member values into discrete dots for a rug/dot plot. */
export function toMemberDots(members: number[]): MemberDot[] {
  return members
    .map((value, index) => ({ index, value }))
    .filter((dot) => Number.isFinite(dot.value));
}

/** A compact summary of the raw member distribution. */
export interface DistributionSummary {
  count: number;
  min: number;
  max: number;
  mean: number;
  median: number;
  stdDev: number;
}

function sorted(values: number[]): number[] {
  return [...values].sort((a, b) => a - b);
}

function percentile(sortedValues: number[], q: number): number {
  if (sortedValues.length === 0) {
    return Number.NaN;
  }
  const position = (q / 100) * (sortedValues.length - 1);
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) {
    return sortedValues[lower];
  }
  const weight = position - lower;
  return sortedValues[lower] * (1 - weight) + sortedValues[upper] * weight;
}

/**
 * Summarize the raw member distribution. This uses only the genuine
 * member-level values — never reconstructed from aggregate statistics.
 */
export function distributionSummary(members: number[]): DistributionSummary {
  const finite = members.filter((value) => Number.isFinite(value));
  const sortedValues = sorted(finite);
  const mean =
    finite.length === 0
      ? Number.NaN
      : finite.reduce((sum, value) => sum + value, 0) / finite.length;
  const variance =
    finite.length === 0
      ? Number.NaN
      : finite.reduce((sum, value) => sum + (value - mean) ** 2, 0) / finite.length;
  return {
    count: finite.length,
    min: finite.length === 0 ? Number.NaN : sortedValues[0],
    max: finite.length === 0 ? Number.NaN : sortedValues[sortedValues.length - 1],
    mean,
    median: percentile(sortedValues, 50),
    stdDev: Math.sqrt(variance),
  };
}

/**
 * Format the ensemble statistics object for a compact readout row, ordered as
 * the API documents them.
 */
export function ensembleStatisticsEntries(
  statistics: EnsembleStatistics
): Array<[key: string, value: number]> {
  return [
    ["mean", statistics.mean],
    ["median", statistics.median],
    ["spread", statistics.spread],
    ["p10", statistics.p10],
    ["p25", statistics.p25],
    ["p50", statistics.p50],
    ["p75", statistics.p75],
    ["p90", statistics.p90],
  ];
}
