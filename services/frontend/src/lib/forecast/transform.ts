import type {
  EnsemblePDF,
  EnsembleStatistics,
  EnsembleStatisticsData,
  ForecastEntry,
} from "@/lib/api/types";
import { isForecastDataVariable } from "@/lib/api/types";
import { formatDayHourInTimeZone } from "@/lib/forecast/time";

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
 * structural keys `lead_time_hours`/`valid_time`/`cycle_time`.
 *
 * The exclusion set is the single authoritative
 * {@link FORECAST_ENTRY_METADATA_FIELDS} boundary: only the backend's declared
 * structural fields are skipped, so a future additive metadata field is
 * excluded here automatically instead of surfacing as a bogus chart (or an
 * invalid `/v1/ensembles` request).
 */
export function forecastVariableCodes(forecasts: ForecastEntry[]): string[] {
  const codes = new Set<string>();
  for (const entry of forecasts) {
    for (const key of Object.keys(entry)) {
      if (isForecastDataVariable(key)) {
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
  valid_time?: string;
  mean: number | null;
  median: number | null;
  spread: number | null;
  p10: number | null;
  p25: number | null;
  p50: number | null;
  p75: number | null;
  p90: number | null;
}

function toFiniteOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/**
 * Build the ensemble-over-time fan chart data from one `/v1/ensembles`
 * response per lead time.
 *
 * `ensembleByLead` maps a lead time to its statistics payload; `leads` defines
 * the ordering (derived from the union of ensemble data and validTimesByLead).
 * Leads without data or carrying non-finite metrics have their metrics normalized
 * to `null` so the chart x-domain is preserved and later valid times remain visible.
 */
export function toEnsembleChartData(
  ensembleByLead: ReadonlyMap<number, EnsembleStatisticsData>,
  validTimesByLead?: ReadonlyMap<number, string>
): EnsembleChartPoint[] {
  const leadSet = new Set<number>(Array.from(ensembleByLead.keys()));
  if (validTimesByLead) {
    for (const lead of Array.from(validTimesByLead.keys())) {
      leadSet.add(lead);
    }
  }
  const leads = Array.from(leadSet).sort((a, b) => a - b);
  return leads.map((lead) => {
    const data = ensembleByLead.get(lead);
    const validTime = data?.valid_time ?? validTimesByLead?.get(lead);
    const stats = data?.statistics;
    return {
      lead_time_hours: lead,
      valid_time: validTime,
      mean: toFiniteOrNull(stats?.mean),
      median: toFiniteOrNull(stats?.median),
      spread: toFiniteOrNull(stats?.spread),
      p10: toFiniteOrNull(stats?.p10),
      p25: toFiniteOrNull(stats?.p25),
      p50: toFiniteOrNull(stats?.p50),
      p75: toFiniteOrNull(stats?.p75),
      p90: toFiniteOrNull(stats?.p90),
    };
  });
}

/**
 * A single point of the ensemble fan chart, prepared for a stacked-area
 * rendering of the P10–P90 and P25–P75 bands.
 *
 * Recharts renders a percentile band as a stacked pair of areas: a transparent
 * bottom stack equal to the band's lower edge, and a second stack equal to the
 * band's height. The resulting union spans exactly `[lower, upper]`.
 * Non-finite or missing statistics are normalized to `null` so points are
 * safely gapped without truncating the time axis.
 */
export interface EnsembleFanPoint {
  lead_time_hours: number;
  valid_time?: string;
  /** Lower edge of the P10–P90 band (stack base), or null if missing/non-finite. */
  p10Base: number | null;
  /** Height of the P10–P90 band, or null if missing/non-finite. */
  p90Height: number | null;
  /** Lower edge of the P25–P75 central band (stack base), or null if missing/non-finite. */
  p25Base: number | null;
  /** Height of the P25–P75 central band, or null if missing/non-finite. */
  p75Height: number | null;
  median: number | null;
  mean: number | null;
  spread: number | null;
}

/** Build the stacked fan-band points for one `/v1/ensembles` response per lead. */
export function toEnsembleFanData(
  ensembleByLead: ReadonlyMap<number, EnsembleStatisticsData>,
  validTimesByLead?: ReadonlyMap<number, string>
): EnsembleFanPoint[] {
  return toEnsembleChartData(ensembleByLead, validTimesByLead).map((point) => {
    const p10 = point.p10;
    const p90 = point.p90;
    const p25 = point.p25;
    const p75 = point.p75;
    return {
      lead_time_hours: point.lead_time_hours,
      valid_time: point.valid_time,
      p10Base: p10 !== null && p90 !== null ? p10 : null,
      p90Height: p10 !== null && p90 !== null ? p90 - p10 : null,
      p25Base: p25 !== null && p75 !== null ? p25 : null,
      p75Height: p25 !== null && p75 !== null ? p75 - p25 : null,
      median: point.median,
      mean: point.mean,
      spread: point.spread,
    };
  });
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

/** A single evaluation point of the continuous PDF line series. */
export interface PdfPoint {
  x: number;
  density: number;
}

/**
 * Convert an API EnsemblePDF payload into chart-ready coordinate points.
 *
 * Missing or empty payloads return an empty array so the chart series can be
 * safely skipped.
 */
export function toPdfPoints(pdf?: EnsemblePDF | null): PdfPoint[] {
  if (!pdf || !Array.isArray(pdf.x) || !Array.isArray(pdf.density)) {
    return [];
  }
  const len = Math.min(pdf.x.length, pdf.density.length);
  const points: PdfPoint[] = [];
  for (let i = 0; i < len; i += 1) {
    const x = pdf.x[i];
    const density = pdf.density[i];
    if (
      typeof x === "number" &&
      Number.isFinite(x) &&
      typeof density === "number" &&
      Number.isFinite(density)
    ) {
      points.push({ x, density });
    }
  }
  return points;
}

/**
 * Compute the shared numeric X-domain for the member distribution visualization.
 *
 * When a PDF is present, the domain spans the full canonical evaluation window
 * `[min(pdf.x), max(pdf.x)]` so continuous tails are not clipped. When the PDF
 * is absent, it defaults to the observed sample extrema `[min, max]`.
 */
export function distributionXDomain(
  summary: DistributionSummary,
  pdf?: EnsemblePDF | null
): [number, number] {
  if (pdf && Array.isArray(pdf.x) && pdf.x.length > 0) {
    const validX = pdf.x.filter((v) => typeof v === "number" && Number.isFinite(v));
    if (validX.length > 0) {
      return [validX[0], validX[validX.length - 1]];
    }
  }
  let min = Number.isFinite(summary.min) ? summary.min : 0;
  let max = Number.isFinite(summary.max) ? summary.max : 1;
  if (min === max) {
    min -= 1;
    max += 1;
  }
  return [min, max];
}

/**
 * Format the ensemble statistics object for a compact readout row, ordered as
 * the API documents them.
 */
export function ensembleStatisticsEntries(
  statistics: EnsembleStatistics
): Array<[key: string, value: number | null]> {
  return [
    ["mean", statistics.mean ?? null],
    ["median", statistics.median ?? null],
    ["spread", statistics.spread ?? null],
    ["p10", statistics.p10 ?? null],
    ["p25", statistics.p25 ?? null],
    ["p50", statistics.p50 ?? null],
    ["p75", statistics.p75 ?? null],
    ["p90", statistics.p90 ?? null],
  ];
}

/** A single valid-time evaluation point of the ensemble phase support series. */
export interface EnsemblePhaseSupportPoint {
  lead_time_hours: number;
  valid_time: string;
  label: string;
  dry: number | null;
  rain: number | null;
  snow: number | null;
  freezing_rain: number | null;
  ice_pellets: number | null;
  unknown: number | null;
  valid_member_count: number | null;
  member_count: number;
  has_data: boolean;
  transition_frequency?: Record<string, number> | null;
}

/**
 * Build time-varying ensemble phase support series data across forecast valid times.
 *
 * For each valid time:
 * - Reads normalized phase support fractions across physical phases (dry, rain, snow, freezing_rain, ice_pellets, unknown).
 * - Normalizes to 100% using available members.
 * - Missing or unusable phase data at a timestamp is gapped (metrics = null) without truncating subsequent valid times.
 * - Retains canonical UTC valid_time for time-axis alignment with active forecast range.
 */
export function toEnsemblePhaseSupportData(
  ensembleByLead: ReadonlyMap<number, EnsembleStatisticsData>,
  validTimesByLead?: ReadonlyMap<number, string>,
  timezone?: string | null
): EnsemblePhaseSupportPoint[] {
  const leadSet = new Set<number>(Array.from(ensembleByLead.keys()));
  if (validTimesByLead) {
    for (const lead of Array.from(validTimesByLead.keys())) {
      leadSet.add(lead);
    }
  }
  const leads = Array.from(leadSet).sort((a, b) => a - b);
  return leads.map((lead) => {
    const data = ensembleByLead.get(lead);
    const validTime = data?.valid_time ?? validTimesByLead?.get(lead) ?? "";

    const phaseSupport = data?.phase_support;
    const hasRawData =
      phaseSupport != null &&
      typeof phaseSupport === "object" &&
      Object.values(phaseSupport).some((v) => typeof v === "number" && Number.isFinite(v) && v > 0);

    let dry: number | null = null;
    let rain: number | null = null;
    let snow: number | null = null;
    let freezing_rain: number | null = null;
    let ice_pellets: number | null = null;
    let unknown: number | null = null;
    let has_data = false;

    if (hasRawData && phaseSupport) {
      const rDry = Number.isFinite(phaseSupport["dry"]) ? phaseSupport["dry"] : 0;
      const rRain = Number.isFinite(phaseSupport["rain"]) ? phaseSupport["rain"] : 0;
      const rSnow = Number.isFinite(phaseSupport["snow"]) ? phaseSupport["snow"] : 0;
      const rFrzr = Number.isFinite(phaseSupport["freezing_rain"])
        ? phaseSupport["freezing_rain"]
        : 0;
      const rIcep = Number.isFinite(phaseSupport["ice_pellets"]) ? phaseSupport["ice_pellets"] : 0;
      const rUnk = Number.isFinite(phaseSupport["unknown"]) ? phaseSupport["unknown"] : 0;

      const sum = rDry + rRain + rSnow + rFrzr + rIcep + rUnk;
      if (sum > 0) {
        dry = (rDry / sum) * 100;
        rain = (rRain / sum) * 100;
        snow = (rSnow / sum) * 100;
        freezing_rain = (rFrzr / sum) * 100;
        ice_pellets = (rIcep / sum) * 100;
        unknown = (rUnk / sum) * 100;
        has_data = true;
      }
    }

    return {
      lead_time_hours: lead,
      valid_time: validTime,
      label: validTime ? formatDayHourInTimeZone(validTime, timezone) : "",
      dry,
      rain,
      snow,
      freezing_rain,
      ice_pellets,
      unknown,
      valid_member_count: data?.valid_member_count ?? null,
      member_count: data?.member_count ?? 30,
      has_data,
      transition_frequency: data?.transition_frequency ?? null,
    };
  });
}
