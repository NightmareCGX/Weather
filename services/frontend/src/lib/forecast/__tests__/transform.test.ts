import type { EnsembleStatisticsData, ForecastEntry } from "@/lib/api/types";
import {
  FORECAST_ENTRY_METADATA_FIELDS,
  isForecastDataVariable,
  isForecastEntryMetadataField,
} from "@/lib/api/types";
import {
  distributionSummary,
  ensembleStatisticsEntries,
  forecastLeadTimes,
  forecastVariableCodes,
  histogramBins,
  toEnsembleChartData,
  toEnsembleFanData,
  toMemberDots,
  toMeteogramSeries,
} from "@/lib/forecast/transform";

const entries: ForecastEntry[] = [
  {
    lead_time_hours: 0,
    valid_time: "2026-07-21T00:00:00Z",
    temperature_2m: 10,
    precipitation_rate: 0,
  },
  {
    lead_time_hours: 6,
    valid_time: "2026-07-21T06:00:00Z",
    temperature_2m: 13,
    precipitation_rate: 3,
  },
  {
    lead_time_hours: 12,
    valid_time: "2026-07-21T12:00:00Z",
    temperature_2m: 16,
    precipitation_rate: 6,
  },
];

/**
 * A forecast entry carrying every structural/metadata field the backend
 * attaches to a point-forecast series (API.md section 2.1: lead offset, valid
 * time, and the additive cross-cycle provenance `cycle_time`).
 */
const entriesWithEveryMetadataField: ForecastEntry[] = [
  {
    lead_time_hours: 0,
    valid_time: "2026-07-21T00:00:00Z",
    cycle_time: "2026-07-21T00:00:00Z",
    temperature_2m: 10,
    precipitation_rate: 0,
  },
  {
    lead_time_hours: 6,
    valid_time: "2026-07-21T06:00:00Z",
    cycle_time: "2026-07-21T00:00:00Z",
    temperature_2m: 13,
    precipitation_rate: 3,
  },
];

describe("FORECAST_ENTRY_METADATA_FIELDS", () => {
  it("classifies every structural forecast-entry field as metadata", () => {
    expect(FORECAST_ENTRY_METADATA_FIELDS.has("lead_time_hours")).toBe(true);
    expect(FORECAST_ENTRY_METADATA_FIELDS.has("valid_time")).toBe(true);
    // The additive cross-cycle provenance field must never render as a
    // variable or be sent as /v1/ensembles?variable=….
    expect(FORECAST_ENTRY_METADATA_FIELDS.has("cycle_time")).toBe(true);
  });

  it("does not classify real forecast variables as metadata", () => {
    expect(FORECAST_ENTRY_METADATA_FIELDS.has("temperature_2m")).toBe(false);
    expect(FORECAST_ENTRY_METADATA_FIELDS.has("precipitation_rate")).toBe(false);
  });

  it("exposes the guards for every field in the set and for variables", () => {
    const metadataFields = Array.from(FORECAST_ENTRY_METADATA_FIELDS);
    for (const field of metadataFields) {
      expect(isForecastEntryMetadataField(field)).toBe(true);
      expect(isForecastDataVariable(field)).toBe(false);
    }
    expect(isForecastEntryMetadataField("temperature_2m")).toBe(false);
    expect(isForecastDataVariable("temperature_2m")).toBe(true);
  });
});

describe("forecastVariableCodes", () => {
  it("returns only the variable keys, excluding structural keys", () => {
    expect(forecastVariableCodes(entries)).toEqual(["temperature_2m", "precipitation_rate"]);
  });

  it("returns an empty array for structural-only entries", () => {
    expect(
      forecastVariableCodes([{ lead_time_hours: 0, valid_time: "2026-07-21T00:00:00Z" }])
    ).toEqual([]);
  });

  it("excludes every metadata/coordinate field, not just cycle_time", () => {
    // The regression contract: metadata/coordinate fields are excluded from
    // renderable forecast variables, so they can never reach chart titles or
    // /v1/ensembles?variable=…. Assert the exclusion over the full documented
    // structural set, not an individual-cycle_time special case.
    const metadataFields = Array.from(FORECAST_ENTRY_METADATA_FIELDS);
    for (const field of metadataFields) {
      expect(forecastVariableCodes(entriesWithEveryMetadataField)).not.toContain(field);
    }
    expect(forecastVariableCodes(entriesWithEveryMetadataField)).toEqual([
      "temperature_2m",
      "precipitation_rate",
    ]);
  });
});

describe("toMeteogramSeries", () => {
  it("maps entries to points preserving lead and valid_time", () => {
    const series = toMeteogramSeries(entries, "temperature_2m");
    expect(series).toEqual([
      { lead_time_hours: 0, valid_time: "2026-07-21T00:00:00Z", value: 10 },
      { lead_time_hours: 6, valid_time: "2026-07-21T06:00:00Z", value: 13 },
      { lead_time_hours: 12, valid_time: "2026-07-21T12:00:00Z", value: 16 },
    ]);
  });

  it("marks missing and non-finite values as null", () => {
    const series = toMeteogramSeries(
      [
        { lead_time_hours: 0, valid_time: "2026-07-21T00:00:00Z", temperature_2m: "n/a" as never },
        { lead_time_hours: 6, valid_time: "2026-07-21T06:00:00Z", temperature_2m: 13 },
      ],
      "temperature_2m"
    );
    expect(series[0].value).toBeNull();
    expect(series[1].value).toBe(13);
  });
});

describe("forecastLeadTimes", () => {
  it("returns unique leads ascending", () => {
    expect(forecastLeadTimes(entries)).toEqual([0, 6, 12]);
  });
});

const ensembleByLead = new Map<number, EnsembleStatisticsData>([
  [
    0,
    {
      model: "gefs",
      lead_time_hours: 0,
      member_count: 5,
      statistics: { mean: 10, median: 10, spread: 2, p10: 7, p25: 9, p50: 10, p75: 11, p90: 13 },
    },
  ],
  [
    6,
    {
      model: "gefs",
      lead_time_hours: 6,
      member_count: 5,
      statistics: { mean: 13, median: 13, spread: 2, p10: 10, p25: 12, p50: 13, p75: 14, p90: 16 },
    },
  ],
]);

describe("toEnsembleChartData", () => {
  it("builds fan-chart points ordered by lead", () => {
    const data = toEnsembleChartData(ensembleByLead);
    expect(data.map((point) => point.lead_time_hours)).toEqual([0, 6]);
    expect(data[1]).toMatchObject({ lead_time_hours: 6, mean: 13, median: 13, p90: 16 });
  });
});

describe("toEnsembleFanData", () => {
  it("computes stacked band bases and heights from percentile statistics", () => {
    const fan = toEnsembleFanData(ensembleByLead);
    expect(fan).toHaveLength(2);
    expect(fan[0]).toMatchObject({
      lead_time_hours: 0,
      p10Base: 7,
      p90Height: 13 - 7,
      p25Base: 9,
      p75Height: 11 - 9,
      median: 10,
      mean: 10,
    });
    expect(fan[1].p90Height).toBeCloseTo(16 - 10, 6);
  });
});

describe("histogramBins", () => {
  it("bins members deterministically with a count per bin", () => {
    const bins = histogramBins([15.5, 17.5, 19.5, 21.5, 23.5]);
    expect(bins.reduce((sum, bin) => sum + bin.count, 0)).toBe(5);
    expect(bins.every((bin) => bin.count >= 0)).toBe(true);
  });

  it("returns no bins for an empty member list", () => {
    expect(histogramBins([])).toEqual([]);
  });

  it("returns a single bin when all members are equal", () => {
    expect(histogramBins([7, 7, 7])).toEqual([{ start: 7, end: 7, count: 3, mid: 7 }]);
  });
});

describe("toMemberDots", () => {
  it("preserves index and value, dropping non-finite members", () => {
    const dots = toMemberDots([15.5, 17.5, Number.NaN, 21.5]);
    expect(dots).toEqual([
      { index: 0, value: 15.5 },
      { index: 1, value: 17.5 },
      { index: 3, value: 21.5 },
    ]);
  });
});

describe("distributionSummary", () => {
  it("computes count/min/max/mean/median/stdDev from raw members", () => {
    const summary = distributionSummary([10, 20, 30, 40, 50]);
    expect(summary.count).toBe(5);
    expect(summary.min).toBe(10);
    expect(summary.max).toBe(50);
    expect(summary.mean).toBe(30);
    expect(summary.median).toBe(30);
    expect(summary.stdDev).toBeCloseTo(Math.sqrt(200), 6);
  });

  it("handles empty member lists with NaN stats", () => {
    const summary = distributionSummary([]);
    expect(summary.count).toBe(0);
    expect(Number.isNaN(summary.mean)).toBe(true);
  });
});

describe("ensembleStatisticsEntries", () => {
  it("returns the documented statistics in order", () => {
    const entries_ = ensembleStatisticsEntries({
      mean: 1,
      median: 2,
      spread: 3,
      p10: 4,
      p25: 5,
      p50: 6,
      p75: 7,
      p90: 8,
    });
    expect(entries_.map(([key]) => key)).toEqual([
      "mean",
      "median",
      "spread",
      "p10",
      "p25",
      "p50",
      "p75",
      "p90",
    ]);
  });
});
