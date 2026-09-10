import type { EnsembleStatisticsData, ForecastEntry } from "@/lib/api/types";
import {
  FORECAST_ENTRY_METADATA_FIELDS,
  RAW_CATEGORICAL_PHASE_VARIABLES,
  isForecastDataVariable,
  isForecastEntryMetadataField,
  isRawCategoricalPhaseVariable,
} from "@/lib/api/types";
import {
  distributionSummary,
  distributionXDomain,
  ensembleStatisticsEntries,
  forecastLeadTimes,
  forecastVariableCodes,
  histogramBins,
  toEnsembleChartData,
  toEnsembleFanData,
  toEnsemblePhaseSupportData,
  toMemberDots,
  toMeteogramSeries,
  toPdfPoints,
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
    expect(FORECAST_ENTRY_METADATA_FIELDS.has("precipitation_type")).toBe(true);
    expect(FORECAST_ENTRY_METADATA_FIELDS.has("precipitation_transition")).toBe(true);
    expect(FORECAST_ENTRY_METADATA_FIELDS.has("precipitation_start_type")).toBe(true);
    expect(FORECAST_ENTRY_METADATA_FIELDS.has("precipitation_end_type")).toBe(true);
    expect(FORECAST_ENTRY_METADATA_FIELDS.has("precipitation_evidence")).toBe(true);
  });

  it("does not classify real forecast variables as metadata", () => {
    expect(FORECAST_ENTRY_METADATA_FIELDS.has("temperature_2m")).toBe(false);
    expect(FORECAST_ENTRY_METADATA_FIELDS.has("precipitation_rate")).toBe(false);
    expect(FORECAST_ENTRY_METADATA_FIELDS.has("precipitation_amount_3h")).toBe(false);
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

describe("RAW_CATEGORICAL_PHASE_VARIABLES", () => {
  it("contains crain, csnow, cfrzr, and cicep", () => {
    expect(RAW_CATEGORICAL_PHASE_VARIABLES.has("crain")).toBe(true);
    expect(RAW_CATEGORICAL_PHASE_VARIABLES.has("csnow")).toBe(true);
    expect(RAW_CATEGORICAL_PHASE_VARIABLES.has("cfrzr")).toBe(true);
    expect(RAW_CATEGORICAL_PHASE_VARIABLES.has("cicep")).toBe(true);
  });

  it("identifies raw categorical phase variables and excludes them from candidate plottable variables", () => {
    for (const code of ["crain", "csnow", "cfrzr", "cicep"]) {
      expect(isRawCategoricalPhaseVariable(code)).toBe(true);
      expect(isForecastDataVariable(code)).toBe(false);
    }
    expect(isRawCategoricalPhaseVariable("precipitation_amount_3h")).toBe(false);
    expect(isForecastDataVariable("precipitation_amount_3h")).toBe(true);
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

  it("extracts precipitation_amount_3h while excluding all transition metadata", () => {
    const precipEntries: ForecastEntry[] = [
      {
        lead_time_hours: 0,
        valid_time: "2026-07-21T00:00:00Z",
        cycle_time: "2026-07-21T00:00:00Z",
        precipitation_amount_3h: undefined,
        precipitation_type: "none",
        precipitation_transition: "none",
        precipitation_start_type: "none",
        precipitation_end_type: "none",
        precipitation_evidence: "exact",
      },
      {
        lead_time_hours: 3,
        valid_time: "2026-07-21T03:00:00Z",
        cycle_time: "2026-07-21T00:00:00Z",
        precipitation_amount_3h: 4.2,
        precipitation_type: "rain",
        precipitation_transition: "persistent_rain",
        precipitation_start_type: "rain",
        precipitation_end_type: "rain",
        precipitation_evidence: "exact",
      },
    ];

    expect(forecastVariableCodes(precipEntries)).toEqual(["precipitation_amount_3h"]);
  });

  it("excludes raw categorical phase variables (crain, csnow, cfrzr, cicep) from both deterministic and ensemble forecasts", () => {
    const deterministicEntries: ForecastEntry[] = [
      {
        lead_time_hours: 3,
        valid_time: "2026-07-21T03:00:00Z",
        temperature_2m: 15.2,
        precipitation_amount_3h: 2.5,
        crain: 1,
        csnow: 0,
        cfrzr: 0,
        cicep: 0,
      },
    ];

    const ensembleEntries: ForecastEntry[] = [
      {
        lead_time_hours: 3,
        valid_time: "2026-07-21T03:00:00Z",
        temperature_2m: 14.8,
        precipitation_amount_3h: 3.1,
        crain: 0.8,
        csnow: 0.2,
        cfrzr: 0,
        cicep: 0,
      },
    ];

    const detCodes = forecastVariableCodes(deterministicEntries);
    expect(detCodes.sort()).toEqual(["precipitation_amount_3h", "temperature_2m"].sort());
    expect(detCodes).not.toContain("crain");
    expect(detCodes).not.toContain("csnow");
    expect(detCodes).not.toContain("cfrzr");
    expect(detCodes).not.toContain("cicep");

    const ensCodes = forecastVariableCodes(ensembleEntries);
    expect(ensCodes.sort()).toEqual(["precipitation_amount_3h", "temperature_2m"].sort());
    expect(ensCodes).not.toContain("crain");
    expect(ensCodes).not.toContain("csnow");
    expect(ensCodes).not.toContain("cfrzr");
    expect(ensCodes).not.toContain("cicep");
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

  it("propagates valid_time from data or optional mapping", () => {
    const withValidTimes = new Map<number, EnsembleStatisticsData>([
      [
        0,
        {
          model: "gefs",
          lead_time_hours: 0,
          valid_time: "2026-09-10T00:00:00Z",
          member_count: 5,
          statistics: {
            mean: 10,
            median: 10,
            spread: 2,
            p10: 7,
            p25: 9,
            p50: 10,
            p75: 11,
            p90: 13,
          },
        },
      ],
    ]);
    const dataFromData = toEnsembleChartData(withValidTimes);
    expect(dataFromData[0].valid_time).toBe("2026-09-10T00:00:00Z");

    const map = new Map<number, string>([[0, "2026-09-10T00:00:00Z"]]);
    const dataFromMap = toEnsembleChartData(ensembleByLead, map);
    expect(dataFromMap[0].valid_time).toBe("2026-09-10T00:00:00Z");
  });
});

describe("toEnsembleFanData", () => {
  it("computes stacked band bases and heights from percentile statistics and preserves valid_time", () => {
    const map = new Map<number, string>([
      [0, "2026-09-10T00:00:00Z"],
      [6, "2026-09-10T06:00:00Z"],
    ]);
    const fan = toEnsembleFanData(ensembleByLead, map);
    expect(fan).toHaveLength(2);
    expect(fan[0]).toMatchObject({
      lead_time_hours: 0,
      valid_time: "2026-09-10T00:00:00Z",
      p10Base: 7,
      p90Height: 13 - 7,
      p25Base: 9,
      p75Height: 11 - 9,
      median: 10,
      mean: 10,
    });
    expect(fan[1].valid_time).toBe("2026-09-10T06:00:00Z");
    expect(fan[1].p90Height).toBeCloseTo(16 - 10, 6);
  });

  describe("NaN and missing percentile resilience (x-axis non-truncation invariant)", () => {
    function makeStats(val: number | null | typeof Number.NaN) {
      if (val === null || (typeof val === "number" && Number.isNaN(val))) {
        return {
          mean: val as any,
          median: val as any,
          spread: val as any,
          p10: val as any,
          p25: val as any,
          p50: val as any,
          p75: val as any,
          p90: val as any,
        };
      }
      return {
        mean: val,
        median: val,
        spread: 2,
        p10: val - 3,
        p25: val - 1,
        p50: val,
        p75: val + 1,
        p90: val + 3,
      };
    }

    function createSeries(values: Array<number | null | typeof Number.NaN>) {
      const map = new Map<number, EnsembleStatisticsData>();
      const validTimes = new Map<number, string>();
      values.forEach((v, idx) => {
        const lead = idx * 3;
        const vt = `2026-09-10T${String(lead).padStart(2, "0")}:00:00Z`;
        validTimes.set(lead, vt);
        map.set(lead, {
          model: "gefs",
          lead_time_hours: lead,
          valid_time: vt,
          member_count: 30,
          statistics: makeStats(v),
        });
      });
      return { map, validTimes };
    }

    it("handles pattern: finite -> finite -> NaN -> finite -> finite without dropping later times", () => {
      const { map, validTimes } = createSeries([10, 12, Number.NaN, 14, 16]);
      const fan = toEnsembleFanData(map, validTimes);

      expect(fan).toHaveLength(5);
      expect(fan.map((p) => p.lead_time_hours)).toEqual([0, 3, 6, 9, 12]);
      expect(fan.map((p) => p.valid_time)).toEqual([
        "2026-09-10T00:00:00Z",
        "2026-09-10T03:00:00Z",
        "2026-09-10T06:00:00Z",
        "2026-09-10T09:00:00Z",
        "2026-09-10T12:00:00Z",
      ]);

      // Point 0 & 1 finite
      expect(fan[0].median).toBe(10);
      expect(fan[1].median).toBe(12);

      // Point 2 (NaN) normalized to null, geometry omitted
      expect(fan[2].p10Base).toBeNull();
      expect(fan[2].p90Height).toBeNull();
      expect(fan[2].p25Base).toBeNull();
      expect(fan[2].p75Height).toBeNull();
      expect(fan[2].median).toBeNull();
      expect(fan[2].mean).toBeNull();

      // Later points 3 & 4 remain visible and finite
      expect(fan[3].median).toBe(14);
      expect(fan[3].p10Base).toBe(11);
      expect(fan[3].p90Height).toBe(6);
      expect(fan[4].median).toBe(16);
    });

    it("handles pattern: NaN -> finite -> finite (missing at start)", () => {
      const { map, validTimes } = createSeries([Number.NaN, 20, 22]);
      const fan = toEnsembleFanData(map, validTimes);

      expect(fan).toHaveLength(3);
      expect(fan[0].valid_time).toBe("2026-09-10T00:00:00Z");
      expect(fan[0].median).toBeNull();
      expect(fan[1].median).toBe(20);
      expect(fan[2].median).toBe(22);
    });

    it("handles pattern: finite -> NaN -> finite (missing in middle)", () => {
      const { map, validTimes } = createSeries([30, Number.NaN, 35]);
      const fan = toEnsembleFanData(map, validTimes);

      expect(fan).toHaveLength(3);
      expect(fan[0].median).toBe(30);
      expect(fan[1].median).toBeNull();
      expect(fan[2].median).toBe(35);
    });

    it("handles pattern: finite -> finite -> NaN (missing at end)", () => {
      const { map, validTimes } = createSeries([40, 42, Number.NaN]);
      const fan = toEnsembleFanData(map, validTimes);

      expect(fan).toHaveLength(3);
      expect(fan[0].median).toBe(40);
      expect(fan[1].median).toBe(42);
      expect(fan[2].median).toBeNull();
      expect(fan[2].valid_time).toBe("2026-09-10T06:00:00Z");
    });

    it("handles multiple separated NaNs across the series", () => {
      const { map, validTimes } = createSeries([
        Number.NaN,
        50,
        Number.NaN,
        52,
        Number.NaN,
        54,
        Number.NaN,
      ]);
      const fan = toEnsembleFanData(map, validTimes);

      expect(fan).toHaveLength(7);
      expect(fan[0].median).toBeNull();
      expect(fan[1].median).toBe(50);
      expect(fan[2].median).toBeNull();
      expect(fan[3].median).toBe(52);
      expect(fan[4].median).toBeNull();
      expect(fan[5].median).toBe(54);
      expect(fan[6].median).toBeNull();

      // Verify all valid_times are preserved in order
      expect(fan.map((p) => p.lead_time_hours)).toEqual([0, 3, 6, 9, 12, 15, 18]);
    });

    it("preserves leads from validTimesByLead even when missing from ensembleByLead", () => {
      const ensembleMap = new Map<number, EnsembleStatisticsData>([
        [
          6,
          {
            model: "gefs",
            lead_time_hours: 6,
            valid_time: "2026-09-10T06:00:00Z",
            member_count: 30,
            statistics: makeStats(25),
          },
        ],
      ]);

      const validTimesMap = new Map<number, string>([
        [0, "2026-09-10T00:00:00Z"],
        [3, "2026-09-10T03:00:00Z"],
        [6, "2026-09-10T06:00:00Z"],
        [9, "2026-09-10T09:00:00Z"],
      ]);

      const fan = toEnsembleFanData(ensembleMap, validTimesMap);
      expect(fan).toHaveLength(4);
      expect(fan.map((p) => p.lead_time_hours)).toEqual([0, 3, 6, 9]);
      expect(fan[0].median).toBeNull();
      expect(fan[1].median).toBeNull();
      expect(fan[2].median).toBe(25);
      expect(fan[3].median).toBeNull();
    });
  });
});

describe("toEnsemblePhaseSupportData", () => {
  const byLeadMulti: Map<number, EnsembleStatisticsData> = new Map([
    [
      0,
      {
        model: "gefs",
        lead_time_hours: 0,
        valid_time: "2026-09-10T00:00:00Z",
        member_count: 30,
        valid_member_count: 30,
        statistics: {
          mean: null,
          median: null,
          spread: null,
          p10: null,
          p25: null,
          p50: null,
          p75: null,
          p90: null,
        },
        // Lead 0 is analysis time, precipitation accumulation is undefined / no phase
        phase_support: null,
      },
    ],
    [
      3,
      {
        model: "gefs",
        lead_time_hours: 3,
        valid_time: "2026-09-10T03:00:00Z",
        member_count: 30,
        valid_member_count: 30,
        statistics: {
          mean: null,
          median: null,
          spread: null,
          p10: null,
          p25: null,
          p50: null,
          p75: null,
          p90: null,
        },
        phase_support: {
          dry: 0.1,
          rain: 0.6,
          snow: 0.2,
          freezing_rain: 0.05,
          ice_pellets: 0.03,
          unknown: 0.02,
        },
      },
    ],
    [
      6,
      {
        model: "gefs",
        lead_time_hours: 6,
        valid_time: "2026-09-10T06:00:00Z",
        member_count: 30,
        valid_member_count: 30,
        statistics: {
          mean: null,
          median: null,
          spread: null,
          p10: null,
          p25: null,
          p50: null,
          p75: null,
          p90: null,
        },
        phase_support: {
          dry: 0.2,
          rain: 0.1,
          snow: 0.7,
          freezing_rain: 0.0,
          ice_pellets: 0.0,
          unknown: 0.0,
        },
      },
    ],
    [
      9,
      {
        model: "gefs",
        lead_time_hours: 9,
        valid_time: "2026-09-10T09:00:00Z",
        member_count: 30,
        valid_member_count: 30,
        statistics: {
          mean: null,
          median: null,
          spread: null,
          p10: null,
          p25: null,
          p50: null,
          p75: null,
          p90: null,
        },
        phase_support: {
          dry: 0.8,
          rain: 0.0,
          snow: 0.15,
          freezing_rain: 0.0,
          ice_pellets: 0.0,
          unknown: 0.05,
        },
      },
    ],
  ]);

  const validTimes = new Map<number, string>([
    [0, "2026-09-10T00:00:00Z"],
    [3, "2026-09-10T03:00:00Z"],
    [6, "2026-09-10T06:00:00Z"],
    [9, "2026-09-10T09:00:00Z"],
  ]);

  it("computes phase support independently for multiple valid times across the forecast series", () => {
    const series = toEnsemblePhaseSupportData(byLeadMulti, validTimes);

    expect(series).toHaveLength(4);
    expect(series.map((p) => p.lead_time_hours)).toEqual([0, 3, 6, 9]);
    expect(series.map((p) => p.valid_time)).toEqual([
      "2026-09-10T00:00:00Z",
      "2026-09-10T03:00:00Z",
      "2026-09-10T06:00:00Z",
      "2026-09-10T09:00:00Z",
    ]);

    // Lead 3 has high rain support
    expect(series[1].has_data).toBe(true);
    expect(series[1].rain).toBeCloseTo(60, 2);
    expect(series[1].snow).toBeCloseTo(20, 2);

    // Lead 6 transitions to snow dominance
    expect(series[2].has_data).toBe(true);
    expect(series[2].snow).toBeCloseTo(70, 2);
    expect(series[2].rain).toBeCloseTo(10, 2);

    // Lead 9 dries out
    expect(series[3].has_data).toBe(true);
    expect(series[3].dry).toBeCloseTo(80, 2);
  });

  it("totals approximately 100% for each valid time with phase data", () => {
    const series = toEnsemblePhaseSupportData(byLeadMulti, validTimes);

    for (const point of series) {
      if (point.has_data) {
        const total =
          (point.dry ?? 0) +
          (point.rain ?? 0) +
          (point.snow ?? 0) +
          (point.freezing_rain ?? 0) +
          (point.ice_pellets ?? 0) +
          (point.unknown ?? 0);
        expect(total).toBeCloseTo(100, 1);
      }
    }
  });

  it("gaps missing/invalid times without dropping subsequent valid times", () => {
    const series = toEnsemblePhaseSupportData(byLeadMulti, validTimes);

    // Lead 0 had null phase_support: must have has_data: false and null metrics
    expect(series[0].has_data).toBe(false);
    expect(series[0].rain).toBeNull();
    expect(series[0].dry).toBeNull();
    expect(series[0].snow).toBeNull();
    expect(series[0].valid_time).toBe("2026-09-10T00:00:00Z");

    // Later leads 3, 6, 9 remain intact and fully populated
    expect(series[1].has_data).toBe(true);
    expect(series[2].has_data).toBe(true);
    expect(series[3].has_data).toBe(true);
  });

  it("formats localized time labels with display timezone", () => {
    const series = toEnsemblePhaseSupportData(byLeadMulti, validTimes, "America/Denver");
    // 2026-09-10T00:00:00Z in Denver MDT is Sep 9, 18:00
    expect(series[0].label).toBe("Sep 9, 18:00");
    // 2026-09-10T06:00:00Z in Denver MDT is Sep 10, 00:00
    expect(series[2].label).toBe("Sep 10, 00:00");
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

  describe("toPdfPoints", () => {
    it("converts valid PDF payload to x/density coordinate objects", () => {
      const pdf = {
        x: [10.0, 15.0, 20.0],
        density: [0.01, 0.15, 0.02],
      };
      expect(toPdfPoints(pdf)).toEqual([
        { x: 10.0, density: 0.01 },
        { x: 15.0, density: 0.15 },
        { x: 20.0, density: 0.02 },
      ]);
    });

    it("handles null, undefined, or empty PDF gracefully", () => {
      expect(toPdfPoints(null)).toEqual([]);
      expect(toPdfPoints(undefined)).toEqual([]);
      expect(toPdfPoints({ x: [], density: [] })).toEqual([]);
    });

    it("filters non-finite numbers", () => {
      const pdf = {
        x: [10.0, Number.NaN, 20.0],
        density: [0.01, 0.15, Number.POSITIVE_INFINITY],
      };
      expect(toPdfPoints(pdf)).toEqual([{ x: 10.0, density: 0.01 }]);
    });
  });

  describe("distributionXDomain", () => {
    const summary = {
      count: 5,
      min: 15.0,
      max: 25.0,
      mean: 20.0,
      median: 20.0,
      stdDev: 3.5,
    };

    it("spans PDF extrema when PDF is present", () => {
      const pdf = {
        x: [10.0, 15.0, 20.0, 25.0, 30.0],
        density: [0.001, 0.05, 0.2, 0.05, 0.001],
      };
      expect(distributionXDomain(summary, pdf)).toEqual([10.0, 30.0]);
    });

    it("falls back to sample min/max when PDF is absent", () => {
      expect(distributionXDomain(summary, null)).toEqual([15.0, 25.0]);
      expect(distributionXDomain(summary, undefined)).toEqual([15.0, 25.0]);
    });

    it("handles empty summary when fallback needed", () => {
      const emptySummary = {
        count: 0,
        min: Number.NaN,
        max: Number.NaN,
        mean: Number.NaN,
        median: Number.NaN,
        stdDev: Number.NaN,
      };
      expect(distributionXDomain(emptySummary, null)).toEqual([0, 1]);
    });

    it("expands identical min/max to prevent zero-width scale when PDF is absent", () => {
      const constantSummary = {
        count: 5,
        min: 20.0,
        max: 20.0,
        mean: 20.0,
        median: 20.0,
        stdDev: 0.0,
      };
      expect(distributionXDomain(constantSummary, null)).toEqual([19.0, 21.0]);
    });
  });
});
