import {
  formatPointPrecipitationDisplay,
  formatTransitionName,
  getBarColorForEntry,
  getMixedPhaseConstituents,
  getPointForecastPhaseLabel,
  getPrecipitationPhaseMeta,
  getTransitionPhases,
  GEFS_PHYSICAL_PHASES,
  GEFS_PHASE_LABELS,
  PRECIPITATION_PHASE_TOKENS,
} from "../precipitation";
import { toEnsemblePhaseSupportData } from "../transform";
import type { EnsembleStatisticsData } from "@/lib/api/types";

describe("precipitation formatting and phase metadata", () => {
  describe("getPrecipitationPhaseMeta", () => {
    it("returns correct metadata for all known physical and interval phases", () => {
      expect(getPrecipitationPhaseMeta("rain")).toEqual(PRECIPITATION_PHASE_TOKENS.rain);
      expect(getPrecipitationPhaseMeta("snow")).toEqual(PRECIPITATION_PHASE_TOKENS.snow);
      expect(getPrecipitationPhaseMeta("freezing_rain")).toEqual(
        PRECIPITATION_PHASE_TOKENS.freezing_rain
      );
      expect(getPrecipitationPhaseMeta("ice_pellets")).toEqual(
        PRECIPITATION_PHASE_TOKENS.ice_pellets
      );
      expect(getPrecipitationPhaseMeta("mixed")).toEqual(PRECIPITATION_PHASE_TOKENS.mixed);
      expect(getPrecipitationPhaseMeta("dry")).toEqual(PRECIPITATION_PHASE_TOKENS.dry);
      expect(getPrecipitationPhaseMeta("none")).toEqual(PRECIPITATION_PHASE_TOKENS.none);
      expect(getPrecipitationPhaseMeta("unknown")).toEqual(PRECIPITATION_PHASE_TOKENS.unknown);
    });

    it("falls back to unknown metadata for undefined or invalid phase", () => {
      expect(getPrecipitationPhaseMeta(null)).toEqual(PRECIPITATION_PHASE_TOKENS.unknown);
      expect(getPrecipitationPhaseMeta(undefined)).toEqual(PRECIPITATION_PHASE_TOKENS.unknown);
      expect(getPrecipitationPhaseMeta("unrecognized_phase")).toEqual(
        PRECIPITATION_PHASE_TOKENS.unknown
      );
    });
  });

  describe("formatTransitionName", () => {
    it("formats persistent phases cleanly without 'persistent' prefix", () => {
      expect(formatTransitionName("persistent_rain")).toBe("Rain");
      expect(formatTransitionName("persistent_snow")).toBe("Snow");
      expect(formatTransitionName("persistent_freezing_rain")).toBe("Freezing Rain");
      expect(formatTransitionName("persistent_ice_pellets")).toBe("Ice Pellets");
    });

    it("formats two-phase transitions with arrows", () => {
      expect(formatTransitionName("rain_to_snow")).toBe("Rain → Snow");
      expect(formatTransitionName("snow_to_rain")).toBe("Snow → Rain");
      expect(formatTransitionName("rain_to_freezing_rain")).toBe("Rain → Freezing Rain");
      expect(formatTransitionName("freezing_rain_to_rain")).toBe("Freezing Rain → Rain");
      expect(formatTransitionName("snow_to_freezing_rain")).toBe("Snow → Freezing Rain");
      expect(formatTransitionName("freezing_rain_to_snow")).toBe("Freezing Rain → Snow");
      expect(formatTransitionName("snow_to_ice_pellets")).toBe("Snow → Ice Pellets");
      expect(formatTransitionName("ice_pellets_to_snow")).toBe("Ice Pellets → Snow");
    });

    it("formats onset transitions and special transitions", () => {
      expect(formatTransitionName("dry_to_rain")).toBe("Dry → Rain");
      expect(formatTransitionName("dry_to_snow")).toBe("Dry → Snow");
      expect(formatTransitionName("wet_to_dry")).toBe("Wet → Dry");
      expect(formatTransitionName("mixed_transition")).toBe("Mixed");
      expect(formatTransitionName("unknown")).toBe("Unclassified");
      expect(formatTransitionName("none")).toBe("Dry");
      expect(formatTransitionName(null)).toBe("Dry");
    });
  });

  describe("getPointForecastPhaseLabel", () => {
    it("returns '—' at f000 (null amount)", () => {
      expect(
        getPointForecastPhaseLabel({
          precipitation_amount_3h: null,
          precipitation_type: "none",
          precipitation_transition: "none",
        })
      ).toBe("—");

      expect(
        getPointForecastPhaseLabel({
          precipitation_amount_3h: undefined,
        })
      ).toBe("—");
    });

    it("returns 'Dry' for zero or trace precipitation", () => {
      expect(
        getPointForecastPhaseLabel({
          precipitation_amount_3h: 0.0,
          precipitation_type: "none",
          precipitation_transition: "none",
        })
      ).toBe("Dry");

      expect(
        getPointForecastPhaseLabel({
          precipitation_amount_3h: 0.02,
          precipitation_type: "none",
          precipitation_transition: "none",
        })
      ).toBe("Dry");
    });

    it("returns persistent single-phase labels", () => {
      expect(
        getPointForecastPhaseLabel({
          precipitation_amount_3h: 4.2,
          precipitation_type: "rain",
          precipitation_transition: "persistent_rain",
        })
      ).toBe("Rain");

      expect(
        getPointForecastPhaseLabel({
          precipitation_amount_3h: 3.8,
          precipitation_type: "snow",
          precipitation_transition: "persistent_snow",
        })
      ).toBe("Snow");

      expect(
        getPointForecastPhaseLabel({
          precipitation_amount_3h: 2.0,
          precipitation_type: "freezing_rain",
          precipitation_transition: "persistent_freezing_rain",
        })
      ).toBe("Freezing Rain");

      expect(
        getPointForecastPhaseLabel({
          precipitation_amount_3h: 1.5,
          precipitation_type: "ice_pellets",
          precipitation_transition: "persistent_ice_pellets",
        })
      ).toBe("Ice Pellets");
    });

    it("returns two-phase transition labels", () => {
      expect(
        getPointForecastPhaseLabel({
          precipitation_amount_3h: 5.1,
          precipitation_type: "mixed",
          precipitation_transition: "rain_to_snow",
        })
      ).toBe("Rain → Snow");

      expect(
        getPointForecastPhaseLabel({
          precipitation_amount_3h: 2.3,
          precipitation_type: "mixed",
          precipitation_transition: "snow_to_rain",
        })
      ).toBe("Snow → Rain");

      expect(
        getPointForecastPhaseLabel({
          precipitation_amount_3h: 1.8,
          precipitation_type: "mixed",
          precipitation_transition: "rain_to_freezing_rain",
        })
      ).toBe("Rain → Freezing Rain");
    });

    it("returns 'Mixed' for complex multi-phase transitions when constituents are unspecified", () => {
      expect(
        getPointForecastPhaseLabel({
          precipitation_amount_3h: 1.1,
          precipitation_type: "mixed",
          precipitation_transition: "mixed_transition",
        })
      ).toBe("Mixed");
    });

    it("classifies Mixed with explicit constituent list derived from categorical flags", () => {
      // Rain + Snow
      expect(
        getPointForecastPhaseLabel({
          precipitation_amount_3h: 0.31,
          precipitation_type: "mixed",
          crain: 1,
          csnow: 1,
          cfrzr: 0,
          cicep: 0,
        })
      ).toBe("Mixed (Rain + Snow)");

      // Rain + Freezing Rain
      expect(
        getPointForecastPhaseLabel({
          precipitation_amount_3h: 0.18,
          precipitation_type: "mixed",
          crain: 1,
          csnow: 0,
          cfrzr: 1,
          cicep: 0,
        })
      ).toBe("Mixed (Rain + Freezing Rain)");

      // Three constituents in deterministic order: Rain + Snow + Ice Pellets
      expect(
        getPointForecastPhaseLabel({
          precipitation_amount_3h: 0.45,
          precipitation_type: "mixed",
          crain: 1,
          csnow: 1,
          cfrzr: 0,
          cicep: 1,
        })
      ).toBe("Mixed (Rain + Snow + Ice Pellets)");
    });

    it("maintains deterministic constituent ordering regardless of flag evaluation sequence", () => {
      const constituents = getMixedPhaseConstituents({
        cicep: 1,
        crain: 1,
        csnow: 1,
        cfrzr: 1,
      });
      // Established physical order: Rain, Snow, Freezing Rain, Ice Pellets
      expect(constituents).toEqual(["Rain", "Snow", "Freezing Rain", "Ice Pellets"]);
    });

    it("returns 'Unclassified' for wet intervals lacking microphysical diagnostics", () => {
      expect(
        getPointForecastPhaseLabel({
          precipitation_amount_3h: 0.8,
          precipitation_type: "unknown",
          precipitation_transition: "unknown",
        })
      ).toBe("Unclassified");
    });
  });

  describe("formatPointPrecipitationDisplay", () => {
    it("formats full point forecast line combining amount and phase", () => {
      expect(
        formatPointPrecipitationDisplay(
          {
            precipitation_amount_3h: 4.2,
            precipitation_type: "rain",
            precipitation_transition: "persistent_rain",
          },
          "mm"
        )
      ).toBe("4.2 mm · Rain");

      expect(
        formatPointPrecipitationDisplay(
          {
            precipitation_amount_3h: 5.1,
            precipitation_type: "mixed",
            precipitation_transition: "rain_to_snow",
          },
          "mm"
        )
      ).toBe("5.1 mm · Rain → Snow");

      expect(
        formatPointPrecipitationDisplay(
          {
            precipitation_amount_3h: 0.31,
            precipitation_type: "mixed",
            crain: 1,
            csnow: 1,
          },
          "mm"
        )
      ).toBe("0.31 mm · Mixed (Rain + Snow)");

      expect(
        formatPointPrecipitationDisplay(
          {
            precipitation_amount_3h: 0.0,
            precipitation_type: "none",
            precipitation_transition: "none",
          },
          "mm"
        )
      ).toBe("0 mm · Dry");
    });

    it("formats GFS deterministic single phase and mixed tooltips with exact constituents", () => {
      // 1. Single phase: Rain
      expect(
        formatPointPrecipitationDisplay(
          {
            precipitation_amount_3h: 0.31,
            precipitation_type: "rain",
            precipitation_transition: "persistent_rain",
            crain: 1,
            csnow: 0,
            cfrzr: 0,
            cicep: 0,
          },
          "mm"
        )
      ).toBe("0.31 mm · Rain");

      // 2. Deterministic Mixed: Rain + Snow (symptom reproduction case)
      expect(
        formatPointPrecipitationDisplay(
          {
            precipitation_amount_3h: 1.83,
            precipitation_type: "mixed",
            precipitation_transition: "mixed_transition",
            crain: 1,
            csnow: 1,
            cfrzr: 0,
            cicep: 0,
          },
          "mm"
        )
      ).toBe("1.83 mm · Mixed (Rain + Snow)");

      // 3. Three-way Mixed: Snow + Freezing Rain + Ice Pellets in deterministic order
      expect(
        formatPointPrecipitationDisplay(
          {
            precipitation_amount_3h: 0.72,
            precipitation_type: "mixed",
            precipitation_transition: "mixed_transition",
            crain: 0,
            csnow: 1,
            cfrzr: 1,
            cicep: 1,
          },
          "mm"
        )
      ).toBe("0.72 mm · Mixed (Snow + Freezing Rain + Ice Pellets)");

      // 4. Four-way Mixed: Rain + Snow + Freezing Rain + Ice Pellets in deterministic order
      expect(
        formatPointPrecipitationDisplay(
          {
            precipitation_amount_3h: 2.0,
            precipitation_type: "mixed",
            precipitation_transition: "mixed_transition",
            crain: 1,
            csnow: 1,
            cfrzr: 1,
            cicep: 1,
          },
          "mm"
        )
      ).toBe("2 mm · Mixed (Rain + Snow + Freezing Rain + Ice Pellets)");
    });

    it("formats GEFS ensemble-mean single-phase and mixed tooltips from mean categorical representation", () => {
      // Single-phase ensemble mean
      expect(
        formatPointPrecipitationDisplay(
          {
            precipitation_amount_3h: 2.1,
            precipitation_type: "rain",
            precipitation_transition: "persistent_rain",
            crain: 0.85,
            csnow: 0.1,
            cfrzr: 0,
            cicep: 0,
          },
          "mm"
        )
      ).toBe("2.1 mm · Rain");

      // Mixed ensemble mean with fractional flags from official geavg mean shards
      expect(
        formatPointPrecipitationDisplay(
          {
            precipitation_amount_3h: 1.83,
            precipitation_type: "mixed",
            precipitation_transition: "mixed_transition",
            crain: 0.65,
            csnow: 0.55,
            cfrzr: 0.05,
            cicep: 0.0,
          },
          "mm"
        )
      ).toBe("1.83 mm · Mixed (Rain + Snow)");
    });

    it("preserves strict separation between GEFS Hourly Forecast phase and Ensemble Phase Support", () => {
      // Hourly Forecast entry from ensemble-mean path
      const gefsMeanEntry = {
        precipitation_amount_3h: 1.83,
        precipitation_type: "mixed",
        precipitation_transition: "mixed_transition",
        crain: 0.65,
        csnow: 0.55,
        cfrzr: 0.0,
        cicep: 0.0,
      };

      // Ensemble statistics data across 30 members for the same valid time
      const ensembleStats: EnsembleStatisticsData = {
        model: "gefs",
        lead_time_hours: 6,
        member_count: 30,
        valid_member_count: 30,
        valid_time: "2026-09-10T06:00:00Z",
        statistics: {
          mean: 1.83,
          median: 1.75,
          spread: 0.45,
          p10: 1.2,
          p25: 1.5,
          p50: 1.75,
          p75: 2.1,
          p90: 2.4,
        },
        phase_support: {
          dry: 0.2,
          rain: 0.5,
          snow: 0.3,
          freezing_rain: 0.0,
          ice_pellets: 0.0,
          unknown: 0.0,
        },
      };

      const byLead = new Map<number, EnsembleStatisticsData>([[6, ensembleStats]]);
      const phaseSupportData = toEnsemblePhaseSupportData(byLead);

      // Hourly Forecast tooltip remains derived ONLY from the ensemble-mean forecast entry
      const hourlyTooltip = formatPointPrecipitationDisplay(gefsMeanEntry, "mm");
      expect(hourlyTooltip).toBe("1.83 mm · Mixed (Rain + Snow)");

      // Ensemble Phase Support data remains derived ONLY from across-member statistics
      expect(phaseSupportData).toHaveLength(1);
      const point = phaseSupportData[0];
      expect(point.dry).toBe(20);
      expect(point.rain).toBe(50);
      expect(point.snow).toBe(30);
      expect(point.freezing_rain).toBe(0);
      expect(point.ice_pellets).toBe(0);
      expect(point.unknown).toBe(0);

      // Verify that member percentages do NOT overwrite Hourly Forecast constituents
      expect(hourlyTooltip).not.toContain("Dry");
      expect(hourlyTooltip).toContain("Rain + Snow");
    });

    it("preserves directed two-phase transition labels even when multiple flags are active", () => {
      expect(
        formatPointPrecipitationDisplay(
          {
            precipitation_amount_3h: 4.5,
            precipitation_type: "mixed",
            precipitation_transition: "rain_to_snow",
            crain: 1.0,
            csnow: 1.0,
          },
          "mm"
        )
      ).toBe("4.5 mm · Rain → Snow");
    });

    it("formats imperial units correctly", () => {
      expect(
        formatPointPrecipitationDisplay(
          {
            precipitation_amount_3h: 0.25,
            precipitation_type: "snow",
            precipitation_transition: "persistent_snow",
          },
          "in"
        )
      ).toBe("0.25 in · Snow");
    });

    it("returns '—' for lead 0 null amount", () => {
      expect(
        formatPointPrecipitationDisplay({
          precipitation_amount_3h: null,
          precipitation_type: "none",
          precipitation_transition: "none",
        })
      ).toBe("—");
    });
  });

  describe("getTransitionPhases", () => {
    it("parses start and end phase from transition identifier", () => {
      expect(getTransitionPhases("rain_to_snow")).toEqual({
        start: "rain",
        end: "snow",
      });
      expect(getTransitionPhases("snow_to_freezing_rain")).toEqual({
        start: "snow",
        end: "freezing_rain",
      });
      expect(getTransitionPhases("persistent_rain")).toBeNull();
      expect(getTransitionPhases(null)).toBeNull();
    });
  });

  describe("getBarColorForEntry", () => {
    it("returns transparent for lead 0 null amount", () => {
      expect(getBarColorForEntry({ precipitation_amount_3h: null })).toBe("transparent");
    });

    it("returns dry color for 0 amount", () => {
      expect(
        getBarColorForEntry({
          precipitation_amount_3h: 0.0,
          precipitation_type: "none",
        })
      ).toBe(PRECIPITATION_PHASE_TOKENS.dry.color);
    });

    it("returns appropriate color tokens for persistent phases", () => {
      expect(
        getBarColorForEntry({
          precipitation_amount_3h: 2.0,
          precipitation_type: "rain",
          precipitation_transition: "persistent_rain",
        })
      ).toBe(PRECIPITATION_PHASE_TOKENS.rain.color);

      expect(
        getBarColorForEntry({
          precipitation_amount_3h: 2.0,
          precipitation_type: "snow",
          precipitation_transition: "persistent_snow",
        })
      ).toBe(PRECIPITATION_PHASE_TOKENS.snow.color);

      expect(
        getBarColorForEntry({
          precipitation_amount_3h: 2.0,
          precipitation_type: "freezing_rain",
          precipitation_transition: "persistent_freezing_rain",
        })
      ).toBe(PRECIPITATION_PHASE_TOKENS.freezing_rain.color);
    });
  });

  describe("GEFS constants and contracts", () => {
    it("contains exactly 6 physical phase buckets without 'mixed'", () => {
      expect(GEFS_PHYSICAL_PHASES).toEqual([
        "dry",
        "rain",
        "snow",
        "freezing_rain",
        "ice_pellets",
        "unknown",
      ]);
      expect(GEFS_PHYSICAL_PHASES).not.toContain("mixed");
    });

    it("has user-facing labels for all 6 physical phases", () => {
      for (const phase of GEFS_PHYSICAL_PHASES) {
        expect(GEFS_PHASE_LABELS[phase]).toBeDefined();
        expect(typeof GEFS_PHASE_LABELS[phase]).toBe("string");
      }
    });
  });
});
