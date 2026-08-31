import {
  formatPointPrecipitationDisplay,
  formatTransitionName,
  getBarColorForEntry,
  getPointForecastPhaseLabel,
  getPrecipitationPhaseMeta,
  getTransitionPhases,
  GEFS_PHYSICAL_PHASES,
  GEFS_PHASE_LABELS,
  PRECIPITATION_PHASE_TOKENS,
} from "../precipitation";

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

    it("returns 'Mixed' for complex multi-phase transitions", () => {
      expect(
        getPointForecastPhaseLabel({
          precipitation_amount_3h: 1.1,
          precipitation_type: "mixed",
          precipitation_transition: "mixed_transition",
        })
      ).toBe("Mixed");
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
            precipitation_amount_3h: 0.0,
            precipitation_type: "none",
            precipitation_transition: "none",
          },
          "mm"
        )
      ).toBe("0 mm · Dry");
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
