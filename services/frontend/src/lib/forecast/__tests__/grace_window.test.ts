import {
  calculateNextBoundaryDelayMs,
  computeServingStartValidTime,
  filterGraceWindowValidTimes,
  filterServableValidTimes,
  isServableValidTime,
  isWithinGraceWindow,
} from "@/lib/forecast/availability";

describe("Serving Left-Boundary Enforcement (Lifecycle V3 Phase 1)", () => {
  describe("computeServingStartValidTime (exact 3h cadence floor)", () => {
    it("floors UTC time to the 3-hour cadence grid and advances immediately at exact boundaries", () => {
      // 05:59:59Z -> 03:00:00Z
      const t055959 = new Date("2026-09-10T05:59:59.000Z").getTime();
      expect(computeServingStartValidTime(t055959)).toBe("2026-09-10T03:00:00.000Z");

      // 06:00:00Z -> 06:00:00Z (exact boundary advances immediately)
      const t060000 = new Date("2026-09-10T06:00:00.000Z").getTime();
      expect(computeServingStartValidTime(t060000)).toBe("2026-09-10T06:00:00.000Z");

      // 06:00:01Z -> 06:00:00Z
      const t060001 = new Date("2026-09-10T06:00:01.000Z").getTime();
      expect(computeServingStartValidTime(t060001)).toBe("2026-09-10T06:00:00.000Z");

      // 07:00:00Z -> 06:00:00Z
      const t070000 = new Date("2026-09-10T07:00:00.000Z").getTime();
      expect(computeServingStartValidTime(t070000)).toBe("2026-09-10T06:00:00.000Z");

      // 08:59:59Z -> 06:00:00Z
      const t085959 = new Date("2026-09-10T08:59:59.000Z").getTime();
      expect(computeServingStartValidTime(t085959)).toBe("2026-09-10T06:00:00.000Z");

      // 09:00:00Z -> 09:00:00Z (exact boundary advances immediately)
      const t090000 = new Date("2026-09-10T09:00:00.000Z").getTime();
      expect(computeServingStartValidTime(t090000)).toBe("2026-09-10T09:00:00.000Z");

      // 09:00:01Z -> 09:00:00Z
      const t090001 = new Date("2026-09-10T09:00:01.000Z").getTime();
      expect(computeServingStartValidTime(t090001)).toBe("2026-09-10T09:00:00.000Z");
    });
  });

  describe("isServableValidTime & filterServableValidTimes", () => {
    // Current simulated time: 2026-09-10T09:00:00Z -> serving_start = 2026-09-10T09:00:00Z
    const at0900 = new Date("2026-09-10T09:00:00.000Z").getTime();

    it("strictly hides valid_time < serving_start and exposes valid_time >= serving_start", () => {
      const t0600 = "2026-09-10T06:00:00.000Z"; // 3h before anchor -> HIDDEN
      const t0859 = "2026-09-10T08:59:59.000Z"; // before anchor -> HIDDEN
      const t0900 = "2026-09-10T09:00:00.000Z"; // exact anchor -> VISIBLE
      const t1200 = "2026-09-10T12:00:00.000Z"; // future -> VISIBLE

      expect(isServableValidTime(t0600, null, at0900)).toBe(false);
      expect(isServableValidTime(t0859, null, at0900)).toBe(false);
      expect(isServableValidTime(t0900, null, at0900)).toBe(true);
      expect(isServableValidTime(t1200, null, at0900)).toBe(true);
    });

    it("advances immediately at exact boundary 09:00:00Z without now - 3h lag", () => {
      // Prior to 09:00:00Z (e.g. 08:59:59Z), 06:00:00Z was visible
      const at0859 = new Date("2026-09-10T08:59:59.000Z").getTime();
      expect(isServableValidTime("2026-09-10T06:00:00.000Z", null, at0859)).toBe(true);

      // At exactly 09:00:00Z, 06:00:00Z is HIDDEN (no stale 3h grace lag)
      expect(isServableValidTime("2026-09-10T06:00:00.000Z", null, at0900)).toBe(false);
    });

    it("prefers backend-authoritative boundary string when provided", () => {
      const backendStart = "2026-09-10T06:00:00.000Z";
      expect(isServableValidTime("2026-09-10T06:00:00.000Z", backendStart)).toBe(true);
      expect(isServableValidTime("2026-09-10T03:00:00.000Z", backendStart)).toBe(false);
    });

    it("filters a list of valid times preserving chronological order", () => {
      const validTimes = [
        "2026-09-10T00:00:00.000Z",
        "2026-09-10T03:00:00.000Z",
        "2026-09-10T06:00:00.000Z",
        "2026-09-10T09:00:00.000Z",
        "2026-09-10T12:00:00.000Z",
      ];

      const filtered = filterServableValidTimes(validTimes, null, at0900);
      expect(filtered).toEqual(["2026-09-10T09:00:00.000Z", "2026-09-10T12:00:00.000Z"]);
    });

    it("handles null or invalid strings safely", () => {
      expect(isServableValidTime(null, null, at0900)).toBe(false);
      expect(isServableValidTime("not-a-date", null, at0900)).toBe(false);
    });
  });

  describe("calculateNextBoundaryDelayMs", () => {
    it("derives exact remaining delay until next 3h boundary from server timestamps", () => {
      const availability = {
        models: [],
        generated_at: "2026-09-11T08:59:20.000Z",
        serving_start_valid_time: "2026-09-11T06:00:00.000Z",
      };

      // Next boundary is 06:00Z + 3h = 09:00:00Z.
      // Remaining server time = 09:00:00 - 08:59:20 = 40,000ms.
      // With 100ms safety buffer = 40,100ms.
      const delay = calculateNextBoundaryDelayMs(availability, 3, 100);
      expect(delay).toBe(40_100);
    });

    it("returns null when serving_start_valid_time is missing", () => {
      expect(calculateNextBoundaryDelayMs(null)).toBeNull();
      expect(calculateNextBoundaryDelayMs({ models: [] })).toBeNull();
    });
  });

  describe("backward compatibility aliases", () => {
    it("isWithinGraceWindow and filterGraceWindowValidTimes enforce the cadence boundary", () => {
      const at0900 = new Date("2026-09-10T09:00:00.000Z").getTime();
      expect(isWithinGraceWindow("2026-09-10T06:00:00.000Z", at0900)).toBe(false);
      expect(isWithinGraceWindow("2026-09-10T09:00:00.000Z", at0900)).toBe(true);

      const times = ["2026-09-10T06:00:00.000Z", "2026-09-10T09:00:00.000Z"];
      expect(filterGraceWindowValidTimes(times, at0900)).toEqual(["2026-09-10T09:00:00.000Z"]);
    });
  });
});
