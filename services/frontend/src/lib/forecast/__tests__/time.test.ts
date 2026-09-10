import {
  formatDayHourInTimeZone,
  formatDayHourUtc,
  formatDayHourWithTimeZone,
  formatFullUtc,
  formatLeadTimeHours,
  formatTimeUtc,
  isValidTimeZone,
} from "@/lib/forecast/time";

describe("time formatters", () => {
  describe("existing UTC formatters (unaltered)", () => {
    it("formats short UTC time", () => {
      expect(formatTimeUtc("2026-07-21T06:00:00Z")).toBe("06:00");
    });

    it("formats UTC day and hour", () => {
      expect(formatDayHourUtc("2026-07-21T06:00:00Z")).toBe("Jul 21, 06:00");
    });

    it("formats full UTC date and time", () => {
      expect(formatFullUtc("2026-07-21T06:00:00Z")).toBe("Jul 21, 2026, 06:00");
    });

    it("formats lead offsets", () => {
      expect(formatLeadTimeHours(0)).toBe("+0h");
      expect(formatLeadTimeHours(6)).toBe("+6h");
    });
  });

  describe("isValidTimeZone", () => {
    it("returns true for valid IANA timezones", () => {
      expect(isValidTimeZone("UTC")).toBe(true);
      expect(isValidTimeZone("America/Denver")).toBe(true);
      expect(isValidTimeZone("Asia/Tokyo")).toBe(true);
      expect(isValidTimeZone("Asia/Kathmandu")).toBe(true);
    });

    it("returns false for invalid timezone strings", () => {
      expect(isValidTimeZone("Invalid/Timezone")).toBe(false);
      expect(isValidTimeZone("not_a_zone")).toBe(false);
      expect(isValidTimeZone("")).toBe(false);
    });
  });

  describe("formatDayHourWithTimeZone (Valid display & tooltip)", () => {
    const instant = "2026-09-10T00:00:00Z";

    it("falls back safely to UTC when timezone is null or undefined", () => {
      expect(formatDayHourWithTimeZone(instant, null)).toBe("Sep 10, 00:00 UTC");
      expect(formatDayHourWithTimeZone(instant, undefined)).toBe("Sep 10, 00:00 UTC");
    });

    it("falls back safely to UTC when timezone is invalid", () => {
      expect(formatDayHourWithTimeZone(instant, "Invalid/Zone")).toBe("Sep 10, 00:00 UTC");
    });

    it("formats Denver summer in MDT", () => {
      // 2026-09-10T00:00:00Z is UTC-6 in MDT -> Sep 9, 18:00 MDT
      const formatted = formatDayHourWithTimeZone(instant, "America/Denver");
      expect(formatted).toMatch(/^Sep 9, 18:00 (MDT|GMT-6)$/);
    });

    it("formats Denver winter in MST", () => {
      // 2026-01-10T00:00:00Z is UTC-7 in MST -> Jan 9, 17:00 MST
      const formatted = formatDayHourWithTimeZone("2026-01-10T00:00:00Z", "America/Denver");
      expect(formatted).toMatch(/^Jan 9, 17:00 (MST|GMT-7)$/);
    });

    it("formats Tokyo time with appropriate timezone name", () => {
      // 2026-09-10T00:00:00Z is UTC+9 -> Sep 10, 09:00 JST / GMT+9
      const formatted = formatDayHourWithTimeZone(instant, "Asia/Tokyo");
      expect(formatted).toMatch(/^Sep 10, 09:00 (JST|GMT\+9)$/);
    });

    it("formats Kathmandu (non-whole-hour UTC+5:45)", () => {
      // 2026-09-10T00:00:00Z is UTC+5:45 -> Sep 10, 05:45 GMT+5:45
      const formatted = formatDayHourWithTimeZone(instant, "Asia/Kathmandu");
      expect(formatted).toMatch(/^Sep 10, 05:45 (GMT\+5:45|\+0545)$/);
    });

    it("formats Adelaide (non-whole-hour UTC+9:30 in northern hemisphere summer)", () => {
      // In Sept, Adelaide is standard time UTC+9:30 (ACST) -> Sep 10, 09:30
      const formatted = formatDayHourWithTimeZone(instant, "Australia/Adelaide");
      expect(formatted).toMatch(/^Sep 10, 09:30 (ACST|GMT\+9:30)$/);
    });
  });

  describe("formatDayHourInTimeZone (X-Axis labels)", () => {
    const instant = "2026-09-10T00:00:00Z";

    it("formats without timezone suffix", () => {
      expect(formatDayHourInTimeZone(instant, null)).toBe("Sep 10, 00:00");
      expect(formatDayHourInTimeZone(instant, "America/Denver")).toBe("Sep 9, 18:00");
      expect(formatDayHourInTimeZone(instant, "Asia/Tokyo")).toBe("Sep 10, 09:00");
      expect(formatDayHourInTimeZone(instant, "Asia/Kathmandu")).toBe("Sep 10, 05:45");
    });

    it("falls back to UTC on invalid timezone without throwing", () => {
      expect(formatDayHourInTimeZone(instant, "Not/Real")).toBe("Sep 10, 00:00");
    });
  });

  describe("DST boundary transitions (America/Denver)", () => {
    it("handles spring-forward (skipping 02:00 local time)", () => {
      // 2026-03-08T08:00:00Z is 01:00 MST
      // 2026-03-08T09:00:00Z is 03:00 MDT (spring forward from 01:59 to 03:00)
      const t1 = "2026-03-08T08:00:00Z";
      const t2 = "2026-03-08T09:00:00Z";

      expect(formatDayHourInTimeZone(t1, "America/Denver")).toBe("Mar 8, 01:00");
      expect(formatDayHourInTimeZone(t2, "America/Denver")).toBe("Mar 8, 03:00");

      expect(formatDayHourWithTimeZone(t1, "America/Denver")).toMatch(/^Mar 8, 01:00 (MST|GMT-7)$/);
      expect(formatDayHourWithTimeZone(t2, "America/Denver")).toMatch(/^Mar 8, 03:00 (MDT|GMT-6)$/);
    });

    it("handles fall-back (repeating 01:00 with distinct timezone offsets in tooltip)", () => {
      // 2026-11-01T07:00:00Z is 01:00 MDT (daylight time)
      // 2026-11-01T08:00:00Z is 01:00 MST (standard time)
      const t1 = "2026-11-01T07:00:00Z";
      const t2 = "2026-11-01T08:00:00Z";

      // X-axis displays repeated 01:00
      expect(formatDayHourInTimeZone(t1, "America/Denver")).toBe("Nov 1, 01:00");
      expect(formatDayHourInTimeZone(t2, "America/Denver")).toBe("Nov 1, 01:00");

      // Tooltip disambiguates through timezone representation
      const tooltip1 = formatDayHourWithTimeZone(t1, "America/Denver");
      const tooltip2 = formatDayHourWithTimeZone(t2, "America/Denver");

      expect(tooltip1).toMatch(/^Nov 1, 01:00 (MDT|GMT-6)$/);
      expect(tooltip2).toMatch(/^Nov 1, 01:00 (MST|GMT-7)$/);
      expect(tooltip1).not.toBe(tooltip2);
    });
  });

  describe("midnight formatting regression coverage (00:00 vs 24:00 portability)", () => {
    it("formats UTC midnight as 00:00 (never 24:00)", () => {
      const utcMidnight = "2026-09-10T00:00:00Z";
      const formatted = formatDayHourInTimeZone(utcMidnight, null);
      expect(formatted).toBe("Sep 10, 00:00");
      expect(formatted).not.toContain("24:00");

      const formattedWithTz = formatDayHourWithTimeZone(utcMidnight, null);
      expect(formattedWithTz).toBe("Sep 10, 00:00 UTC");
      expect(formattedWithTz).not.toContain("24:00");

      // Also verify older UTC formatters
      expect(formatTimeUtc(utcMidnight)).toBe("00:00");
      expect(formatTimeUtc(utcMidnight)).not.toContain("24:00");

      expect(formatDayHourUtc(utcMidnight)).toBe("Sep 10, 00:00");
      expect(formatDayHourUtc(utcMidnight)).not.toContain("24:00");

      expect(formatFullUtc(utcMidnight)).toBe("Sep 10, 2026, 00:00");
      expect(formatFullUtc(utcMidnight)).not.toContain("24:00");
    });

    it("formats Denver local midnight as 00:00 MDT (never 24:00)", () => {
      // 2026-09-10T06:00:00Z in America/Denver (MDT) is midnight
      const denverMidnight = "2026-09-10T06:00:00Z";
      const formatted = formatDayHourInTimeZone(denverMidnight, "America/Denver");
      expect(formatted).toBe("Sep 10, 00:00");
      expect(formatted).not.toContain("24:00");

      const formattedWithTz = formatDayHourWithTimeZone(denverMidnight, "America/Denver");
      expect(formattedWithTz).toMatch(/^Sep 10, 00:00 (MDT|GMT-6)$/);
      expect(formattedWithTz).not.toContain("24:00");
    });
  });
});
