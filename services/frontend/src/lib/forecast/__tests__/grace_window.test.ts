import { filterGraceWindowValidTimes, isWithinGraceWindow } from "@/lib/forecast/availability";

describe("3-Hour UI Grace Window (Lifecycle V2)", () => {
  // Fixed simulated time: 2026-09-04T09:20:00Z
  // Grace threshold: now - 3h = 2026-09-04T06:20:00Z
  const nowMs = new Date("2026-09-04T09:20:00.000Z").getTime();

  it("strictly hides valid_time < now - 3h and shows valid_time >= now - 3h", () => {
    const t0600 = "2026-09-04T06:00:00.000Z"; // 20m before threshold -> HIDDEN
    const t0619 = "2026-09-04T06:19:59.000Z"; // 1s before threshold -> HIDDEN
    const t0620 = "2026-09-04T06:20:00.000Z"; // exact threshold -> VISIBLE
    const t0621 = "2026-09-04T06:21:00.000Z"; // after threshold -> VISIBLE
    const t0900 = "2026-09-04T09:00:00.000Z"; // future relative to threshold -> VISIBLE
    const t1200 = "2026-09-04T12:00:00.000Z"; // future -> VISIBLE

    expect(isWithinGraceWindow(t0600, nowMs)).toBe(false);
    expect(isWithinGraceWindow(t0619, nowMs)).toBe(false);
    expect(isWithinGraceWindow(t0620, nowMs)).toBe(true);
    expect(isWithinGraceWindow(t0621, nowMs)).toBe(true);
    expect(isWithinGraceWindow(t0900, nowMs)).toBe(true);
    expect(isWithinGraceWindow(t1200, nowMs)).toBe(true);
  });

  it("filters a list of valid times preserving chronological order", () => {
    const validTimes = [
      "2026-09-04T00:00:00.000Z", // expired (< 06:20Z)
      "2026-09-04T03:00:00.000Z", // expired (< 06:20Z)
      "2026-09-04T06:00:00.000Z", // expired (< 06:20Z)
      "2026-09-04T06:20:00.000Z", // visible
      "2026-09-04T09:00:00.000Z", // visible
      "2026-09-04T12:00:00.000Z", // visible
    ];

    const filtered = filterGraceWindowValidTimes(validTimes, nowMs);
    expect(filtered).toEqual([
      "2026-09-04T06:20:00.000Z",
      "2026-09-04T09:00:00.000Z",
      "2026-09-04T12:00:00.000Z",
    ]);
  });

  it("handles null or invalid strings safely", () => {
    expect(isWithinGraceWindow(null, nowMs)).toBe(false);
    expect(isWithinGraceWindow("not-a-date", nowMs)).toBe(false);
  });

  it("simulates time advance in long-lived browser session aging out older valid time", () => {
    const validTimes = ["2026-09-04T06:30:00.000Z", "2026-09-04T09:30:00.000Z"];

    // At 09:20Z: threshold is 06:20Z -> 06:30Z is still visible
    expect(filterGraceWindowValidTimes(validTimes, nowMs)).toEqual([
      "2026-09-04T06:30:00.000Z",
      "2026-09-04T09:30:00.000Z",
    ]);

    // 20 minutes later (09:40Z): threshold becomes 06:40Z -> 06:30Z ages out!
    const laterMs = new Date("2026-09-04T09:40:00.000Z").getTime();
    expect(filterGraceWindowValidTimes(validTimes, laterMs)).toEqual(["2026-09-04T09:30:00.000Z"]);
  });
});
