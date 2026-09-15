import { buildLegendGradient, getLegendTicks } from "@/lib/map/legend";

describe("buildLegendGradient", () => {
  it("builds a linear gradient from the legend stops", () => {
    const gradient = buildLegendGradient([
      [-40, "#0000ff"],
      [0, "#00ff00"],
      [40, "#ff0000"],
    ]);

    expect(gradient).toContain("linear-gradient(to right");
    expect(gradient).toContain("#0000ff 0%");
    expect(gradient).toContain("#00ff00 50%");
    expect(gradient).toContain("#ff0000 100%");
  });

  it("returns a transparent gradient for an empty stop list", () => {
    expect(buildLegendGradient([])).toContain("linear-gradient");
  });
});

describe("getLegendTicks", () => {
  it("returns empty array for empty stops", () => {
    expect(getLegendTicks([])).toEqual([]);
  });

  it("returns all ticks when stops are spaced widely enough", () => {
    const stops: [number, string][] = [
      [0, "#fff"],
      [10, "#aaa"],
      [25, "#000"],
    ];
    const ticks = getLegendTicks(stops);
    expect(ticks).toHaveLength(3);
    expect(ticks[0]).toEqual({ value: 0, label: "0", positionPercent: 0 });
    expect(ticks[1]).toEqual({ value: 10, label: "10", positionPercent: 40 });
    expect(ticks[2]).toEqual({ value: 25, label: "25", positionPercent: 100 });
  });

  it("applies stretch map directly when configured", () => {
    const stops: [number, string][] = [
      [0.0, "#a50026"],
      [3.0, "#fee090"],
      [9.0, "#4575b4"],
      [20.0, "#ffffff"],
    ];
    const ticks = getLegendTicks(stops, "cloud_ceiling");
    expect(ticks).toHaveLength(4);
    expect(ticks[0]).toEqual({ value: 0, label: "0", positionPercent: 0 });
    expect(ticks[1]).toEqual({ value: 3, label: "3", positionPercent: 30 });
    expect(ticks[2]).toEqual({ value: 9, label: "9", positionPercent: 60 });
    expect(ticks[3]).toEqual({ value: 20, label: "20", positionPercent: 100 });
  });

  it("retains all uniform wind_10m ticks across 0..120 km/h range without thinning", () => {
    const windStops: [number, string][] = [
      [0.0, "#ffffff"],
      [20.0, "#c7e9c0"],
      [40.0, "#74c476"],
      [60.0, "#41ab5d"],
      [80.0, "#4292c6"],
      [100.0, "#08519c"],
      [120.0, "#49006a"],
    ];
    const ticks = getLegendTicks(windStops, "wind_10m");
    expect(ticks).toHaveLength(7);
    expect(ticks.map((t) => t.value)).toEqual([0, 20, 40, 60, 80, 100, 120]);
    expect(ticks.map((t) => t.label)).toEqual(["0", "20", "40", "60", "80", "100", "120"]);
  });

  it("drops overlapping close stops and keeps minimum visual distance", () => {
    const stops: [number, string][] = [
      [0, "#1"],
      [0.5, "#2"],
      [1.0, "#3"],
      [3.0, "#4"],
      [6.0, "#5"],
      [10.0, "#6"],
      [24.0, "#7"],
    ];
    const ticks = getLegendTicks(stops, null, 18);
    expect(ticks.length).toBeLessThan(stops.length);
    expect(ticks[0].value).toBe(0);
    expect(ticks[ticks.length - 1].value).toBe(24);
    for (let i = 1; i < ticks.length; i++) {
      expect(ticks[i].positionPercent - ticks[i - 1].positionPercent).toBeGreaterThanOrEqual(18);
    }
  });
});
