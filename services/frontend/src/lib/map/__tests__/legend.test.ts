import { buildLegendGradient } from "@/lib/map/legend";

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
