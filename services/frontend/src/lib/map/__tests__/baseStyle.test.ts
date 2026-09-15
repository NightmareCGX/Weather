import { buildBaseStyle } from "@/lib/map/baseStyle";

describe("buildBaseStyle", () => {
  it("returns a raster-only v8 style with an OSM base source and layer", () => {
    const style = buildBaseStyle();

    expect(style.version).toBe(8);

    const osmSource = style.sources["osm"];
    expect(osmSource).toBeDefined();
    expect(osmSource).toMatchObject({
      type: "raster",
      tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
    });

    expect(style.layers).toContainEqual(
      expect.objectContaining({
        id: "osm",
        type: "raster",
        source: "osm",
        paint: expect.objectContaining({
          "raster-brightness-min": 1.0,
          "raster-brightness-max": 0.0,
          "raster-saturation": -1.0,
        }),
      })
    );
  });
});
