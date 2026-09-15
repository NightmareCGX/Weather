import { buildBaseStyle } from "@/lib/map/baseStyle";

describe("buildBaseStyle", () => {
  it("returns a raster-only v8 style with an OSM base source and layer", () => {
    const style = buildBaseStyle();

    expect(style.version).toBe(8);

    const osmSource = style.sources["osm"];
    expect(osmSource).toBeDefined();
    expect(osmSource).toMatchObject({
      type: "raster",
      tiles: [
        "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
      ],
    });

    expect(style.layers).toContainEqual(
      expect.objectContaining({ id: "osm", type: "raster", source: "osm" })
    );
  });
});
