import type { StyleSpecification } from "maplibre-gl";

/**
 * Base map style for the weather platform.
 *
 * Raster-only: OpenStreetMap base tiles. No glyphs/sprites are required, so
 * the style stays self-contained and works offline in tests (maplibre is
 * mocked there, so no tile requests fire).
 */
export function buildBaseStyle(): StyleSpecification {
  return {
    version: 8,
    sources: {
      osm: {
        type: "raster",
        tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
        tileSize: 256,
        attribution: "© OpenStreetMap contributors",
      },
    },
    layers: [
      {
        id: "osm",
        type: "raster",
        source: "osm",
      },
    ],
  };
}
