import type { StyleSpecification } from "maplibre-gl";

/**
 * Base map style for the weather platform.
 *
 * Uses Esri's World Dark Gray Canvas, a high-performance, dark-themed
 * raster basemap designed specifically for meteorological and spatial
 * data visualization. Free, keyless, and served via AWS CloudFront CDN.
 */
export function buildBaseStyle(): StyleSpecification {
  return {
    version: 8,
    sources: {
      osm: {
        type: "raster",
        tiles: [
          "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        ],
        tileSize: 256,
        attribution: "Esri, HERE, Garmin, © OpenStreetMap contributors",
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
