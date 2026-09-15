import type { StyleSpecification } from "maplibre-gl";

/**
 * Base map style for the weather platform.
 *
 * Uses official OpenStreetMap base tiles inverted to a dark theme via
 * MapLibre's native WebGL raster shader properties. Raster-only: no glyphs/sprites
 * are required, keeping the style self-contained and offline-testable.
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
        paint: {
          // Hardware-accelerated dark inversion via MapLibre WebGL raster shader:
          // [brightness-min: 1.0, brightness-max: 0.0] inverts the luminance range (1 - RGB),
          // desaturates all color to grayscale, and applies slight contrast for dark tech HUD theme.
          "raster-brightness-min": 1.0,
          "raster-brightness-max": 0.0,
          "raster-saturation": -1.0,
          "raster-contrast": 0.1,
        },
      },
    ],
  };
}
