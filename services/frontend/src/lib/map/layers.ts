import type { Map } from "maplibre-gl";
import type { SpatialLayer } from "@/lib/api/types";

/**
 * Configure the weather raster source/layer on a MapLibre map from
 * `/v1/maps` metadata.
 *
 * `/v1/maps` is a metadata-only endpoint: it returns the tile template,
 * zoom range, and legend, but the backend does not serve tile imagery yet.
 * The tile template is passed to MapLibre unmodified — MapLibre substitutes
 * `{z}/{x}/{y}` and preserves the query string. When tiles are unavailable
 * the browser requests 404 and MapLibre logs per-tile errors while the base
 * map keeps rendering; this graceful degradation is accepted behavior, not a
 * blocker.
 */
export function applyWeatherLayer(map: Map, layer: SpatialLayer): void {
  removeWeatherLayer(map);

  map.addSource("weather", {
    type: "raster",
    tiles: [layer.tile_url_template],
    tileSize: 256,
    minzoom: layer.min_zoom,
    maxzoom: layer.max_zoom,
  });
  map.addLayer({
    id: "weather",
    type: "raster",
    source: "weather",
    paint: { "raster-opacity": 0.6 },
  });
}

/**
 * Remove the weather raster layer/source if present.
 *
 * Called before applying a new layer (re-apply) and when a selection yields no
 * layer (e.g. the metadata fetch fails or the selection is cleared), so a
 * stale forecast field never lingers over the base map.
 */
export function removeWeatherLayer(map: Map): void {
  if (map.getSource("weather") === undefined) {
    return;
  }
  if (map.getLayer("weather") !== undefined) {
    map.removeLayer("weather");
  }
  map.removeSource("weather");
}
