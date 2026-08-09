import { clearInstances, MockMap } from "@/test-utils/maplibre";
import type { Map as MapLibreMap } from "maplibre-gl";

import type { SpatialLayer } from "@/lib/api/types";
import { applyWeatherLayer } from "@/lib/map/layers";

const layer: SpatialLayer = {
  tile_url_template: "/v1/maps/gfs/temperature_2m/surface/{z}/{x}/{y}.png?lead_time_hours=12",
  min_zoom: 0,
  max_zoom: 9,
  lead_time_hours: 12,
  legend: {
    unit: "°C",
    stops: [
      [-40, "#0000ff"],
      [0, "#00ff00"],
      [40, "#ff0000"],
    ],
  },
};

function mockMap(): MapLibreMap {
  return new MockMap() as unknown as MapLibreMap;
}

beforeEach(() => {
  clearInstances();
});

describe("applyWeatherLayer", () => {
  it("adds a weather raster source and layer from /v1/maps metadata", () => {
    const map = mockMap();

    applyWeatherLayer(map, layer);

    expect((map.addSource as unknown as jest.Mock).mock.calls[0]).toEqual([
      "weather",
      expect.objectContaining({
        type: "raster",
        tiles: [layer.tile_url_template],
        minzoom: 0,
        maxzoom: 9,
      }),
    ]);
    expect((map.addLayer as unknown as jest.Mock).mock.calls[0]).toEqual([
      expect.objectContaining({ id: "weather", type: "raster", source: "weather" }),
    ]);
  });

  it("passes the tile template through unmodified", () => {
    const map = mockMap();

    applyWeatherLayer(map, layer);

    expect((map.addSource as unknown as jest.Mock).mock.calls[0][1]).toEqual(
      expect.objectContaining({ tiles: [layer.tile_url_template] })
    );
  });

  it("removes and re-adds the layer on re-apply", () => {
    const map = mockMap();

    applyWeatherLayer(map, layer);
    applyWeatherLayer(map, { ...layer, lead_time_hours: 24 });

    expect((map.removeLayer as unknown as jest.Mock).mock.calls).toEqual([["weather"]]);
    expect((map.removeSource as unknown as jest.Mock).mock.calls).toEqual([["weather"]]);
    expect((map.addSource as unknown as jest.Mock).mock.calls).toHaveLength(2);
  });
});
