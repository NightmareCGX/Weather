import { render } from "@testing-library/react";

import {
  clearInstances,
  clearMarkers,
  getInstances,
  getMarkers,
  MockMap,
} from "@/test-utils/maplibre";

import { WeatherMap } from "@/components/map/WeatherMap";
import type { SelectedLocation, SpatialLayer } from "@/lib/api/types";

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

const coordinates: SelectedLocation = {
  name: "38.1911, -106.8175",
  object: "coordinates",
  latitude: 38.1911,
  longitude: -106.8175,
  elevation_m: null,
  region: null,
  country: null,
  id: null,
  resolvedVia: "coordinates",
};

beforeEach(() => {
  clearInstances();
  clearMarkers();
});

function renderMap(props: Partial<Parameters<typeof WeatherMap>[0]> = {}) {
  return render(
    <WeatherMap
      layer={layer}
      selectedLocation={null}
      validTime={null}
      onSelect={jest.fn()}
      {...props}
    />
  );
}

describe("WeatherMap", () => {
  it("creates one map and configures the weather layer from metadata", () => {
    renderMap();

    const [map] = getInstances();
    expect(map).toBeInstanceOf(MockMap);
    expect(map.addControl).toHaveBeenCalled();
    expect((map.addSource as jest.Mock).mock.calls[0]).toEqual([
      "weather",
      expect.objectContaining({ tiles: [layer.tile_url_template] }),
    ]);
    expect((map.addLayer as jest.Mock).mock.calls[0]).toEqual([
      expect.objectContaining({ id: "weather", source: "weather" }),
    ]);
  });

  it("re-syncs the weather layer when the metadata changes", () => {
    const { rerender } = renderMap();
    const [map] = getInstances();

    rerender(
      <WeatherMap
        layer={{ ...layer, lead_time_hours: 24 }}
        selectedLocation={null}
        validTime={null}
        onSelect={jest.fn()}
      />
    );

    expect(map.removeLayer).toHaveBeenCalledWith("weather");
    expect(map.removeSource).toHaveBeenCalledWith("weather");
    expect(map.addSource).toHaveBeenCalledTimes(2);
  });

  it("replaces the weather layer immediately when map.loaded() is false but style is loaded (tiles in flight)", () => {
    const { rerender } = renderMap();
    const [map] = getInstances();

    // Simulate active raster tile downloads: loaded() is false, but isStyleLoaded() is true
    map.isLoaded = false;
    expect(map.loaded()).toBe(false);
    expect(map.isStyleLoaded()).toBe(true);

    rerender(
      <WeatherMap
        layer={{
          ...layer,
          lead_time_hours: 24,
          tile_url_template:
            "/v1/maps/gfs/temperature_2m/surface/{z}/{x}/{y}.png?lead_time_hours=24",
        }}
        selectedLocation={null}
        validTime={null}
        onSelect={jest.fn()}
      />
    );

    // Must still replace the layer immediately without waiting for tile downloads to finish
    expect(map.removeLayer).toHaveBeenCalledWith("weather");
    expect(map.removeSource).toHaveBeenCalledWith("weather");
    expect(map.addSource).toHaveBeenCalledWith(
      "weather",
      expect.objectContaining({
        tiles: ["/v1/maps/gfs/temperature_2m/surface/{z}/{x}/{y}.png?lead_time_hours=24"],
      })
    );
  });

  it("delays applying weather layer when style is not ready on mount and applies latest layer on load", () => {
    // Set MockMap default to style not loaded during mount
    clearInstances();
    MockMap.defaultIsStyleLoaded = false;

    const { rerender } = render(
      <WeatherMap layer={layer} selectedLocation={null} validTime={null} onSelect={jest.fn()} />
    );
    const [map] = getInstances();
    expect(map.isStyleLoaded()).toBe(false);

    // Initial mount with style not ready must not have applied layer yet
    expect(map.addSource).not.toHaveBeenCalled();

    // Update layer while style is still loading
    rerender(
      <WeatherMap
        layer={{
          ...layer,
          lead_time_hours: 36,
          tile_url_template:
            "/v1/maps/gfs/temperature_2m/surface/{z}/{x}/{y}.png?lead_time_hours=36",
        }}
        selectedLocation={null}
        validTime={null}
        onSelect={jest.fn()}
      />
    );
    expect(map.addSource).not.toHaveBeenCalled();

    // Now style finishes loading and fires 'load' event
    map._isStyleLoaded = true;
    map.fire("load");

    // The latest layer (lead_time_hours: 36) must be applied
    expect(map.addSource).toHaveBeenCalledWith(
      "weather",
      expect.objectContaining({
        tiles: ["/v1/maps/gfs/temperature_2m/surface/{z}/{x}/{y}.png?lead_time_hours=36"],
      })
    );
  });

  it("removes the weather layer when the layer becomes null (selection cleared/failed)", () => {
    const { rerender } = renderMap();
    const [map] = getInstances();

    rerender(
      <WeatherMap layer={null} selectedLocation={null} validTime={null} onSelect={jest.fn()} />
    );

    expect(map.removeLayer).toHaveBeenCalledWith("weather");
    expect(map.removeSource).toHaveBeenCalledWith("weather");
  });

  it("rapidly transitions A -> B -> C layers directly with only C remaining installed", () => {
    const { rerender } = renderMap();
    const [map] = getInstances();

    const layerA: SpatialLayer = { ...layer, lead_time_hours: 6 };
    const layerB: SpatialLayer = { ...layer, lead_time_hours: 12 };
    const layerC: SpatialLayer = { ...layer, lead_time_hours: 18 };

    // Initial A
    rerender(
      <WeatherMap layer={layerA} selectedLocation={null} validTime={null} onSelect={jest.fn()} />
    );
    expect(map.addSource).toHaveBeenLastCalledWith(
      "weather",
      expect.objectContaining({ tiles: [layerA.tile_url_template] })
    );

    // Switch to B
    rerender(
      <WeatherMap layer={layerB} selectedLocation={null} validTime={null} onSelect={jest.fn()} />
    );
    expect(map.addSource).toHaveBeenLastCalledWith(
      "weather",
      expect.objectContaining({ tiles: [layerB.tile_url_template] })
    );

    // Switch to C
    rerender(
      <WeatherMap layer={layerC} selectedLocation={null} validTime={null} onSelect={jest.fn()} />
    );
    expect(map.addSource).toHaveBeenLastCalledWith(
      "weather",
      expect.objectContaining({ tiles: [layerC.tile_url_template] })
    );
  });

  it("transitions across all four selector dimensions (model, variable, initial time, lead time)", () => {
    const { rerender } = renderMap();
    const [map] = getInstances();

    // 1. Lead time transition
    const leadLayer: SpatialLayer = {
      ...layer,
      lead_time_hours: 18,
      tile_url_template:
        "/v1/maps/gfs/temperature_2m/surface/{z}/{x}/{y}.png?lead_time_hours=18&initial_time=2026-08-13T00%3A00%3A00Z",
    };
    rerender(
      <WeatherMap layer={leadLayer} selectedLocation={null} validTime={null} onSelect={jest.fn()} />
    );
    expect(map.addSource).toHaveBeenLastCalledWith(
      "weather",
      expect.objectContaining({ tiles: [leadLayer.tile_url_template] })
    );

    // 2. Variable transition
    const variableLayer: SpatialLayer = {
      ...layer,
      tile_url_template:
        "/v1/maps/gfs/precipitation_rate/surface/{z}/{x}/{y}.png?lead_time_hours=18&initial_time=2026-08-13T00%3A00%3A00Z",
    };
    rerender(
      <WeatherMap
        layer={variableLayer}
        selectedLocation={null}
        validTime={null}
        onSelect={jest.fn()}
      />
    );
    expect(map.addSource).toHaveBeenLastCalledWith(
      "weather",
      expect.objectContaining({ tiles: [variableLayer.tile_url_template] })
    );

    // 3. Initial time transition
    const initialTimeLayer: SpatialLayer = {
      ...layer,
      tile_url_template:
        "/v1/maps/gfs/precipitation_rate/surface/{z}/{x}/{y}.png?lead_time_hours=18&initial_time=2026-08-13T06%3A00%3A00Z",
    };
    rerender(
      <WeatherMap
        layer={initialTimeLayer}
        selectedLocation={null}
        validTime={null}
        onSelect={jest.fn()}
      />
    );
    expect(map.addSource).toHaveBeenLastCalledWith(
      "weather",
      expect.objectContaining({ tiles: [initialTimeLayer.tile_url_template] })
    );

    // 4. Model transition
    const modelLayer: SpatialLayer = {
      ...layer,
      tile_url_template:
        "/v1/maps/gefs/precipitation_rate/surface/{z}/{x}/{y}.png?lead_time_hours=18&initial_time=2026-08-13T06%3A00%3A00Z",
    };
    rerender(
      <WeatherMap
        layer={modelLayer}
        selectedLocation={null}
        validTime={null}
        onSelect={jest.fn()}
      />
    );
    expect(map.addSource).toHaveBeenLastCalledWith(
      "weather",
      expect.objectContaining({ tiles: [modelLayer.tile_url_template] })
    );
  });

  it("removes the map on unmount", () => {
    const { unmount } = renderMap();
    const [map] = getInstances();

    unmount();

    expect(map.remove).toHaveBeenCalled();
  });

  it("fires onSelect with the clicked coordinates", () => {
    const onSelect = jest.fn();
    renderMap({ onSelect });
    const [map] = getInstances();

    // The click listener is only attached after the map loads, matching the
    // real maplibre lifecycle.
    map.fire("load");
    map.fire("click", { lngLat: { lat: 38.1911, lng: -106.8175 } });

    expect(onSelect).toHaveBeenCalledTimes(1);
    expect(onSelect.mock.calls[0][0]).toMatchObject({
      object: "coordinates",
      resolvedVia: "coordinates",
      latitude: 38.1911,
      longitude: -106.8175,
    });
  });

  it("ignores click events without resolved coordinates", () => {
    const onSelect = jest.fn();
    renderMap({ onSelect });
    const [map] = getInstances();

    map.fire("load");
    map.fire("click", {});

    expect(onSelect).not.toHaveBeenCalled();
  });

  it("adds and positions a marker when a location is selected", () => {
    renderMap({ selectedLocation: coordinates });

    const [marker] = getMarkers();
    expect(marker).toBeDefined();
    expect(marker.setLngLat).toHaveBeenCalledWith([coordinates.longitude, coordinates.latitude]);
    // `setLngLat` must run before `addTo`: maplibre's `addTo` reads the
    // marker's coordinates, so an un-positioned marker throws.
    expect(marker.setLngLat.mock.invocationCallOrder[0]).toBeLessThan(
      marker.addTo.mock.invocationCallOrder[0]
    );
  });

  it("moves the existing marker when the selection changes", () => {
    const { rerender } = renderMap({ selectedLocation: coordinates });
    const [marker] = getMarkers();

    const other: SelectedLocation = { ...coordinates, latitude: 40, longitude: -105 };
    rerender(
      <WeatherMap layer={layer} selectedLocation={other} validTime={null} onSelect={jest.fn()} />
    );

    expect(getMarkers()).toHaveLength(1);
    expect(marker.setLngLat).toHaveBeenLastCalledWith([-105, 40]);
  });

  it("removes the marker when the selection is cleared", () => {
    const { rerender } = renderMap({ selectedLocation: coordinates });
    const [marker] = getMarkers();

    rerender(
      <WeatherMap layer={layer} selectedLocation={null} validTime={null} onSelect={jest.fn()} />
    );

    expect(marker.remove).toHaveBeenCalled();
  });
});
