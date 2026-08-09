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

describe("WeatherMap", () => {
  it("creates one map and configures the weather layer from metadata", () => {
    render(<WeatherMap layer={layer} selectedLocation={null} onSelect={jest.fn()} />);

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
    const { rerender } = render(
      <WeatherMap layer={layer} selectedLocation={null} onSelect={jest.fn()} />
    );
    const [map] = getInstances();

    rerender(
      <WeatherMap
        layer={{ ...layer, lead_time_hours: 24 }}
        selectedLocation={null}
        onSelect={jest.fn()}
      />
    );

    expect(map.removeLayer).toHaveBeenCalledWith("weather");
    expect(map.removeSource).toHaveBeenCalledWith("weather");
    expect(map.addSource).toHaveBeenCalledTimes(2);
  });

  it("removes the map on unmount", () => {
    const { unmount } = render(
      <WeatherMap layer={layer} selectedLocation={null} onSelect={jest.fn()} />
    );
    const [map] = getInstances();

    unmount();

    expect(map.remove).toHaveBeenCalled();
  });

  it("fires onSelect with the clicked coordinates", () => {
    const onSelect = jest.fn();
    render(<WeatherMap layer={layer} selectedLocation={null} onSelect={onSelect} />);
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
    render(<WeatherMap layer={layer} selectedLocation={null} onSelect={onSelect} />);
    const [map] = getInstances();

    map.fire("load");
    map.fire("click", {});

    expect(onSelect).not.toHaveBeenCalled();
  });

  it("adds and positions a marker when a location is selected", () => {
    render(<WeatherMap layer={layer} selectedLocation={coordinates} onSelect={jest.fn()} />);

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
    const { rerender } = render(
      <WeatherMap layer={layer} selectedLocation={coordinates} onSelect={jest.fn()} />
    );
    const [marker] = getMarkers();

    const other: SelectedLocation = { ...coordinates, latitude: 40, longitude: -105 };
    rerender(<WeatherMap layer={layer} selectedLocation={other} onSelect={jest.fn()} />);

    expect(getMarkers()).toHaveLength(1);
    expect(marker.setLngLat).toHaveBeenLastCalledWith([-105, 40]);
  });

  it("removes the marker when the selection is cleared", () => {
    const { rerender } = render(
      <WeatherMap layer={layer} selectedLocation={coordinates} onSelect={jest.fn()} />
    );
    const [marker] = getMarkers();

    rerender(<WeatherMap layer={layer} selectedLocation={null} onSelect={jest.fn()} />);

    expect(marker.remove).toHaveBeenCalled();
  });
});
