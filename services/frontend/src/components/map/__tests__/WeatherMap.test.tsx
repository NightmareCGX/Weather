import { render } from "@testing-library/react";

import { clearInstances, getInstances, MockMap } from "@/test-utils/maplibre";

import { WeatherMap } from "@/components/map/WeatherMap";
import type { SpatialLayer } from "@/lib/api/types";

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

beforeEach(() => {
  clearInstances();
});

describe("WeatherMap", () => {
  it("creates one map and configures the weather layer from metadata", () => {
    render(<WeatherMap layer={layer} />);

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
    const { rerender } = render(<WeatherMap layer={layer} />);
    const [map] = getInstances();

    rerender(<WeatherMap layer={{ ...layer, lead_time_hours: 24 }} />);

    expect(map.removeLayer).toHaveBeenCalledWith("weather");
    expect(map.removeSource).toHaveBeenCalledWith("weather");
    expect(map.addSource).toHaveBeenCalledTimes(2);
  });

  it("removes the map on unmount", () => {
    const { unmount } = render(<WeatherMap layer={layer} />);
    const [map] = getInstances();

    unmount();

    expect(map.remove).toHaveBeenCalled();
  });
});
