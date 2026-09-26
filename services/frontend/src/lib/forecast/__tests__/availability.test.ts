import {
  buildForecastOptions,
  buildPinnedTileUrl,
  defaultModel,
  defaultVariable,
  findInitialTime,
  findModel,
  findVariable,
  PREFERRED_VARIABLE_ORDER,
  resolveSpatialLayer,
  resolveValidTime,
  sortVariablesByCanonicalOrder,
} from "@/lib/forecast/availability";
import type { ForecastAvailability } from "@/lib/api/types";

const mockAvailability: ForecastAvailability = {
  models: [
    {
      id: "gfs",
      name: "Global Forecast System",
      is_ensemble: false,
      variables: [
        {
          id: "temperature_2m",
          name: "2-Meter Temperature",
          unit: "°C",
          initial_times: [
            {
              value: "2026-08-13T00:00:00Z",
              lead_time_hours: [0, 6, 12, 18],
            },
            {
              value: "2026-08-13T06:00:00Z",
              lead_time_hours: [0, 6, 12],
            },
          ],
          layer: {
            tile_url_template:
              "/v1/maps/gfs/temperature_2m/surface/{z}/{x}/{y}.png?lead_time_hours={lead_time_hours}&initial_time={initial_time}",
            min_zoom: 0,
            max_zoom: 9,
            legend: {
              unit: "°C",
              stops: [
                [-40, "#0000ff"],
                [40, "#ff0000"],
              ],
            },
          },
        },
      ],
    },
  ],
};

describe("availability helpers", () => {
  it("resolves default model, variable, and initial time", () => {
    expect(defaultModel(mockAvailability)).toBe("gfs");
    const model = findModel(mockAvailability, "gfs");
    expect(defaultVariable(model)).toBe("temperature_2m");
    const variable = findVariable(model, "temperature_2m");
    const initialTime = findInitialTime(variable, "2026-08-13T00:00:00Z");
    expect(initialTime?.lead_time_hours).toEqual([0, 6, 12, 18]);
  });

  it("builds cascading forecast options", () => {
    const options = buildForecastOptions(mockAvailability, {
      model: "gfs",
      variable: "temperature_2m",
    });
    expect(options.models.length).toBe(1);
    expect(options.model?.id).toBe("gfs");
    expect(options.variables.length).toBe(1);
    expect(options.variable?.id).toBe("temperature_2m");
    expect(options.initialTimes.length).toBe(2);
    expect(options.leadTimes).toEqual([0, 6, 12, 18]);
    expect(options.validTimes?.length).toBeGreaterThan(0);
  });

  it("computes valid time correctly", () => {
    expect(resolveValidTime("2026-08-13T00:00:00Z", 6)).toBe("2026-08-13T06:00:00.000Z");
    expect(resolveValidTime(null, 6)).toBeNull();
    expect(resolveValidTime("2026-08-13T00:00:00Z", null)).toBeNull();
  });

  it("sorts variables by canonical meteorological priority", () => {
    const rawVariables: any[] = [
      { id: "snow_depth", name: "Snow Depth" },
      { id: "cloud_ceiling", name: "Cloud Ceiling" },
      { id: "cloud_cover_3h", name: "Cloud Cover" },
      { id: "visibility", name: "Visibility" },
      { id: "precipitation_rate", name: "Precipitation Rate" },
      { id: "precipitation_amount_3h", name: "3-Hour Precipitation" },
      { id: "wind_gust", name: "Wind Gust" },
      { id: "wind_10m", name: "10-Meter Wind" },
      { id: "relative_humidity_2m", name: "Relative Humidity" },
      { id: "temperature_2m", name: "Temperature" },
      { id: "custom_unknown_var", name: "Unknown Variable" },
    ];

    const sorted = sortVariablesByCanonicalOrder(rawVariables);
    const sortedIds = sorted.map((v) => v.id);

    expect(sortedIds).toEqual([
      "temperature_2m",
      "relative_humidity_2m",
      "wind_10m",
      "wind_gust",
      "precipitation_amount_3h",
      "precipitation_rate",
      "visibility",
      "cloud_cover_3h",
      "cloud_ceiling",
      "snow_depth",
      "custom_unknown_var",
    ]);
  });

  it("prioritizes temperature_2m in defaultVariable even if not at index 0", () => {
    const modelWithUnsortedVars: any = {
      id: "gfs",
      name: "GFS",
      variables: [
        { id: "cloud_ceiling", name: "Cloud Ceiling" },
        { id: "wind_10m", name: "10-Meter Wind" },
        { id: "temperature_2m", name: "Temperature" },
      ],
    };

    expect(defaultVariable(modelWithUnsortedVars)).toBe("temperature_2m");
  });

  it("falls back to the first sorted variable if temperature_2m is absent", () => {
    const modelWithoutTemp: any = {
      id: "gefs_subset",
      name: "GEFS Subset",
      variables: [
        { id: "snow_depth", name: "Snow Depth" },
        { id: "relative_humidity_2m", name: "Relative Humidity" },
      ],
    };

    expect(defaultVariable(modelWithoutTemp)).toBe("relative_humidity_2m");
  });
});

describe("buildPinnedTileUrl", () => {
  const pinnedTemplate =
    "/v1/maps/gfs/temperature_2m/surface/{z}/{x}/{y}.png?valid_time={valid_time}&initial_time={source_cycle}";

  it("substitutes the source cycle so tile URLs are immutable per serving cycle", () => {
    expect(
      buildPinnedTileUrl(pinnedTemplate, "2026-08-13T12%3A00%3A00Z", "2026-08-13T00:00:00Z")
    ).toBe(
      "/v1/maps/gfs/temperature_2m/surface/{z}/{x}/{y}.png?valid_time=2026-08-13T12%3A00%3A00Z&initial_time=2026-08-13T00%3A00%3A00Z"
    );
  });

  it("strips the pinned parameter when the serving cycle is unknown", () => {
    expect(buildPinnedTileUrl(pinnedTemplate, "2026-08-13T12%3A00%3A00Z", null)).toBe(
      "/v1/maps/gfs/temperature_2m/surface/{z}/{x}/{y}.png?valid_time=2026-08-13T12%3A00%3A00Z"
    );
  });

  it("leaves templates without the source-cycle placeholder untouched", () => {
    const unpinnedTemplate =
      "/v1/maps/gfs/temperature_2m/surface/{z}/{x}/{y}.png?valid_time={valid_time}";
    expect(
      buildPinnedTileUrl(unpinnedTemplate, "2026-08-13T12%3A00%3A00Z", "2026-08-13T00:00:00Z")
    ).toBe(
      "/v1/maps/gfs/temperature_2m/surface/{z}/{x}/{y}.png?valid_time=2026-08-13T12%3A00%3A00Z"
    );
  });
});

describe("resolveSpatialLayer", () => {
  it("synchronously constructs authoritative SpatialLayer from availability descriptor with valid_time", () => {
    const layer = resolveSpatialLayer(mockAvailability, {
      model: "gfs",
      variable: "temperature_2m",
      validTime: "2026-08-13T12:00:00Z",
    });

    expect(layer).not.toBeNull();
    expect(layer?.tile_url_template).toBe(
      "/v1/maps/gfs/temperature_2m/surface/{z}/{x}/{y}.png?valid_time=2026-08-13T12%3A00%3A00Z"
    );
    expect(layer?.min_zoom).toBe(0);
    expect(layer?.max_zoom).toBe(9);
    expect(layer?.valid_time).toBe("2026-08-13T12:00:00Z");
    expect(layer?.legend.unit).toBe("°C");
  });

  it("pins tile URLs to the per-valid-time source cycle when the backend template carries it", () => {
    const pinnedAvailability: ForecastAvailability = {
      models: [
        {
          id: "gfs",
          name: "Global Forecast System",
          is_ensemble: false,
          variables: [
            {
              id: "temperature_2m",
              name: "2-Meter Temperature",
              unit: "°C",
              initial_times: [
                {
                  value: "2026-08-13T00:00:00Z",
                  lead_time_hours: [0, 6, 12],
                },
              ],
              valid_times: [
                {
                  valid_time: "2026-08-13T12:00:00Z",
                  source_cycle: "2026-08-13T00:00:00Z",
                  lead_time_hours: 12,
                  servable: true,
                  available_members: 1,
                  expected_members: 1,
                  coverage_ratio: 1,
                },
              ],
              layer: {
                tile_url_template:
                  "/v1/maps/gfs/temperature_2m/surface/{z}/{x}/{y}.png?lead_time_hours={lead_time_hours}&initial_time={initial_time}",
                valid_time_tile_url_template:
                  "/v1/maps/gfs/temperature_2m/surface/{z}/{x}/{y}.png?valid_time={valid_time}&initial_time={source_cycle}",
                min_zoom: 0,
                max_zoom: 9,
                legend: {
                  unit: "°C",
                  stops: [
                    [-40, "#0000ff"],
                    [40, "#ff0000"],
                  ],
                },
              },
            },
          ],
        },
      ],
    };

    const layer = resolveSpatialLayer(pinnedAvailability, {
      model: "gfs",
      variable: "temperature_2m",
      validTime: "2026-08-13T12:00:00Z",
    });

    expect(layer).not.toBeNull();
    expect(layer?.tile_url_template).toBe(
      "/v1/maps/gfs/temperature_2m/surface/{z}/{x}/{y}.png?valid_time=2026-08-13T12%3A00%3A00Z&initial_time=2026-08-13T00%3A00%3A00Z"
    );
    expect(layer?.source_cycle).toBe("2026-08-13T00:00:00Z");
  });

  it("returns null when the selection carries no valid time (Lifecycle V2 selections always do)", () => {
    expect(
      resolveSpatialLayer(mockAvailability, {
        model: "gfs",
        variable: "temperature_2m",
        validTime: "",
      })
    ).toBeNull();
  });

  it("returns null for incomplete, invalid, or absent selections", () => {
    expect(resolveSpatialLayer(null, null)).toBeNull();
    expect(
      resolveSpatialLayer(mockAvailability, {
        model: "nonexistent",
        variable: "temperature_2m",
      })
    ).toBeNull();
    expect(
      resolveSpatialLayer(mockAvailability, {
        model: "gfs",
        variable: "nonexistent-variable",
      })
    ).toBeNull();
  });
});
