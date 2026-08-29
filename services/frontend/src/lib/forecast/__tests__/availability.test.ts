import {
  buildForecastOptions,
  defaultInitialTime,
  defaultLeadTime,
  defaultModel,
  defaultVariable,
  findInitialTime,
  findModel,
  findVariable,
  resolveSpatialLayer,
  resolveValidTime,
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
  it("resolves default model, variable, initial time, and lead time", () => {
    expect(defaultModel(mockAvailability)).toBe("gfs");
    const model = findModel(mockAvailability, "gfs");
    expect(defaultVariable(model)).toBe("temperature_2m");
    const variable = findVariable(model, "temperature_2m");
    expect(defaultInitialTime(variable)).toBe("2026-08-13T00:00:00Z");
    const initialTime = findInitialTime(variable, "2026-08-13T00:00:00Z");
    expect(defaultLeadTime(initialTime)).toBe(0);
  });

  it("builds cascading forecast options", () => {
    const options = buildForecastOptions(mockAvailability, {
      model: "gfs",
      variable: "temperature_2m",
      initialTime: "2026-08-13T00:00:00Z",
      leadTimeHours: 6,
    });
    expect(options.models.length).toBe(1);
    expect(options.model?.id).toBe("gfs");
    expect(options.variables.length).toBe(1);
    expect(options.variable?.id).toBe("temperature_2m");
    expect(options.initialTimes.length).toBe(2);
    expect(options.leadTimes).toEqual([0, 6, 12, 18]);
  });

  it("computes valid time correctly", () => {
    expect(resolveValidTime("2026-08-13T00:00:00Z", 6)).toBe("2026-08-13T06:00:00.000Z");
    expect(resolveValidTime(null, 6)).toBeNull();
    expect(resolveValidTime("2026-08-13T00:00:00Z", null)).toBeNull();
  });
});

describe("resolveSpatialLayer", () => {
  it("synchronously constructs authoritative SpatialLayer from availability descriptor", () => {
    const layer = resolveSpatialLayer(mockAvailability, {
      model: "gfs",
      variable: "temperature_2m",
      initialTime: "2026-08-13T00:00:00Z",
      leadTimeHours: 12,
    });

    expect(layer).not.toBeNull();
    expect(layer?.tile_url_template).toBe(
      "/v1/maps/gfs/temperature_2m/surface/{z}/{x}/{y}.png?lead_time_hours=12&initial_time=2026-08-13T00%3A00%3A00Z"
    );
    expect(layer?.min_zoom).toBe(0);
    expect(layer?.max_zoom).toBe(9);
    expect(layer?.lead_time_hours).toBe(12);
    expect(layer?.legend.unit).toBe("°C");
  });

  it("returns null for incomplete, invalid, or absent selections", () => {
    expect(resolveSpatialLayer(null, null)).toBeNull();
    expect(
      resolveSpatialLayer(mockAvailability, {
        model: "nonexistent",
        variable: "temperature_2m",
        initialTime: "2026-08-13T00:00:00Z",
        leadTimeHours: 6,
      })
    ).toBeNull();
    expect(
      resolveSpatialLayer(mockAvailability, {
        model: "gfs",
        variable: "temperature_2m",
        initialTime: "2026-08-13T00:00:00Z",
        leadTimeHours: 999,
      })
    ).toBeNull();
  });
});
