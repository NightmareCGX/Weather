import { renderHook } from "@testing-library/react";

import { useMapLayer } from "@/hooks/useMapLayer";
import { useForecastSelection } from "@/context/forecast-selection";
import type { ForecastSelection } from "@/lib/forecast/availability";
import type { ForecastAvailability, SpatialLayer } from "@/lib/api/types";

jest.mock("../../context/forecast-selection");

const mockUseForecastSelection = useForecastSelection as jest.MockedFunction<
  typeof useForecastSelection
>;

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
              lead_time_hours: [0, 6, 12, 18, 24],
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
        {
          id: "precipitation_rate",
          name: "Precipitation Rate",
          unit: "mm/h",
          initial_times: [
            {
              value: "2026-08-13T00:00:00Z",
              lead_time_hours: [0, 6, 12],
            },
          ],
          layer: {
            tile_url_template:
              "/v1/maps/gfs/precipitation_rate/surface/{z}/{x}/{y}.png?lead_time_hours={lead_time_hours}&initial_time={initial_time}",
            min_zoom: 0,
            max_zoom: 9,
            legend: {
              unit: "mm/h",
              stops: [
                [0, "#ffffff"],
                [50, "#0000ff"],
              ],
            },
          },
        },
      ],
    },
    {
      id: "gefs",
      name: "Global Ensemble Forecast System",
      is_ensemble: true,
      variables: [
        {
          id: "temperature_2m",
          name: "2-Meter Temperature",
          unit: "°C",
          initial_times: [
            {
              value: "2026-08-13T00:00:00Z",
              lead_time_hours: [0, 6, 12, 18, 24],
            },
          ],
          layer: {
            tile_url_template:
              "/v1/maps/gefs/temperature_2m/surface/{z}/{x}/{y}.png?lead_time_hours={lead_time_hours}&initial_time={initial_time}",
            min_zoom: 0,
            max_zoom: 9,
            legend: {
              unit: "°C",
              stops: [
                [-40, "#313695"],
                [45, "#a50026"],
              ],
            },
          },
        },
      ],
    },
  ],
};

const selectionLead6: ForecastSelection = {
  model: "gfs",
  variable: "temperature_2m",
  initialTime: "2026-08-13T00:00:00Z",
  leadTimeHours: 6,
};

const selectionLead12: ForecastSelection = {
  model: "gfs",
  variable: "temperature_2m",
  initialTime: "2026-08-13T00:00:00Z",
  leadTimeHours: 12,
};

const selectionPrecip: ForecastSelection = {
  model: "gfs",
  variable: "precipitation_rate",
  initialTime: "2026-08-13T00:00:00Z",
  leadTimeHours: 6,
};

const selection06Z: ForecastSelection = {
  model: "gfs",
  variable: "temperature_2m",
  initialTime: "2026-08-13T06:00:00Z",
  leadTimeHours: 6,
};

const selectionGefs: ForecastSelection = {
  model: "gefs",
  variable: "temperature_2m",
  initialTime: "2026-08-13T00:00:00Z",
  leadTimeHours: 12,
};

function mockContextValue(
  selection: ForecastSelection | null,
  options?: {
    availability?: ForecastAvailability | null;
    status?: "idle" | "loading" | "success" | "error";
    error?: string | null;
  }
) {
  return {
    availability: options?.availability !== undefined ? options.availability : mockAvailability,
    status: options?.status ?? "success",
    error: options?.error ?? null,
    selection,
    validTime: null,
    options: {
      models: [],
      model: null,
      variables: [],
      initialTimes: [],
      variable: null,
      initialTime: null,
      leadTimes: [],
    },
    setModel: jest.fn(),
    setVariable: jest.fn(),
    setInitialTime: jest.fn(),
    setLeadTimeHours: jest.fn(),
    retry: jest.fn(),
  };
}

describe("useMapLayer (synchronous authoritative layer resolution)", () => {
  it("returns null when selection is null", () => {
    mockUseForecastSelection.mockReturnValue(mockContextValue(null));
    const { result } = renderHook(() => useMapLayer());

    expect(result.current.layer).toBeNull();
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it("returns null and loading=true while availability is loading", () => {
    mockUseForecastSelection.mockReturnValue(
      mockContextValue(null, { availability: null, status: "loading" })
    );
    const { result } = renderHook(() => useMapLayer());

    expect(result.current.layer).toBeNull();
    expect(result.current.loading).toBe(true);
    expect(result.current.error).toBeNull();
  });

  it("surfaces availability errors", () => {
    mockUseForecastSelection.mockReturnValue(
      mockContextValue(null, {
        availability: null,
        status: "error",
        error: "Unable to load forecast data.",
      })
    );
    const { result } = renderHook(() => useMapLayer());

    expect(result.current.layer).toBeNull();
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBe("Unable to load forecast data.");
  });

  it("synchronously resolves authoritative SpatialLayer for valid-time selection", () => {
    const validTimeSelection: ForecastSelection = {
      model: "gfs",
      variable: "temperature_2m",
      validTime: "2026-08-13T06:00:00Z",
    };
    mockUseForecastSelection.mockReturnValue(mockContextValue(validTimeSelection));
    const { result } = renderHook(() => useMapLayer());

    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
    expect(result.current.layer?.valid_time).toBe("2026-08-13T06:00:00Z");
    expect(result.current.layer?.tile_url_template).toContain(
      "valid_time=2026-08-13T06%3A00%3A00Z"
    );
  });

  it("synchronously resolves authoritative SpatialLayer for initial selection", () => {
    mockUseForecastSelection.mockReturnValue(mockContextValue(selectionLead6));
    const { result } = renderHook(() => useMapLayer());

    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
    expect(result.current.layer).toEqual<SpatialLayer>({
      tile_url_template:
        "/v1/maps/gfs/temperature_2m/surface/{z}/{x}/{y}.png?lead_time_hours=6&initial_time=2026-08-13T00%3A00%3A00Z",
      min_zoom: 0,
      max_zoom: 9,
      lead_time_hours: 6,
      legend: {
        unit: "°C",
        stops: [
          [-40, "#0000ff"],
          [40, "#ff0000"],
        ],
      },
    });
  });

  it("synchronously transitions on lead time change (+6h -> +12h) without metadata network delay", () => {
    mockUseForecastSelection.mockReturnValue(mockContextValue(selectionLead6));
    const { result, rerender } = renderHook(() => useMapLayer());

    expect(result.current.layer?.lead_time_hours).toBe(6);

    // Transition to lead 12h
    mockUseForecastSelection.mockReturnValue(mockContextValue(selectionLead12));
    rerender();

    // Immediately authoritative without waiting for an HTTP roundtrip
    expect(result.current.layer?.lead_time_hours).toBe(12);
    expect(result.current.layer?.tile_url_template).toContain("lead_time_hours=12");
  });

  it("synchronously transitions on variable change (temperature_2m -> precipitation_rate)", () => {
    mockUseForecastSelection.mockReturnValue(mockContextValue(selectionLead6));
    const { result, rerender } = renderHook(() => useMapLayer());

    expect(result.current.layer?.legend.unit).toBe("°C");

    // Transition to precipitation
    mockUseForecastSelection.mockReturnValue(mockContextValue(selectionPrecip));
    rerender();

    expect(result.current.layer?.legend.unit).toBe("mm/h");
    expect(result.current.layer?.tile_url_template).toContain("precipitation_rate");
  });

  it("synchronously transitions on initial time change (00Z -> 06Z)", () => {
    mockUseForecastSelection.mockReturnValue(mockContextValue(selectionLead6));
    const { result, rerender } = renderHook(() => useMapLayer());

    expect(result.current.layer?.tile_url_template).toContain("2026-08-13T00%3A00%3A00Z");

    // Transition to 06Z cycle
    mockUseForecastSelection.mockReturnValue(mockContextValue(selection06Z));
    rerender();

    expect(result.current.layer?.tile_url_template).toContain("2026-08-13T06%3A00%3A00Z");
  });

  it("synchronously transitions on model change (GFS -> GEFS)", () => {
    mockUseForecastSelection.mockReturnValue(mockContextValue(selectionLead6));
    const { result, rerender } = renderHook(() => useMapLayer());

    expect(result.current.layer?.tile_url_template).toContain("/v1/maps/gfs/");

    // Transition to GEFS
    mockUseForecastSelection.mockReturnValue(mockContextValue(selectionGefs));
    rerender();

    expect(result.current.layer?.tile_url_template).toContain("/v1/maps/gefs/");
    expect(result.current.layer?.lead_time_hours).toBe(12);
  });

  it("leaves only C authoritative in rapid A -> B -> C transitions with zero lag", () => {
    mockUseForecastSelection.mockReturnValue(mockContextValue(selectionLead6));
    const { result, rerender } = renderHook(() => useMapLayer());

    expect(result.current.layer?.lead_time_hours).toBe(6);

    // Rapid switch B
    mockUseForecastSelection.mockReturnValue(mockContextValue(selectionLead12));
    rerender();
    expect(result.current.layer?.lead_time_hours).toBe(12);

    // Rapid switch C
    mockUseForecastSelection.mockReturnValue(mockContextValue(selectionGefs));
    rerender();
    expect(result.current.layer?.lead_time_hours).toBe(12);
    expect(result.current.layer?.tile_url_template).toContain("/v1/maps/gefs/");
  });

  it("returns null when selection refers to an unavailable lead time", () => {
    const invalidSelection: ForecastSelection = {
      model: "gfs",
      variable: "temperature_2m",
      initialTime: "2026-08-13T00:00:00Z",
      leadTimeHours: 999, // not in availability
    };
    mockUseForecastSelection.mockReturnValue(mockContextValue(invalidSelection));
    const { result } = renderHook(() => useMapLayer());

    expect(result.current.layer).toBeNull();
  });
});
