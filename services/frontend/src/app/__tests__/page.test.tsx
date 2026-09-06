import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";

import HomePage from "@/app/page";
import { ForecastSelectionProvider } from "@/context/forecast-selection";
import { SelectedLocationProvider } from "@/context/selected-location";
import type { SpatialLayer } from "@/lib/api/types";

let lastRenderedLayer: SpatialLayer | null = null;

// The real WeatherMap uses `next/dynamic(..., { ssr: false })`. Mock to record props.
jest.mock("../../components/map/WeatherMap", () => {
  function WeatherMapStub({ layer }: { layer: SpatialLayer | null }) {
    lastRenderedLayer = layer;
    return <div data-testid="weather-map" data-layer-lead={layer?.lead_time_hours} />;
  }
  return { __esModule: true, default: WeatherMapStub, WeatherMap: WeatherMapStub };
});

function renderPage() {
  return render(
    <ForecastSelectionProvider>
      <SelectedLocationProvider>
        <HomePage />
      </SelectedLocationProvider>
    </ForecastSelectionProvider>
  );
}

const mockFetch = jest.fn<Promise<Response>, [RequestInfo | URL, RequestInit?]>();

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(body),
  } as Response;
}

const searchResult = {
  id: "city_aspen",
  object: "city",
  name: "Aspen",
  region: "Colorado",
  country: "USA",
  elevation_m: null,
  latitude: 38.19,
  longitude: -106.82,
};

const availabilityPayload = {
  object: "forecast_availability",
  data: {
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
                lead_time_hours: [0, 6, 12, 18],
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
  },
  has_more: false,
  next_cursor: null,
};

function routeFetch(input: RequestInfo | URL) {
  const url = String(input);
  if (url.startsWith("/v1/forecast/availability")) {
    return Promise.resolve(jsonResponse(availabilityPayload));
  }
  if (url.startsWith("/v1/models")) {
    return Promise.resolve(
      jsonResponse({
        object: "list",
        data: [
          {
            id: "gfs",
            object: "model",
            name: "Global Forecast System",
            center_id: "noaa",
            is_ensemble: false,
            resolution_km: 25,
          },
        ],
        has_more: false,
        next_cursor: null,
      })
    );
  }
  if (url.startsWith("/v1/variables")) {
    return Promise.resolve(
      jsonResponse({
        object: "list",
        data: [
          { id: "temperature_2m", object: "variable", name: "2-Meter Temperature", unit: "°C" },
          {
            id: "precipitation_rate",
            object: "variable",
            name: "Precipitation Rate",
            unit: "mm/h",
          },
        ],
        has_more: false,
        next_cursor: null,
      })
    );
  }
  if (url.startsWith("/v1/search")) {
    return Promise.resolve(
      jsonResponse({
        object: "list",
        data: [searchResult],
        has_more: false,
        next_cursor: null,
      })
    );
  }
  if (url.startsWith("/v1/points")) {
    return Promise.resolve(
      jsonResponse({
        object: "point_forecast",
        data: {
          location: {
            latitude: 38.19,
            longitude: -106.82,
            elevation_m: null,
            resolved_via: "city",
          },
          generated_at: "2026-07-21T00:00:00Z",
          model: "gfs",
          forecasts: [
            { lead_time_hours: 0, valid_time: "2026-07-21T00:00:00Z", temperature_2m: 13 },
            { lead_time_hours: 6, valid_time: "2026-07-21T06:00:00Z", temperature_2m: 15 },
          ],
        },
        has_more: false,
        next_cursor: null,
      })
    );
  }
  if (url.startsWith("/v1/ensembles")) {
    return Promise.resolve(
      jsonResponse({
        object: "ensemble_statistics",
        data: {
          model: "gfs",
          lead_time_hours: 0,
          member_count: 1,
          statistics: {
            mean: 13,
            median: 13,
            spread: 0,
            p10: 13,
            p25: 13,
            p50: 13,
            p75: 13,
            p90: 13,
          },
        },
        has_more: false,
        next_cursor: null,
      })
    );
  }
  return Promise.resolve(
    jsonResponse({ object: "list", data: [], has_more: false, next_cursor: null })
  );
}

beforeEach(() => {
  lastRenderedLayer = null;
  mockFetch.mockReset();
  mockFetch.mockImplementation(routeFetch);
  globalThis.fetch = mockFetch as unknown as typeof fetch;
});

describe("HomePage", () => {
  it("renders the header, layer controls, and search after loading availability", async () => {
    renderPage();

    expect(screen.getByRole("heading", { name: "Weather Platform" })).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByLabelText("Model")).toBeInTheDocument();
    });
    await waitFor(() => {
      expect(screen.getByLabelText(/Search for a city/)).toBeInTheDocument();
    });

    await act(async () => {});
    await waitFor(() => {
      expect(screen.getByText("Global Forecast System")).toBeInTheDocument();
    });
  });

  it("fetches availability on load and resolves layer synchronously without /v1/maps fetch", async () => {
    renderPage();

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith("/v1/forecast/availability", expect.any(Object));
    });

    // The map is immediately installed with the synchronous layer descriptor
    await waitFor(() => {
      expect(lastRenderedLayer).not.toBeNull();
      expect(lastRenderedLayer?.tile_url_template).toContain("/v1/maps/gfs/temperature_2m/");
    });

    // No redundant /v1/maps metadata roundtrip occurs
    const mapsCalls = mockFetch.mock.calls.filter((call) =>
      String(call[0]).startsWith("/v1/maps?")
    );
    expect(mapsCalls.length).toBe(0);
  });

  it("shows the Valid Time control derived from availability", async () => {
    renderPage();

    await waitFor(() => {
      expect(screen.getByLabelText("Valid time")).toBeInTheDocument();
    });
    expect(screen.queryByLabelText("Initial time")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Lead time")).not.toBeInTheDocument();
  });

  it("synchronously updates the map layer when valid time changes", async () => {
    renderPage();

    await waitFor(() => {
      expect(screen.getByLabelText("Valid time")).toBeInTheDocument();
    });

    // Change valid time
    const validSelect = (await screen.findByLabelText("Valid time")) as HTMLSelectElement;
    const optionEls = validSelect.querySelectorAll("option");
    if (optionEls.length > 1) {
      const nextVal = optionEls[1].value;
      fireEvent.change(validSelect, { target: { value: nextVal } });
      expect(lastRenderedLayer?.valid_time).toBe(nextVal);
    }
  });

  it("synchronously updates the map layer when variable changes", async () => {
    renderPage();

    await waitFor(() => {
      expect(screen.getByLabelText("Variable")).toBeInTheDocument();
    });

    // Change variable to precipitation_rate
    fireEvent.change(screen.getByLabelText("Variable"), {
      target: { value: "precipitation_rate" },
    });

    expect(lastRenderedLayer?.legend.unit).toBe("mm/h");
    expect(lastRenderedLayer?.tile_url_template).toContain("precipitation_rate");
  });

  it("synchronously updates the map layer when model changes (GFS -> GEFS)", async () => {
    renderPage();

    await waitFor(() => {
      expect(screen.getByLabelText("Model")).toBeInTheDocument();
    });

    // Change model to GEFS
    fireEvent.change(screen.getByLabelText("Model"), { target: { value: "gefs" } });

    expect(lastRenderedLayer?.tile_url_template).toContain("/v1/maps/gefs/");
  });

  it("selecting a search result opens the forecast dashboard and fetches /v1/points", async () => {
    renderPage();

    const input = await screen.findByLabelText(/Search for a city/);
    fireEvent.change(input, { target: { value: "Aspen" } });
    fireEvent.focus(input);

    const option = await screen.findByText("Aspen");
    fireEvent.mouseDown(option);

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith(
        "/v1/points?models=gfs&units=metric&city_id=city_aspen",
        expect.any(Object)
      );
    });
    // The dashboard is present.
    await waitFor(() => {
      expect(screen.getByText("Hourly Forecast")).toBeInTheDocument();
    });
  });
});
