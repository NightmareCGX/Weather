import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";

import HomePage from "@/app/page";
import { ForecastSelectionProvider } from "@/context/forecast-selection";
import { SelectedLocationProvider } from "@/context/selected-location";

// The real WeatherMap uses `next/dynamic(..., { ssr: false })`, which resolves
// its component asynchronously and renders `ForwardRef(LoadableComponent)`.
// In jsdom that dynamic resolution does not complete to a stable element, so
// the page test mocks the module to a simple stub.
jest.mock("../../components/map/WeatherMap", () => {
  function WeatherMapStub() {
    return <div data-testid="weather-map" />;
  }
  return { __esModule: true, default: WeatherMapStub };
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
                lead_time_hours: [6],
              },
            ],
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
  if (url.startsWith("/v1/maps")) {
    return Promise.resolve(
      jsonResponse({
        object: "spatial_layer",
        data: {
          tile_url_template:
            "/v1/maps/gfs/temperature_2m/surface/{z}/{x}/{y}.png?lead_time_hours=6&initial_time=2026-08-13T00:00:00Z",
          min_zoom: 0,
          max_zoom: 9,
          lead_time_hours: 6,
          legend: { unit: "°C", stops: [[-40, "#0000ff"]] },
        },
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
            { lead_time_hours: 6, valid_time: "2026-07-21T06:00:00Z", temperature_2m: 13 },
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
          lead_time_hours: 6,
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

  it("fetches availability and map metadata on load", async () => {
    renderPage();

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith("/v1/forecast/availability", expect.any(Object));
    });
    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining("/v1/maps?model=gfs"),
        expect.any(Object)
      );
    });
  });

  it("shows the Initial Time and Lead Time controls derived from availability", async () => {
    renderPage();

    await waitFor(() => {
      expect(screen.getByLabelText("Initial time")).toBeInTheDocument();
    });
    expect(screen.getByText("2026-08-13 00Z")).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByLabelText("Lead time")).toBeInTheDocument();
    });
    expect(screen.getByText("+6h")).toBeInTheDocument();
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
