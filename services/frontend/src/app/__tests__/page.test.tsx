import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";

import HomePage from "@/app/page";
import { MapConfigProvider } from "@/context/map-config";
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
    <MapConfigProvider>
      <SelectedLocationProvider>
        <HomePage />
      </SelectedLocationProvider>
    </MapConfigProvider>
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

function routeFetch(input: RequestInfo | URL) {
  const url = String(input);
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
            "/v1/maps/gfs/temperature_2m/surface/{z}/{x}/{y}.png?lead_time_hours=12",
          min_zoom: 0,
          max_zoom: 9,
          lead_time_hours: 12,
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
            { lead_time_hours: 0, valid_time: "2026-07-21T00:00:00Z", temperature_2m: 10 },
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
          model: "gefs",
          lead_time_hours: 0,
          member_count: 5,
          statistics: {
            mean: 10,
            median: 10,
            spread: 2,
            p10: 7,
            p25: 9,
            p50: 10,
            p75: 11,
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
  it("renders the header, layer controls, and search after loading model and layer metadata", async () => {
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

  it("fetches /v1/models and /v1/maps on load", async () => {
    renderPage();

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith("/v1/models", expect.any(Object));
    });
    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining("/v1/maps?model=gfs"),
        expect.any(Object)
      );
    });
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
