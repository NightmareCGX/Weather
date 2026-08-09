import { act, render, screen, waitFor } from "@testing-library/react";

import HomePage from "@/app/page";
import { MapConfigProvider } from "@/context/map-config";

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
      <HomePage />
    </MapConfigProvider>
  );
}

const mockFetch = jest.fn<Promise<Response>, [RequestInfo | URL, RequestInit?]>();

function jsonResponse(body: unknown): Response {
  return { ok: true, status: 200, json: () => Promise.resolve(body) } as Response;
}

beforeEach(() => {
  mockFetch.mockReset();
  mockFetch.mockImplementation((input: RequestInfo | URL) => {
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
    return Promise.resolve(
      jsonResponse({ object: "list", data: [], has_more: false, next_cursor: null })
    );
  });
  globalThis.fetch = mockFetch as unknown as typeof fetch;
});

describe("HomePage", () => {
  it("renders the header and layer controls after loading models and layer metadata", async () => {
    renderPage();

    expect(screen.getByRole("heading", { name: "Weather Platform" })).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByLabelText("Model")).toBeInTheDocument();
    });

    await act(async () => {});
    await waitFor(() => {
      expect(screen.getByText("Global Forecast System")).toBeInTheDocument();
    });
  });

  it("fetches /v1/models and /v1/maps metadata", async () => {
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
});
