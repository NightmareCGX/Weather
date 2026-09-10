import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";

import HomePage from "@/app/page";
import { ForecastSelectionProvider } from "@/context/forecast-selection";
import {
  SelectedLocationProvider,
  _resetStartupLocationPromiseForTesting,
} from "@/context/selected-location";
import type { ApproximateStartupLocation, SelectedLocation, SpatialLayer } from "@/lib/api/types";

let lastRenderedLayer: SpatialLayer | null = null;
let lastRenderedLocation: SelectedLocation | null = null;
let lastRenderedApproximateLocation: ApproximateStartupLocation | null = null;
let lastOnSelect: ((loc: SelectedLocation) => void) | null = null;

// The real WeatherMap uses `next/dynamic(..., { ssr: false })`. Mock to record props.
jest.mock("../../components/map/WeatherMap", () => {
  function WeatherMapStub({
    layer,
    selectedLocation,
    approximateLocation,
    onSelect,
    onLocate,
  }: {
    layer: SpatialLayer | null;
    selectedLocation: SelectedLocation | null;
    approximateLocation?: ApproximateStartupLocation | null;
    onSelect: (loc: SelectedLocation) => void;
    onLocate?: () => void;
  }) {
    lastRenderedLayer = layer;
    lastRenderedLocation = selectedLocation;
    lastRenderedApproximateLocation = approximateLocation ?? null;
    lastOnSelect = onSelect;
    return (
      <div
        data-testid="weather-map"
        data-layer-lead={layer?.lead_time_hours}
        data-has-location={selectedLocation !== null}
        data-has-approximate-location={
          approximateLocation !== null && approximateLocation !== undefined
        }
      >
        {onLocate && (
          <button type="button" aria-label="Locate me" onClick={onLocate}>
            Locate me
          </button>
        )}
      </div>
    );
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

const boulderResult: SelectedLocation = {
  id: "city_boulder",
  object: "city",
  name: "Boulder",
  region: "Colorado",
  country: "USA",
  elevation_m: 1624,
  latitude: 40.015,
  longitude: -105.27,
  resolvedVia: "city",
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
    const q = new URL(url, "http://localhost").searchParams.get("q") ?? "";
    const data = q.toLowerCase().includes("boulder") ? [boulderResult] : [searchResult];
    return Promise.resolve(
      jsonResponse({
        object: "list",
        data,
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
  if (url.includes("/locate")) {
    return Promise.resolve(
      jsonResponse({
        latitude: 39.7392,
        longitude: -104.9903,
        city: "Denver",
        region: "Colorado",
        country: "US",
        approximate: true,
      })
    );
  }
  return Promise.resolve(
    jsonResponse({ object: "list", data: [], has_more: false, next_cursor: null })
  );
}

beforeEach(() => {
  lastRenderedLayer = null;
  lastRenderedLocation = null;
  lastRenderedApproximateLocation = null;
  lastOnSelect = null;
  _resetStartupLocationPromiseForTesting();
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

  it("clicking Close (X) clears selected location, removes sidebar and marker", async () => {
    renderPage();

    const input = await screen.findByLabelText(/Search for a city/);
    fireEvent.change(input, { target: { value: "Aspen" } });
    fireEvent.focus(input);

    const option = await screen.findByText("Aspen");
    fireEvent.mouseDown(option);

    await waitFor(() => {
      expect(screen.getByText("Hourly Forecast")).toBeInTheDocument();
    });
    expect(lastRenderedLocation?.name).toBe("Aspen");

    const closeBtn = screen.getByRole("button", { name: "Close forecast panel" });
    fireEvent.click(closeBtn);

    await waitFor(() => {
      expect(screen.queryByText("Hourly Forecast")).not.toBeInTheDocument();
    });
    expect(lastRenderedLocation).toBeNull();
  });

  it("clicking Collapse minimizes forecast panel while keeping selectedLocation and marker", async () => {
    renderPage();

    const input = await screen.findByLabelText(/Search for a city/);
    fireEvent.change(input, { target: { value: "Aspen" } });
    fireEvent.focus(input);

    const option = await screen.findByText("Aspen");
    fireEvent.mouseDown(option);

    await waitFor(() => {
      expect(screen.getByText("Hourly Forecast")).toBeInTheDocument();
    });
    expect(lastRenderedLocation?.name).toBe("Aspen");

    const collapseBtn = screen.getByRole("button", { name: "Collapse forecast panel" });
    expect(collapseBtn).toHaveAttribute("aria-expanded", "true");
    expect(collapseBtn).toHaveAttribute("aria-controls", "forecast-panel-content");

    fireEvent.click(collapseBtn);

    const contentWrapper = document.getElementById("forecast-panel-content");
    expect(contentWrapper).toHaveClass("hidden");
    expect(lastRenderedLocation?.name).toBe("Aspen");

    const expandBtn = screen.getByRole("button", { name: "Expand forecast panel" });
    expect(expandBtn).toHaveAttribute("aria-expanded", "false");
    expect(expandBtn).toHaveAttribute("aria-controls", "forecast-panel-content");
  });

  it("clicking Expand restores forecast panel without refetching /v1/points", async () => {
    renderPage();

    const input = await screen.findByLabelText(/Search for a city/);
    fireEvent.change(input, { target: { value: "Aspen" } });
    fireEvent.focus(input);

    const option = await screen.findByText("Aspen");
    fireEvent.mouseDown(option);

    await waitFor(() => {
      expect(screen.getByText("Hourly Forecast")).toBeInTheDocument();
    });

    const initialPointsCalls = mockFetch.mock.calls.filter((call) =>
      String(call[0]).startsWith("/v1/points")
    ).length;
    expect(initialPointsCalls).toBeGreaterThan(0);

    // Collapse panel
    const collapseBtn = screen.getByRole("button", { name: "Collapse forecast panel" });
    fireEvent.click(collapseBtn);

    const contentWrapper = document.getElementById("forecast-panel-content");
    expect(contentWrapper).toHaveClass("hidden");

    // Expand panel
    const expandBtn = screen.getByRole("button", { name: "Expand forecast panel" });
    fireEvent.click(expandBtn);

    expect(contentWrapper).not.toHaveClass("hidden");
    expect(screen.getByText("Hourly Forecast")).toBeInTheDocument();
    expect(lastRenderedLocation?.name).toBe("Aspen");

    // Verify no additional /v1/points fetch was made merely because of collapse/expand
    const afterPointsCalls = mockFetch.mock.calls.filter((call) =>
      String(call[0]).startsWith("/v1/points")
    ).length;
    expect(afterPointsCalls).toBe(initialPointsCalls);
  });

  it("selecting a new location while collapsed automatically expands the panel", async () => {
    renderPage();

    const input = await screen.findByLabelText(/Search for a city/);
    fireEvent.change(input, { target: { value: "Aspen" } });
    fireEvent.focus(input);

    const option = await screen.findByText("Aspen");
    fireEvent.mouseDown(option);

    await waitFor(() => {
      expect(screen.getByText("Hourly Forecast")).toBeInTheDocument();
    });

    // Collapse panel
    const collapseBtn = screen.getByRole("button", { name: "Collapse forecast panel" });
    fireEvent.click(collapseBtn);
    expect(document.getElementById("forecast-panel-content")).toHaveClass("hidden");

    // Simulate new location selection via map onSelect
    act(() => {
      lastOnSelect?.(boulderResult);
    });

    await waitFor(() => {
      expect(lastRenderedLocation?.name).toBe("Boulder");
    });

    // Panel should automatically be expanded
    const newCollapseBtn = screen.getByRole("button", { name: "Collapse forecast panel" });
    expect(newCollapseBtn).toHaveAttribute("aria-expanded", "true");
    expect(document.getElementById("forecast-panel-content")).not.toHaveClass("hidden");
  });

  it("clicking Locate Me successfully acquires location and opens forecast panel", async () => {
    const mockGetCurrentPosition = jest.fn();
    Object.defineProperty(globalThis, "navigator", {
      value: {
        geolocation: {
          getCurrentPosition: mockGetCurrentPosition,
        },
      },
      writable: true,
      configurable: true,
    });

    renderPage();

    const locateBtn = await screen.findByRole("button", { name: "Locate me" });
    expect(locateBtn).toBeInTheDocument();

    fireEvent.click(locateBtn);
    expect(mockGetCurrentPosition).toHaveBeenCalledTimes(1);

    const [successCb] = mockGetCurrentPosition.mock.calls[0];
    act(() => {
      successCb({
        coords: {
          latitude: 39.7392,
          longitude: -104.9903,
        },
      });
    });

    await waitFor(() => {
      expect(screen.getByText("Hourly Forecast")).toBeInTheDocument();
      expect(screen.getByText("39.7392, -104.9903")).toBeInTheDocument();
    });
  });

  it("Locate Me permission denial displays non-blocking alert without calling /v1/locate", async () => {
    const mockGetCurrentPosition = jest.fn();
    Object.defineProperty(globalThis, "navigator", {
      value: {
        geolocation: {
          getCurrentPosition: mockGetCurrentPosition,
        },
      },
      writable: true,
      configurable: true,
    });

    renderPage();

    const locateBtn = await screen.findByRole("button", { name: "Locate me" });
    // Count locate calls before Locate Me click
    const initialLocateCalls = mockFetch.mock.calls.filter((call) =>
      String(call[0]).includes("/locate")
    ).length;

    fireEvent.click(locateBtn);

    const [, errorCb] = mockGetCurrentPosition.mock.calls[0];
    act(() => {
      errorCb({
        code: 1, // PERMISSION_DENIED
        PERMISSION_DENIED: 1,
        message: "User denied Geolocation",
      });
    });

    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent("Location access denied");
    });

    // Strict privacy invariant: verify /v1/locate was NOT called as a fallback for Locate Me
    const afterLocateCalls = mockFetch.mock.calls.filter((call) =>
      String(call[0]).includes("/locate")
    ).length;
    expect(afterLocateCalls).toBe(initialLocateCalls);
  });

  it("passes startup approximate location to WeatherMap while leaving selectedLocation null", async () => {
    renderPage();

    await waitFor(() => {
      expect(lastRenderedApproximateLocation).not.toBeNull();
    });

    expect(lastRenderedApproximateLocation).toMatchObject({
      latitude: 39.7392,
      city: "Denver",
      source: "ip",
    });
    expect(lastRenderedLocation).toBeNull();
    expect(screen.queryByText("Hourly Forecast")).not.toBeInTheDocument();
  });
});
