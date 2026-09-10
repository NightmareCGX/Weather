import { act, fireEvent, render, screen } from "@testing-library/react";

import { LayerControls } from "@/components/map/LayerControls";
import { ForecastSelectionProvider, useForecastSelection } from "@/context/forecast-selection";
import {
  SelectedLocationProvider,
  _resetStartupLocationPromiseForTesting,
  useSelectedLocation,
} from "@/context/selected-location";
import type { SelectedLocation } from "@/lib/api/types";

/**
 * The LayerControls component renders from the shared forecast-selection
 * context (which fetches `/v1/forecast/availability`). These tests exercise
 * the data-driven behavior against a mocked fetch.
 */

const mockFetch = jest.fn<Promise<Response>, [RequestInfo | URL, RequestInit?]>();

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(body),
  } as Response;
}

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
          {
            id: "precipitation_rate",
            name: "Precipitation Rate",
            unit: "mm/h",
            initial_times: [
              {
                value: "2026-08-13T00:00:00Z",
                lead_time_hours: [6],
              },
            ],
          },
          {
            id: "relative_humidity_2m",
            name: "Relative Humidity",
            unit: "%",
            initial_times: [
              {
                value: "2026-08-13T00:00:00Z",
                lead_time_hours: [6],
              },
            ],
          },
          {
            id: "wind_gust",
            name: "Wind Gust",
            unit: "km/h",
            initial_times: [
              {
                value: "2026-08-13T00:00:00Z",
                lead_time_hours: [6],
              },
            ],
          },
          {
            id: "visibility",
            name: "Visibility",
            unit: "m",
            initial_times: [
              {
                value: "2026-08-13T00:00:00Z",
                lead_time_hours: [6],
              },
            ],
          },
          {
            id: "snow_depth",
            name: "Snow Depth",
            unit: "m",
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

function renderControls() {
  return render(
    <ForecastSelectionProvider>
      <LayerControls />
    </ForecastSelectionProvider>
  );
}

beforeEach(() => {
  mockFetch.mockReset();
  _resetStartupLocationPromiseForTesting();
  mockFetch.mockImplementation((input: RequestInfo | URL) => {
    const url = String(input);
    if (url.startsWith("/v1/forecast/availability")) {
      return Promise.resolve(jsonResponse(availabilityPayload));
    }
    return Promise.resolve(
      jsonResponse({ object: "list", data: [], has_more: false, next_cursor: null })
    );
  });
  globalThis.fetch = mockFetch as unknown as typeof fetch;
});

describe("LayerControls (data-driven)", () => {
  it("renders model, variable, and valid time controls derived from availability", async () => {
    renderControls();

    expect(await screen.findByLabelText("Model")).toBeInTheDocument();
    expect(screen.getByLabelText("Variable")).toBeInTheDocument();
    expect(screen.getByLabelText("Valid time")).toBeInTheDocument();

    // Initial time and Lead time are removed
    expect(screen.queryByLabelText("Initial time")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Lead time")).not.toBeInTheDocument();

    // Only the database-driven options appear.
    expect(screen.getByText("Global Forecast System")).toBeInTheDocument();
    expect(screen.getByText("2-Meter Temperature")).toBeInTheDocument();
    expect(screen.getByText("Precipitation Rate")).toBeInTheDocument();
    expect(screen.getByText("Relative Humidity")).toBeInTheDocument();
    expect(screen.getByText("Wind Gust")).toBeInTheDocument();
    expect(screen.getByText("Visibility")).toBeInTheDocument();
    expect(screen.getByText("Snow Depth")).toBeInTheDocument();
    expect(screen.getByText("Aug 13, 06:00 UTC")).toBeInTheDocument();
  });

  it("does not render removed lead time or initial time dropdowns", async () => {
    renderControls();

    await screen.findByLabelText("Valid time");
    expect(screen.queryByLabelText("Lead time")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Initial time")).not.toBeInTheDocument();
  });

  it("shows a loading state while availability is being fetched", () => {
    let resolveFetch!: (value: Response) => void;
    mockFetch.mockImplementationOnce(
      () => new Promise<Response>((resolve) => (resolveFetch = resolve))
    );
    renderControls();
    expect(screen.getByText(/Loading forecast options/)).toBeInTheDocument();
    resolveFetch(jsonResponse(availabilityPayload));
  });

  it("renders the controls once availability resolves", async () => {
    let resolveFetch!: (value: Response) => void;
    mockFetch.mockImplementationOnce(
      () => new Promise<Response>((resolve) => (resolveFetch = resolve))
    );
    renderControls();
    expect(screen.getByText(/Loading forecast options/)).toBeInTheDocument();
    resolveFetch(jsonResponse(availabilityPayload));
    expect(await screen.findByLabelText("Model")).toBeInTheDocument();
  });

  it("shows an error state with retry when availability fails", async () => {
    mockFetch.mockRejectedValueOnce(new TypeError("Failed to fetch"));
    renderControls();
    expect(await screen.findByRole("alert")).toHaveTextContent(/Unable to load forecast data/);
    expect(screen.getByText("Retry")).toBeInTheDocument();
  });

  it("shows an empty state when the database has no models", async () => {
    mockFetch.mockImplementationOnce(() =>
      Promise.resolve(
        jsonResponse({
          object: "forecast_availability",
          data: { models: [] },
          has_more: false,
          next_cursor: null,
        })
      )
    );
    renderControls();
    expect(await screen.findByText("No forecast data available.")).toBeInTheDocument();
  });

  describe("selected-location local time display", () => {
    const denverLocation: SelectedLocation = {
      name: "Denver",
      object: "city",
      id: "city_denver",
      resolvedVia: "city",
      latitude: 39.7392,
      longitude: -104.9903,
      elevation_m: 1609,
      region: "Colorado",
      country: "USA",
    };

    const tokyoLocation: SelectedLocation = {
      name: "Tokyo",
      object: "city",
      id: "city_tokyo",
      resolvedVia: "city",
      latitude: 35.6762,
      longitude: 139.6503,
      elevation_m: 40,
      region: "Tokyo",
      country: "Japan",
    };

    function LocationHarness() {
      const { selectLocation, clearSelection } = useSelectedLocation();
      const { selection } = useForecastSelection();
      return (
        <div>
          <button onClick={() => selectLocation(denverLocation)}>Select Denver</button>
          <button onClick={() => selectLocation(tokyoLocation)}>Select Tokyo</button>
          <button onClick={() => clearSelection()}>Clear Location</button>
          <span data-testid="canonical-valid-time">{selection?.validTime ?? "none"}</span>
          <LayerControls />
        </div>
      );
    }

    function renderWithLocation() {
      return render(
        <ForecastSelectionProvider>
          <SelectedLocationProvider>
            <LocationHarness />
          </SelectedLocationProvider>
        </ForecastSelectionProvider>
      );
    }

    it("renders UTC display when no location is selected", async () => {
      renderWithLocation();

      expect(await screen.findByLabelText("Valid time")).toBeInTheDocument();
      const validTimeDisplay = screen.getByTestId("valid-time");
      expect(validTimeDisplay).toHaveTextContent("Valid Aug 13, 06:00 UTC");

      // Dropdown option remains UTC
      expect(screen.getByText("Aug 13, 06:00 UTC")).toBeInTheDocument();
    });

    it("localizes adjacent display for Denver while dropdown remains UTC", async () => {
      renderWithLocation();

      await screen.findByLabelText("Valid time");

      act(() => {
        fireEvent.click(screen.getByText("Select Denver"));
      });

      // Dropdown option MUST remain in UTC
      expect(screen.getByText("Aug 13, 06:00 UTC")).toBeInTheDocument();

      // Adjacent display updates to Denver local time (UTC-6 in Aug -> 00:00 MDT)
      const validTimeDisplay = screen.getByTestId("valid-time");
      expect(validTimeDisplay.textContent).toMatch(/^Valid Aug 13, 00:00 (MDT|GMT-6)$/);

      // Canonical forecast valid time must NOT change
      expect(screen.getByTestId("canonical-valid-time")).toHaveTextContent(
        "2026-08-13T06:00:00.000Z"
      );
    });

    it("localizes adjacent display for Tokyo without changing canonical valid time", async () => {
      renderWithLocation();

      await screen.findByLabelText("Valid time");

      act(() => {
        fireEvent.click(screen.getByText("Select Tokyo"));
      });

      // Dropdown option MUST remain in UTC
      expect(screen.getByText("Aug 13, 06:00 UTC")).toBeInTheDocument();

      // Adjacent display updates to Tokyo local time (UTC+9 in Aug -> 15:00)
      const validTimeDisplay = screen.getByTestId("valid-time");
      expect(validTimeDisplay.textContent).toMatch(/^Valid Aug 13, 15:00 (JST|GMT\+9)$/);

      // Canonical valid time remains untouched
      expect(screen.getByTestId("canonical-valid-time")).toHaveTextContent(
        "2026-08-13T06:00:00.000Z"
      );
    });

    it("restores UTC display when location selection is cleared", async () => {
      renderWithLocation();

      await screen.findByLabelText("Valid time");

      // Select Denver first
      act(() => {
        fireEvent.click(screen.getByText("Select Denver"));
      });
      expect(screen.getByTestId("valid-time").textContent).toMatch(
        /^Valid Aug 13, 00:00 (MDT|GMT-6)$/
      );

      // Clear location
      act(() => {
        fireEvent.click(screen.getByText("Clear Location"));
      });

      // Adjacent display returns to UTC
      expect(screen.getByTestId("valid-time")).toHaveTextContent("Valid Aug 13, 06:00 UTC");
    });

    it("uses startup coarse IP timezone for adjacent label when available and no location is selected", async () => {
      mockFetch.mockImplementation((input: RequestInfo | URL) => {
        const url = String(input);
        if (url.startsWith("/v1/forecast/availability")) {
          return Promise.resolve(jsonResponse(availabilityPayload));
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
      });

      renderWithLocation();

      await screen.findByLabelText("Valid time");

      // Startup coarse IP localization formats adjacent valid label in Mountain Time (MDT / GMT-6)
      const validTimeDisplay = await screen.findByTestId("valid-time");
      expect(validTimeDisplay.textContent).toMatch(/^Valid Aug 13, 00:00 (MDT|GMT-6)$/);

      // Dropdown option MUST still remain in canonical UTC
      expect(screen.getByText("Aug 13, 06:00 UTC")).toBeInTheDocument();

      // Selecting Tokyo overrides startup Denver timezone
      act(() => {
        fireEvent.click(screen.getByText("Select Tokyo"));
      });
      expect(validTimeDisplay.textContent).toMatch(/^Valid Aug 13, 15:00 (JST|GMT\+9)$/);

      // Clearing selection restores startup Denver timezone
      act(() => {
        fireEvent.click(screen.getByText("Clear Location"));
      });
      expect(validTimeDisplay.textContent).toMatch(/^Valid Aug 13, 00:00 (MDT|GMT-6)$/);
    });
  });
});
