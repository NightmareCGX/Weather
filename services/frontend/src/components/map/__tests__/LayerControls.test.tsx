import { render, screen } from "@testing-library/react";

import { LayerControls } from "@/components/map/LayerControls";
import { ForecastSelectionProvider } from "@/context/forecast-selection";

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
});
