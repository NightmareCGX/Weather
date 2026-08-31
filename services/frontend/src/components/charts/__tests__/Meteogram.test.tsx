import { render, screen } from "@testing-library/react";

import { Meteogram } from "@/components/charts/Meteogram";
import type { ForecastEntry } from "@/lib/api/types";

// ResponsiveContainer cannot measure dimensions in jsdom; give it a fixed box
// so the chart children render deterministically.
jest.mock("recharts", () => {
  const actual = jest.requireActual("recharts");
  return {
    ...actual,
    ResponsiveContainer: ({ children, ...props }: { children: React.ReactNode }) => (
      <div data-testid="responsive" style={{ width: 640, height: 192 }} {...props}>
        {children}
      </div>
    ),
  };
});

const temperatureEntries: ForecastEntry[] = [
  { lead_time_hours: 0, valid_time: "2026-07-21T00:00:00Z", temperature_2m: 10 },
  { lead_time_hours: 6, valid_time: "2026-07-21T06:00:00Z", temperature_2m: 13 },
  { lead_time_hours: 12, valid_time: "2026-07-21T12:00:00Z", temperature_2m: 16 },
];

describe("Meteogram", () => {
  it("renders a labeled chart for a temperature variable", () => {
    render(
      <Meteogram
        forecasts={temperatureEntries}
        variableCode="temperature_2m"
        meta={{ name: "Temperature (2 m)", unit: "°C" }}
      />
    );

    expect(
      screen.getByRole("img", { name: "Temperature (2 m) hourly forecast over lead time" })
    ).toBeInTheDocument();
    expect(screen.getByText("Temperature (2 m)")).toBeInTheDocument();
    expect(screen.getByText("°C")).toBeInTheDocument();
  });

  it("renders a labeled chart for precipitation", () => {
    render(
      <Meteogram
        forecasts={[
          { lead_time_hours: 0, valid_time: "2026-07-21T00:00:00Z", precipitation_rate: 0 },
          { lead_time_hours: 6, valid_time: "2026-07-21T06:00:00Z", precipitation_rate: 3 },
        ]}
        variableCode="precipitation_rate"
        meta={{ name: "Precipitation Rate", unit: "mm/h" }}
      />
    );

    expect(
      screen.getByRole("img", { name: "Precipitation Rate hourly forecast over lead time" })
    ).toBeInTheDocument();
  });

  it("renders 3-Hour Precipitation with phase legend and handles lead 0 null value", () => {
    const precipEntries: ForecastEntry[] = [
      {
        lead_time_hours: 0,
        valid_time: "2026-07-21T00:00:00Z",
        precipitation_amount_3h: undefined,
        precipitation_type: "none",
        precipitation_transition: "none",
      },
      {
        lead_time_hours: 3,
        valid_time: "2026-07-21T03:00:00Z",
        precipitation_amount_3h: 4.2,
        precipitation_type: "rain",
        precipitation_transition: "persistent_rain",
      },
      {
        lead_time_hours: 6,
        valid_time: "2026-07-21T06:00:00Z",
        precipitation_amount_3h: 5.1,
        precipitation_type: "mixed",
        precipitation_transition: "rain_to_snow",
      },
    ];

    render(
      <Meteogram
        forecasts={precipEntries}
        variableCode="precipitation_amount_3h"
        meta={{ name: "3-Hour Precipitation", unit: "mm" }}
      />
    );

    expect(
      screen.getByRole("img", { name: "3-Hour Precipitation hourly forecast over lead time" })
    ).toBeInTheDocument();
    expect(screen.getByText("3-Hour Precipitation")).toBeInTheDocument();
    expect(screen.getByText("mm")).toBeInTheDocument();

    // Check phase legend elements
    expect(screen.getByText("Phases:")).toBeInTheDocument();
    expect(screen.getByText("Rain")).toBeInTheDocument();
    expect(screen.getByText("Snow")).toBeInTheDocument();
    expect(screen.getByText("Freezing Rain")).toBeInTheDocument();
    expect(screen.getByText("Ice Pellets")).toBeInTheDocument();
    expect(screen.getByText("Mixed")).toBeInTheDocument();
  });
});
