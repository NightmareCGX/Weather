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
});
