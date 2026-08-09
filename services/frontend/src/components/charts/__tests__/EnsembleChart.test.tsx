import { render, screen } from "@testing-library/react";

import { EnsembleChart } from "@/components/charts/EnsembleChart";
import type { EnsembleStatisticsData } from "@/lib/api/types";

jest.mock("recharts", () => {
  const actual = jest.requireActual("recharts");
  return {
    ...actual,
    ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
      <div data-testid="responsive" style={{ width: 640, height: 192 }}>
        {children}
      </div>
    ),
  };
});

const byLead = new Map<number, EnsembleStatisticsData>([
  [
    0,
    {
      model: "gefs",
      lead_time_hours: 0,
      member_count: 5,
      statistics: { mean: 10, median: 10, spread: 2, p10: 7, p25: 9, p50: 10, p75: 11, p90: 13 },
    },
  ],
  [
    6,
    {
      model: "gefs",
      lead_time_hours: 6,
      member_count: 5,
      statistics: { mean: 13, median: 13, spread: 2, p10: 10, p25: 12, p50: 13, p75: 14, p90: 16 },
    },
  ],
]);

describe("EnsembleChart", () => {
  it("renders a percentile-fan labeled chart for the variable", () => {
    render(<EnsembleChart byLead={byLead} variableLabel="Temperature (2 m)" />);

    expect(
      screen.getByRole("img", {
        name: "Temperature (2 m) ensemble percentile fan over lead time",
      })
    ).toBeInTheDocument();
    expect(screen.getByText(/Temperature \(2 m\) — percentile range/)).toBeInTheDocument();
    expect(screen.getByText(/P10–P90 band/)).toBeInTheDocument();
  });
});
