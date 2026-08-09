import { render, screen } from "@testing-library/react";

import { EnsembleDistribution } from "@/components/charts/EnsembleDistribution";
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

const withMembers: EnsembleStatisticsData = {
  model: "gefs",
  lead_time_hours: 6,
  member_count: 5,
  statistics: {
    mean: 17.5,
    median: 17.5,
    spread: 3.16,
    p10: 13.9,
    p25: 15.5,
    p50: 17.5,
    p75: 19.5,
    p90: 21.1,
  },
  members: [15.5, 17.5, 19.5, 21.5, 23.5],
};

const withoutMembers: EnsembleStatisticsData = {
  model: "gefs",
  lead_time_hours: 6,
  member_count: 5,
  statistics: {
    mean: 17.5,
    median: 17.5,
    spread: 3.16,
    p10: 13.9,
    p25: 15.5,
    p50: 17.5,
    p75: 19.5,
    p90: 21.1,
  },
};

const baseProps = {
  selectedLead: 6,
  variableLabel: "Temperature (2 m)",
};

describe("EnsembleDistribution", () => {
  it("renders a histogram and member rug from raw members when present", () => {
    render(
      <EnsembleDistribution {...baseProps} data={withMembers} status="success" error={null} />
    );

    expect(
      screen.getByRole("img", { name: /Histogram of 5 ensemble members/ })
    ).toBeInTheDocument();
    expect(
      screen.getByRole("img", { name: /Member values for Temperature \(2 m\)/ })
    ).toBeInTheDocument();
    expect(screen.getByText(/Member distribution · \+6h/)).toBeInTheDocument();
    expect(screen.getByText("Min")).toBeInTheDocument();
    expect(screen.getByText("Max")).toBeInTheDocument();
  });

  it("shows an honest unavailable state when members are absent", () => {
    render(
      <EnsembleDistribution {...baseProps} data={withoutMembers} status="success" error={null} />
    );

    expect(screen.getByText(/returned no raw member values/)).toBeInTheDocument();
    // No fabricated histogram is rendered.
    expect(screen.queryByRole("img", { name: /Histogram of/ })).not.toBeInTheDocument();
  });

  it("shows a loading state while fetching", () => {
    render(<EnsembleDistribution {...baseProps} data={null} status="loading" error={null} />);
    expect(screen.getByText(/Loading ensemble distribution…/)).toBeInTheDocument();
  });

  it("shows an error state on request failure", () => {
    render(
      <EnsembleDistribution
        {...baseProps}
        data={null}
        status="error"
        error="Failed to load the ensemble distribution."
      />
    );
    expect(screen.getByRole("alert")).toHaveTextContent(
      "Failed to load the ensemble distribution."
    );
  });

  it("shows an unavailable state when there is no data", () => {
    render(<EnsembleDistribution {...baseProps} data={null} status="success" error={null} />);
    expect(screen.getByText(/No ensemble distribution available for \+6h/)).toBeInTheDocument();
  });
});
