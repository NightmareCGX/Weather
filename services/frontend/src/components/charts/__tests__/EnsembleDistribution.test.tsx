import React from "react";
import { render, screen } from "@testing-library/react";

import { EnsembleDistribution } from "@/components/charts/EnsembleDistribution";
import { histogramBins } from "@/lib/forecast/transform";
import type { EnsembleStatisticsData } from "@/lib/api/types";

jest.mock("recharts", () => {
  const actual = jest.requireActual("recharts");
  return {
    ...actual,
    ResponsiveContainer: ({ children }: { children: React.ReactElement }) => (
      <div data-testid="responsive" style={{ width: 640, height: 192 }}>
        {React.isValidElement(children)
          ? React.cloneElement(children, { width: 640, height: 192 } as React.Attributes)
          : children}
      </div>
    ),
  };
});

const withMembersAndPdf: EnsembleStatisticsData = {
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
  pdf: {
    x: [10.0, 15.0, 20.0, 25.0, 30.0],
    density: [0.001, 0.05, 0.2, 0.05, 0.001],
  },
};

const withMembersNullPdf: EnsembleStatisticsData = {
  model: "gefs",
  lead_time_hours: 6,
  member_count: 5,
  statistics: {
    mean: 20.0,
    median: 20.0,
    spread: 0.0,
    p10: 20.0,
    p25: 20.0,
    p50: 20.0,
    p75: 20.0,
    p90: 20.0,
  },
  members: [20.0, 20.0, 20.0, 20.0, 20.0],
  pdf: null,
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
  it("renders a histogram, PDF line, and member rug from raw members and pdf when present", () => {
    const { container } = render(
      <EnsembleDistribution {...baseProps} data={withMembersAndPdf} status="success" error={null} />
    );

    expect(
      screen.getByRole("img", { name: /Histogram and PDF of 5 ensemble members/ })
    ).toBeInTheDocument();
    expect(
      screen.getByRole("img", { name: /Member values for Temperature \(2 m\)/ })
    ).toBeInTheDocument();
    expect(screen.getByText(/Member distribution · \+6h/)).toBeInTheDocument();
    expect(screen.getByText("Min")).toBeInTheDocument();
    expect(screen.getByText("Max")).toBeInTheDocument();
    expect(screen.getByText(/canonical Gaussian kernel density estimate/)).toBeInTheDocument();

    // Semantic SVG mark assertions for histogram bars
    const expectedBins = histogramBins(withMembersAndPdf.members!).length;
    const bars = container.querySelectorAll(".recharts-rectangle");
    expect(bars.length).toBe(expectedBins);
    bars.forEach((bar) => {
      expect(bar.getAttribute("d")).toMatch(/^M\s*[\d.]+/);
    });

    // Semantic SVG mark assertion for PDF continuous line curve
    const lineCurve = container.querySelector(".recharts-line-curve");
    expect(lineCurve).toBeInTheDocument();
    expect(lineCurve?.getAttribute("d")).toMatch(/^M\s*[\d.]+/);

    // Semantic SVG mark assertion for member rug dots
    const dots = container.querySelectorAll(".recharts-scatter-symbol");
    expect(dots.length).toBe(withMembersAndPdf.members!.length);
  });

  it("handles null pdf gracefully by rendering histogram, dots, and warning note", () => {
    const { container } = render(
      <EnsembleDistribution
        {...baseProps}
        data={withMembersNullPdf}
        status="success"
        error={null}
      />
    );

    expect(
      screen.getByRole("img", { name: /Histogram and PDF of 5 ensemble members/ })
    ).toBeInTheDocument();
    expect(
      screen.getByRole("img", { name: /Member values for Temperature \(2 m\)/ })
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Continuous probability density is unavailable for this lead time/)
    ).toBeInTheDocument();

    // Histogram bars must still render when PDF is null
    const expectedBins = histogramBins(withMembersNullPdf.members!).length;
    const bars = container.querySelectorAll(".recharts-rectangle");
    expect(bars.length).toBe(expectedBins);
    bars.forEach((bar) => {
      expect(bar.getAttribute("d")).toMatch(/^M\s*[\d.]+/);
    });

    // PDF line must be absent
    const lineCurve = container.querySelector(".recharts-line-curve");
    expect(lineCurve).not.toBeInTheDocument();

    // Rug dots must still be present
    const dots = container.querySelectorAll(".recharts-scatter-symbol");
    expect(dots.length).toBe(withMembersNullPdf.members!.length);
  });

  it("shows an honest unavailable state when members are absent", () => {
    render(
      <EnsembleDistribution {...baseProps} data={withoutMembers} status="success" error={null} />
    );

    expect(screen.getByText(/returned no raw member values/)).toBeInTheDocument();
    // No fabricated histogram is rendered.
    expect(screen.queryByRole("img", { name: /Histogram/ })).not.toBeInTheDocument();
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
