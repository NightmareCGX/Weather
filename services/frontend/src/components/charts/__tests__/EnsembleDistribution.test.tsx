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
  validTime: "2026-09-10T06:00:00Z",
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
    expect(screen.getByText(/Member distribution · Sep 10, 06:00 UTC/)).toBeInTheDocument();
    expect(screen.queryByText(/\+6h/)).not.toBeInTheDocument();
    expect(screen.getByText("p0.1")).toBeInTheDocument();
    expect(screen.getByText("p99.9")).toBeInTheDocument();
    expect(screen.queryByText("Min")).not.toBeInTheDocument();
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

  it("localizes valid-time header for America/Denver and Asia/Tokyo", () => {
    const { rerender } = render(
      <EnsembleDistribution
        {...baseProps}
        data={withMembersAndPdf}
        status="success"
        error={null}
        timezone="America/Denver"
      />
    );

    // 2026-09-10T06:00:00Z in Denver MDT is Sep 10, 00:00 MDT
    expect(screen.getByText(/Member distribution · Sep 10, 00:00 (MDT|GMT-6)/)).toBeInTheDocument();

    rerender(
      <EnsembleDistribution
        {...baseProps}
        data={withMembersAndPdf}
        status="success"
        error={null}
        timezone="Asia/Tokyo"
      />
    );

    // 2026-09-10T06:00:00Z in Tokyo is Sep 10, 15:00 JST
    expect(
      screen.getByText(/Member distribution · Sep 10, 15:00 (JST|GMT\+9)/)
    ).toBeInTheDocument();
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

  it("shows an honest unavailable state when there are neither members nor a stored line", () => {
    render(
      <EnsembleDistribution {...baseProps} data={withoutMembers} status="success" error={null} />
    );

    expect(
      screen.getByText(/returned no raw member values and no stored distribution/)
    ).toBeInTheDocument();
    // No fabricated histogram is rendered.
    expect(screen.queryByRole("img", { name: /Histogram/ })).not.toBeInTheDocument();
  });

  it("draws the stored distribution when the members have been reclaimed", () => {
    // The case a fully converted store reaches: the members are gone, the container is the answer,
    // and the line has to be drawable without any member values -- including the bin edges, which
    // the payload carries because a container holds no values to derive them from.
    const storedOnly: EnsembleStatisticsData = {
      ...withoutMembers,
      histogram_stored: { edges: [10, 12, 14, 16, 18, 20], counts: [1, 1, 1, 1, 1] },
    };
    const { container } = render(
      <EnsembleDistribution {...baseProps} data={storedOnly} status="success" error={null} />
    );

    expect(screen.getByText(/Stored distribution · Sep 10, 06:00 UTC/)).toBeInTheDocument();
    expect(
      screen.getByRole("img", { name: /Stored distribution for Temperature/ })
    ).toBeInTheDocument();
    expect(
      screen.getByText(/members have been aggregated into a summary container/)
    ).toBeInTheDocument();
    // Bars are drawn, one per bin the payload declared, and no member rug is fabricated.
    expect(container.querySelectorAll(".recharts-rectangle").length).toBe(5);
    expect(container.querySelectorAll(".recharts-scatter-symbol").length).toBe(0);
    // The summary cells fall back to the stored statistics rather than to "—".
    expect(screen.getByText("17.5")).toBeInTheDocument();
  });

  it("keeps the member line when the stored line is delivered beside it", () => {
    // The comparison state, unchanged: both grids agree, so both lines are drawn and the bars stay
    // the member sample's.
    const both: EnsembleStatisticsData = {
      ...withMembersAndPdf,
      histogram_members: { edges: [10, 15, 20, 25, 30], counts: [1, 1, 1, 1, 1] },
      histogram_stored: { edges: [10, 15, 20, 25, 30], counts: [1, 0, 3, 1, 0] },
    };
    const { container } = render(
      <EnsembleDistribution {...baseProps} data={both} status="success" error={null} />
    );

    expect(screen.getByText(/Member distribution · Sep 10, 06:00 UTC/)).toBeInTheDocument();
    expect(
      screen.getByRole("img", { name: /Histogram and PDF of 5 ensemble members/ })
    ).toBeInTheDocument();
    // Two lines: the dashed stored line and the PDF. The bars are still the member sample's own
    // Sturges bins, not the delivered grid's -- the member count is what says which source the
    // chart is about, and a stored payload beside members does not change that.
    expect(container.querySelectorAll(".recharts-line-curve").length).toBe(2);
    expect(container.querySelectorAll(".recharts-rectangle").length).toBe(
      histogramBins(withMembersAndPdf.members!).length
    );
  });

  it("draws the stored curve on a converted store, where no member curve exists", () => {
    // The curve is what a reader actually looks at, so a converted store has to be able to draw
    // one. The API sends the same KDE over the stored distribution on the canonical grid, so the
    // chart falls back to it exactly as it falls back to the stored histogram.
    const storedOnlyWithCurve: EnsembleStatisticsData = {
      ...withoutMembers,
      histogram_stored: { edges: [10, 12, 14, 16, 18, 20], counts: [1, 1, 1, 1, 1] },
      pdf_stored: {
        x: [10.0, 12.5, 15.0, 17.5, 20.0],
        density: [0.01, 0.12, 0.2, 0.12, 0.01],
      },
    };
    const { container } = render(
      <EnsembleDistribution
        {...baseProps}
        data={storedOnlyWithCurve}
        status="success"
        error={null}
      />
    );

    expect(screen.getByText(/Stored distribution · Sep 10, 06:00 UTC/)).toBeInTheDocument();
    // The stored curve is drawn -- one line, since no stored histogram line is delivered without
    // members to compare it against.
    expect(container.querySelectorAll(".recharts-line-curve").length).toBe(1);
  });

  it("draws no curve when the encoding cannot state the distribution's shape", () => {
    // The API omits `pdf_stored` for a distribution with a point mass wider than a member, because
    // a smooth curve over that smear would overstate its spread several-fold. The chart draws the
    // bars and no curve, which is the absence rather than a wrong shape.
    const storedOnlyNoCurve: EnsembleStatisticsData = {
      ...withoutMembers,
      histogram_stored: { edges: [10, 12, 14, 16, 18, 20], counts: [1, 1, 1, 1, 1] },
    };
    const { container } = render(
      <EnsembleDistribution {...baseProps} data={storedOnlyNoCurve} status="success" error={null} />
    );

    expect(container.querySelectorAll(".recharts-rectangle").length).toBe(5);
    expect(container.querySelectorAll(".recharts-line-curve").length).toBe(0);
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
    expect(
      screen.getByText(/No ensemble distribution available for Sep 10, 06:00 UTC/)
    ).toBeInTheDocument();
    expect(screen.queryByText(/\+6h/)).not.toBeInTheDocument();
  });
});
