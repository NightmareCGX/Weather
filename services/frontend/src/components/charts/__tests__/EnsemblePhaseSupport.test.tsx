import React from "react";
import { render, screen } from "@testing-library/react";

import { EnsemblePhaseSupport, EnsemblePhaseSupportTooltip } from "../EnsemblePhaseSupport";
import type { EnsembleStatisticsData } from "@/lib/api/types";

let lastChartData: any[] = [];
let lastXAxisProps: any = null;

jest.mock("recharts", () => {
  const actual = jest.requireActual("recharts");
  return {
    ...actual,
    ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
      <div data-testid="responsive" style={{ width: 640, height: 224 }}>
        {children}
      </div>
    ),
    BarChart: ({ data, children, ...props }: any) => {
      lastChartData = data;
      if (Array.isArray(children)) {
        for (const child of children) {
          if (child && child.props && child.props.dataKey === "valid_time") {
            lastXAxisProps = child.props;
          }
        }
      }
      return (
        <actual.BarChart data={data} {...props}>
          {children}
        </actual.BarChart>
      );
    },
    XAxis: (props: any) => {
      lastXAxisProps = props;
      return <actual.XAxis {...props} />;
    },
    Tooltip: (props: any) => <actual.Tooltip {...props} />,
  };
});

describe("EnsemblePhaseSupport (Time-Varying)", () => {
  const multiLeadData = new Map<number, EnsembleStatisticsData>([
    [
      0,
      {
        model: "gefs",
        lead_time_hours: 0,
        valid_time: "2026-09-10T00:00:00Z",
        member_count: 30,
        valid_member_count: 30,
        statistics: {
          mean: null,
          median: null,
          spread: null,
          p10: null,
          p25: null,
          p50: null,
          p75: null,
          p90: null,
        },
        phase_support: null, // Lead 0 analysis time has no accumulation
      },
    ],
    [
      3,
      {
        model: "gefs",
        lead_time_hours: 3,
        valid_time: "2026-09-10T03:00:00Z",
        member_count: 30,
        valid_member_count: 30,
        statistics: {
          mean: null,
          median: null,
          spread: null,
          p10: null,
          p25: null,
          p50: null,
          p75: null,
          p90: null,
        },
        phase_support: {
          dry: 0.1,
          rain: 0.52,
          snow: 0.26,
          freezing_rain: 0.08,
          ice_pellets: 0.03,
          unknown: 0.01,
        },
        transition_frequency: {
          rain_to_snow: 0.27,
          persistent_rain: 0.25,
        },
      },
    ],
    [
      6,
      {
        model: "gefs",
        lead_time_hours: 6,
        valid_time: "2026-09-10T06:00:00Z",
        member_count: 30,
        valid_member_count: 30,
        statistics: {
          mean: null,
          median: null,
          spread: null,
          p10: null,
          p25: null,
          p50: null,
          p75: null,
          p90: null,
        },
        phase_support: {
          dry: 0.05,
          rain: 0.15,
          snow: 0.75,
          freezing_rain: 0.05,
          ice_pellets: 0.0,
          unknown: 0.0,
        },
      },
    ],
  ]);

  const validTimes = new Map<number, string>([
    [0, "2026-09-10T00:00:00Z"],
    [3, "2026-09-10T03:00:00Z"],
    [6, "2026-09-10T06:00:00Z"],
  ]);

  it("renders time-varying 100% stacked bar chart with phase taxonomy badges", () => {
    render(<EnsemblePhaseSupport byLead={multiLeadData} validTimesByLead={validTimes} />);

    expect(
      screen.getByRole("img", {
        name: /Ensemble phase support over time/i,
      })
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Ensemble Phase Support — time-varying support \(0–100%\)/)
    ).toBeInTheDocument();

    // Check that taxonomy badges exist for all 6 physical phases
    expect(screen.getByTestId("phase-badge-dry")).toBeInTheDocument();
    expect(screen.getByTestId("phase-badge-rain")).toBeInTheDocument();
    expect(screen.getByTestId("phase-badge-snow")).toBeInTheDocument();
    expect(screen.getByTestId("phase-badge-freezing_rain")).toBeInTheDocument();
    expect(screen.getByTestId("phase-badge-ice_pellets")).toBeInTheDocument();
    expect(screen.getByTestId("phase-badge-unknown")).toBeInTheDocument();

    // Invariant: no Mixed category in GEFS physical taxonomy
    expect(screen.queryByTestId("phase-badge-mixed")).not.toBeInTheDocument();

    // All valid times are retained in chart data
    expect(lastChartData).toHaveLength(3);
    expect(lastChartData.map((d) => d.valid_time)).toEqual([
      "2026-09-10T00:00:00Z",
      "2026-09-10T03:00:00Z",
      "2026-09-10T06:00:00Z",
    ]);

    // Lead 0 is gapped (has_data = false, metrics = null) without truncating later times
    expect(lastChartData[0].has_data).toBe(false);
    expect(lastChartData[0].rain).toBeNull();

    // Lead 3 & 6 are populated and total 100%
    expect(lastChartData[1].has_data).toBe(true);
    expect(lastChartData[1].rain).toBeCloseTo(52, 1);
    expect(lastChartData[1].snow).toBeCloseTo(26, 1);
    const totalLead3 =
      lastChartData[1].dry +
      lastChartData[1].rain +
      lastChartData[1].snow +
      lastChartData[1].freezing_rain +
      lastChartData[1].ice_pellets +
      lastChartData[1].unknown;
    expect(totalLead3).toBeCloseTo(100, 1);

    expect(lastChartData[2].has_data).toBe(true);
    expect(lastChartData[2].snow).toBeCloseTo(75, 1);
  });

  it("localizes X-axis valid times for America/Denver and Asia/Tokyo", () => {
    const { rerender } = render(
      <EnsemblePhaseSupport
        byLead={multiLeadData}
        validTimesByLead={validTimes}
        timezone="America/Denver"
      />
    );

    // 2026-09-10T00:00:00Z in Denver MDT is Sep 9, 18:00
    expect(lastXAxisProps.tickFormatter("2026-09-10T00:00:00Z")).toBe("Sep 9, 18:00");
    // 2026-09-10T06:00:00Z in Denver MDT is Sep 10, 00:00
    expect(lastXAxisProps.tickFormatter("2026-09-10T06:00:00Z")).toBe("Sep 10, 00:00");

    rerender(
      <EnsemblePhaseSupport
        byLead={multiLeadData}
        validTimesByLead={validTimes}
        timezone="Asia/Tokyo"
      />
    );

    // 2026-09-10T00:00:00Z in Tokyo JST is Sep 10, 09:00
    expect(lastXAxisProps.tickFormatter("2026-09-10T00:00:00Z")).toBe("Sep 10, 09:00");
    // 2026-09-10T06:00:00Z in Tokyo JST is Sep 10, 15:00
    expect(lastXAxisProps.tickFormatter("2026-09-10T06:00:00Z")).toBe("Sep 10, 15:00");
  });

  it("renders secondary member phase transitions", () => {
    render(<EnsemblePhaseSupport byLead={multiLeadData} validTimesByLead={validTimes} />);

    expect(screen.getByText(/Member Phase Transitions/i)).toBeInTheDocument();
    expect(screen.getByText("Rain → Snow")).toBeInTheDocument();
    expect(screen.getByText("· 27%")).toBeInTheDocument();
  });

  it("supports backward-compatible single snapshot rendering via phaseSupport prop", () => {
    const singlePhaseSupport = {
      dry: 0.1,
      rain: 0.52,
      snow: 0.26,
      freezing_rain: 0.08,
      ice_pellets: 0.03,
      unknown: 0.01,
    };

    render(
      <EnsemblePhaseSupport
        phaseSupport={singlePhaseSupport}
        validTime="2026-09-10T12:00:00Z"
        memberCount={30}
      />
    );

    expect(lastChartData).toHaveLength(1);
    expect(lastChartData[0].valid_time).toBe("2026-09-10T12:00:00Z");
    expect(lastChartData[0].rain).toBeCloseTo(52, 1);
  });

  describe("EnsemblePhaseSupportTooltip", () => {
    it("renders exact localized valid time, phase names, percentages, and member count", () => {
      const activePoint = {
        lead_time_hours: 3,
        valid_time: "2026-09-10T03:00:00Z",
        label: "Sep 10, 03:00",
        dry: 10,
        rain: 52,
        snow: 26,
        freezing_rain: 8,
        ice_pellets: 3,
        unknown: 1,
        valid_member_count: 30,
        member_count: 30,
        has_data: true,
      };

      render(
        <EnsemblePhaseSupportTooltip
          active={true}
          payload={[{ payload: activePoint } as any]}
          timezone="UTC"
        />
      );

      expect(screen.getByText("Sep 10, 03:00 UTC")).toBeInTheDocument();
      expect(screen.getByText("30 members")).toBeInTheDocument();
      expect(screen.getByText("Rain")).toBeInTheDocument();
      expect(screen.getByText("52%")).toBeInTheDocument();
      expect(screen.getByText("Snow")).toBeInTheDocument();
      expect(screen.getByText("26%")).toBeInTheDocument();
    });

    it("renders honest missing message when hovering a gapped timestamp lacking data", () => {
      const emptyPoint = {
        lead_time_hours: 0,
        valid_time: "2026-09-10T00:00:00Z",
        label: "Sep 10, 00:00",
        dry: null,
        rain: null,
        snow: null,
        freezing_rain: null,
        ice_pellets: null,
        unknown: null,
        valid_member_count: null,
        member_count: 30,
        has_data: false,
      };

      render(
        <EnsemblePhaseSupportTooltip
          active={true}
          payload={[{ payload: emptyPoint } as any]}
          timezone="UTC"
        />
      );

      expect(screen.getByText("Sep 10, 00:00 UTC")).toBeInTheDocument();
      expect(screen.getByText(/No ensemble phase data available/i)).toBeInTheDocument();
    });
  });
});
