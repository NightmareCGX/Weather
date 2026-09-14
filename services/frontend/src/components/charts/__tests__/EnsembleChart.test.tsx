import { render, screen } from "@testing-library/react";

import { EnsembleChart, EnsembleChartTooltip } from "@/components/charts/EnsembleChart";
import type { EnsembleStatisticsData } from "@/lib/api/types";

let lastChartData: any[] = [];
let lastXAxisProps: any = null;
let lastTooltipProps: any = null;

jest.mock("recharts", () => {
  const actual = jest.requireActual("recharts");
  return {
    ...actual,
    ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
      <div data-testid="responsive" style={{ width: 640, height: 192 }}>
        {children}
      </div>
    ),
    ComposedChart: ({ data, children, ...props }: any) => {
      lastChartData = data;
      if (Array.isArray(children)) {
        for (const child of children) {
          if (child && child.props && typeof child.props.labelFormatter === "function") {
            lastTooltipProps = child.props;
          }
          if (child && child.props && typeof child.props.tickFormatter === "function") {
            lastXAxisProps = child.props;
          }
        }
      }
      return (
        <actual.ComposedChart data={data} {...props}>
          {children}
        </actual.ComposedChart>
      );
    },
    XAxis: (props: any) => {
      lastXAxisProps = props;
      return <actual.XAxis {...props} />;
    },
    Tooltip: (props: any) => {
      lastTooltipProps = props;
      return <actual.Tooltip {...props} />;
    },
  };
});

const byLeadWithValidTimes = new Map<number, EnsembleStatisticsData>([
  [
    0,
    {
      model: "gefs",
      lead_time_hours: 0,
      valid_time: "2026-09-10T00:00:00Z",
      member_count: 5,
      statistics: { mean: 10, median: 10, spread: 2, p10: 7, p25: 9, p50: 10, p75: 11, p90: 13 },
    },
  ],
  [
    6,
    {
      model: "gefs",
      lead_time_hours: 6,
      valid_time: "2026-09-10T06:00:00Z",
      member_count: 5,
      statistics: { mean: 13, median: 13, spread: 2, p10: 10, p25: 12, p50: 13, p75: 14, p90: 16 },
    },
  ],
]);

describe("EnsembleChart", () => {
  it("renders a percentile-fan labeled chart for the variable with time axis", () => {
    render(<EnsembleChart byLead={byLeadWithValidTimes} variableLabel="Temperature (2 m)" />);

    expect(
      screen.getByRole("img", {
        name: "Temperature (2 m) ensemble percentile fan over time",
      })
    ).toBeInTheDocument();
    expect(screen.getByText(/Temperature \(2 m\) — percentile range/)).toBeInTheDocument();
    expect(screen.getByText(/P10–P90 band/)).toBeInTheDocument();
  });

  it("renders valid-time calendar timestamps in UTC on X-axis and tooltip when no timezone provided", () => {
    render(<EnsembleChart byLead={byLeadWithValidTimes} variableLabel="Temperature (2 m)" />);

    // X-axis uses valid_time dataKey and formats in UTC
    expect(lastXAxisProps.dataKey).toBe("valid_time");
    expect(lastXAxisProps.tickFormatter("2026-09-10T00:00:00Z")).toBe("Sep 10, 00:00");
    expect(lastXAxisProps.tickFormatter("2026-09-10T06:00:00Z")).toBe("Sep 10, 06:00");

    // Tooltip formats valid_time with UTC suffix
    expect(
      lastTooltipProps.labelFormatter("2026-09-10T00:00:00Z", [
        { payload: { valid_time: "2026-09-10T00:00:00Z" } },
      ])
    ).toBe("Sep 10, 00:00 UTC");

    // Regression check: no user-facing +0h or +6h axis labels
    expect(lastXAxisProps.tickFormatter("2026-09-10T00:00:00Z")).not.toContain("+0h");
    expect(lastXAxisProps.tickFormatter("2026-09-10T06:00:00Z")).not.toContain("+6h");
  });

  it("localizes X-axis and tooltip for America/Denver", () => {
    render(
      <EnsembleChart
        byLead={byLeadWithValidTimes}
        variableLabel="Temperature (2 m)"
        timezone="America/Denver"
      />
    );

    // 2026-09-10T00:00:00Z in MDT is Sep 9, 18:00
    // 2026-09-10T06:00:00Z in MDT is Sep 10, 00:00 (crosses local midnight)
    expect(lastXAxisProps.tickFormatter("2026-09-10T00:00:00Z")).toBe("Sep 9, 18:00");
    expect(lastXAxisProps.tickFormatter("2026-09-10T06:00:00Z")).toBe("Sep 10, 00:00");

    expect(
      lastTooltipProps.labelFormatter("2026-09-10T00:00:00Z", [
        { payload: { valid_time: "2026-09-10T00:00:00Z" } },
      ])
    ).toMatch(/^Sep 9, 18:00 (MDT|GMT-6)$/);
  });

  it("localizes X-axis and tooltip for Asia/Tokyo without changing canonical data", () => {
    render(
      <EnsembleChart
        byLead={byLeadWithValidTimes}
        variableLabel="Temperature (2 m)"
        timezone="Asia/Tokyo"
      />
    );

    // 2026-09-10T00:00:00Z in Tokyo is Sep 10, 09:00 JST
    // 2026-09-10T06:00:00Z in Tokyo is Sep 10, 15:00 JST
    expect(lastXAxisProps.tickFormatter("2026-09-10T00:00:00Z")).toBe("Sep 10, 09:00");
    expect(lastXAxisProps.tickFormatter("2026-09-10T06:00:00Z")).toBe("Sep 10, 15:00");

    expect(
      lastTooltipProps.labelFormatter("2026-09-10T00:00:00Z", [
        { payload: { valid_time: "2026-09-10T00:00:00Z" } },
      ])
    ).toMatch(/^Sep 10, 09:00 (JST|GMT\+9)$/);

    // Underlying chart data points retain canonical ISO valid_time
    expect(lastChartData[0].valid_time).toBe("2026-09-10T00:00:00Z");
    expect(lastChartData[1].valid_time).toBe("2026-09-10T06:00:00Z");
  });

  it("resolves valid_time via validTimesByLead mapping when byLead records omit valid_time", () => {
    const byLeadWithoutValidTimes = new Map<number, EnsembleStatisticsData>([
      [
        0,
        {
          model: "gefs",
          lead_time_hours: 0,
          member_count: 5,
          statistics: {
            mean: 10,
            median: 10,
            spread: 2,
            p10: 7,
            p25: 9,
            p50: 10,
            p75: 11,
            p90: 13,
          },
        },
      ],
    ]);

    const validTimesMap = new Map<number, string>([[0, "2026-09-10T00:00:00Z"]]);

    render(
      <EnsembleChart
        byLead={byLeadWithoutValidTimes}
        variableLabel="Temperature (2 m)"
        validTimesByLead={validTimesMap}
      />
    );

    expect(lastChartData[0].valid_time).toBe("2026-09-10T00:00:00Z");
    expect(lastXAxisProps.tickFormatter(lastChartData[0].valid_time)).toBe("Sep 10, 00:00");
  });

  describe("x-axis retention with NaN percentile points", () => {
    it("retains full x-domain when an interior valid time has NaN statistics", () => {
      const byLeadWithNaN = new Map<number, EnsembleStatisticsData>([
        [
          0,
          {
            model: "gefs",
            lead_time_hours: 0,
            valid_time: "2026-09-10T00:00:00Z",
            member_count: 30,
            statistics: {
              mean: 10,
              median: 10,
              spread: 2,
              p10: 7,
              p25: 9,
              p50: 10,
              p75: 11,
              p90: 13,
            },
          },
        ],
        [
          3,
          {
            model: "gefs",
            lead_time_hours: 3,
            valid_time: "2026-09-10T03:00:00Z",
            member_count: 30,
            statistics: {
              mean: 11,
              median: 11,
              spread: 2,
              p10: 8,
              p25: 10,
              p50: 11,
              p75: 12,
              p90: 14,
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
            // NaN stats (e.g. cloud ceiling when clear)
            statistics: {
              mean: Number.NaN,
              median: Number.NaN,
              spread: Number.NaN,
              p10: Number.NaN,
              p25: Number.NaN,
              p50: Number.NaN,
              p75: Number.NaN,
              p90: Number.NaN,
            },
          },
        ],
        [
          9,
          {
            model: "gefs",
            lead_time_hours: 9,
            valid_time: "2026-09-10T09:00:00Z",
            member_count: 30,
            statistics: {
              mean: 14,
              median: 14,
              spread: 2,
              p10: 11,
              p25: 13,
              p50: 14,
              p75: 15,
              p90: 17,
            },
          },
        ],
        [
          12,
          {
            model: "gefs",
            lead_time_hours: 12,
            valid_time: "2026-09-10T12:00:00Z",
            member_count: 30,
            statistics: {
              mean: 15,
              median: 15,
              spread: 2,
              p10: 12,
              p25: 14,
              p50: 15,
              p75: 16,
              p90: 18,
            },
          },
        ],
      ]);

      render(<EnsembleChart byLead={byLeadWithNaN} variableLabel="Cloud Ceiling Height" />);

      // All 5 timestamps must be retained in chart data (finite, finite, NaN, finite, finite)
      expect(lastChartData).toHaveLength(5);
      expect(lastChartData.map((d) => d.valid_time)).toEqual([
        "2026-09-10T00:00:00Z",
        "2026-09-10T03:00:00Z",
        "2026-09-10T06:00:00Z",
        "2026-09-10T09:00:00Z",
        "2026-09-10T12:00:00Z",
      ]);

      // Point 2 has null for all geometry
      expect(lastChartData[2].p10Base).toBeNull();
      expect(lastChartData[2].p90Height).toBeNull();
      expect(lastChartData[2].median).toBeNull();
      expect(lastChartData[2].mean).toBeNull();

      // Later points 3 and 4 remain finite and visible
      expect(lastChartData[3].median).toBe(14);
      expect(lastChartData[4].median).toBe(15);

      // Tooltip formatting for null / NaN renders "—" and not "0" or "NaN"
      expect(lastTooltipProps.formatter(null, "Median (P50)", {})).toEqual(["—", "Median (P50)"]);
      expect(lastTooltipProps.formatter(Number.NaN, "Mean", {})).toEqual(["—", "Mean"]);
      expect(lastTooltipProps.formatter(15, "Mean", {})).toEqual(["15", "Mean"]);
    });
  });

  describe("EnsembleChartTooltip", () => {
    const samplePayload = [
      {
        payload: {
          lead_time_hours: 6,
          valid_time: "2026-09-10T06:00:00Z",
          median: 22.78,
          mean: 22.66,
          p10: 21.9,
          p25: 22.4,
          p75: 22.82,
          p90: 22.77,
          p10Base: 21.9,
          p90Height: 0.87,
          p25Base: 22.4,
          p75Height: 0.42,
        },
      },
    ];

    it("renders actual percentile ranges, median, and mean with unit", () => {
      render(
        <EnsembleChartTooltip active={true} payload={samplePayload} unit="°C" timezone="UTC" />
      );

      // Shows valid time header
      expect(screen.getByText("Sep 10, 06:00 UTC")).toBeInTheDocument();

      // Shows central tendencies
      expect(screen.getByText("Median (P50)")).toBeInTheDocument();
      expect(screen.getByText("22.78 °C")).toBeInTheDocument();
      expect(screen.getByText("Mean")).toBeInTheDocument();
      expect(screen.getByText("22.66 °C")).toBeInTheDocument();

      // Shows actual percentile ranges, NOT height spans
      expect(screen.getByText("P25–P75")).toBeInTheDocument();
      expect(screen.getByText("22.4 – 22.82 °C")).toBeInTheDocument();
      expect(screen.getByText("P10–P90")).toBeInTheDocument();
      expect(screen.getByText("21.9 – 22.77 °C")).toBeInTheDocument();

      // Implementation details p10Base and p25Base must NOT be exposed
      expect(screen.queryByText(/p10Base/i)).not.toBeInTheDocument();
      expect(screen.queryByText(/p25Base/i)).not.toBeInTheDocument();
      expect(screen.queryByText(/p90Height/i)).not.toBeInTheDocument();
      expect(screen.queryByText(/p75Height/i)).not.toBeInTheDocument();
    });

    it("handles null and non-finite percentile values gracefully with dashes", () => {
      const payloadWithNulls = [
        {
          payload: {
            lead_time_hours: 6,
            valid_time: "2026-09-10T06:00:00Z",
            median: null,
            mean: Number.NaN,
            p10: null,
            p25: null,
            p75: null,
            p90: null,
          },
        },
      ];

      render(<EnsembleChartTooltip active={true} payload={payloadWithNulls} unit="°C" />);

      const dashes = screen.getAllByText("—");
      expect(dashes.length).toBeGreaterThanOrEqual(4);
    });

    it("returns null when inactive or payload is empty", () => {
      const { container: inactiveContainer } = render(
        <EnsembleChartTooltip active={false} payload={samplePayload} />
      );
      expect(inactiveContainer.firstChild).toBeNull();

      const { container: emptyContainer } = render(
        <EnsembleChartTooltip active={true} payload={[]} />
      );
      expect(emptyContainer.firstChild).toBeNull();
    });
  });
});
