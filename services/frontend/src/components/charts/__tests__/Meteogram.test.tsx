import { render, screen } from "@testing-library/react";

import { Meteogram, getWindArrowStride } from "@/components/charts/Meteogram";
import type { ForecastEntry } from "@/lib/api/types";

let lastChartData: any[] = [];
let lastTooltipProps: any = null;

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
    ComposedChart: ({ data, children, ...props }: any) => {
      lastChartData = data;
      if (Array.isArray(children)) {
        for (const child of children) {
          if (child && child.props && typeof child.props.labelFormatter === "function") {
            lastTooltipProps = child.props;
          }
        }
      }
      return (
        <actual.ComposedChart
          width={props.width || 640}
          height={props.height || 192}
          data={data}
          {...props}
        >
          {children}
        </actual.ComposedChart>
      );
    },
    Tooltip: (props: any) => {
      lastTooltipProps = props;
      return <actual.Tooltip {...props} />;
    },
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

  describe("timezone-aware X-axis and tooltip rendering", () => {
    it("renders in UTC when timezone is not provided", () => {
      render(
        <Meteogram
          forecasts={temperatureEntries}
          variableCode="temperature_2m"
          meta={{ name: "Temperature (2 m)", unit: "°C" }}
        />
      );

      expect(lastChartData[0].label).toBe("Jul 21, 00:00");
      expect(lastChartData[1].label).toBe("Jul 21, 06:00");

      const tooltipText = lastTooltipProps.labelFormatter(lastChartData[0].label, [
        { payload: lastChartData[0] },
      ]);
      expect(tooltipText).toBe("Jul 21, 00:00 UTC");
    });

    it("localizes X-axis and tooltip for America/Denver across local midnight", () => {
      // 2026-07-21T00:00:00Z in Denver (MDT, UTC-6) is Jul 20, 18:00 MDT
      // 2026-07-21T06:00:00Z in Denver is Jul 21, 00:00 MDT (crosses midnight)
      render(
        <Meteogram
          forecasts={temperatureEntries}
          variableCode="temperature_2m"
          meta={{ name: "Temperature (2 m)", unit: "°C" }}
          timezone="America/Denver"
        />
      );

      expect(lastChartData[0].label).toBe("Jul 20, 18:00");
      expect(lastChartData[1].label).toBe("Jul 21, 00:00");
      expect(lastChartData[2].label).toBe("Jul 21, 06:00");

      const tooltip0 = lastTooltipProps.labelFormatter(lastChartData[0].label, [
        { payload: lastChartData[0] },
      ]);
      expect(tooltip0).toMatch(/^Jul 20, 18:00 (MDT|GMT-6)$/);

      const tooltip1 = lastTooltipProps.labelFormatter(lastChartData[1].label, [
        { payload: lastChartData[1] },
      ]);
      expect(tooltip1).toMatch(/^Jul 21, 00:00 (MDT|GMT-6)$/);
    });

    it("disambiguates repeated local hour during DST fall-back in tooltip", () => {
      const fallbackEntries: ForecastEntry[] = [
        { lead_time_hours: 0, valid_time: "2026-11-01T07:00:00Z", temperature_2m: 5 },
        { lead_time_hours: 1, valid_time: "2026-11-01T08:00:00Z", temperature_2m: 4 },
      ];

      render(
        <Meteogram
          forecasts={fallbackEntries}
          variableCode="temperature_2m"
          meta={{ name: "Temperature (2 m)", unit: "°C" }}
          timezone="America/Denver"
        />
      );

      // Both points show 01:00 on X-axis
      expect(lastChartData[0].label).toBe("Nov 1, 01:00");
      expect(lastChartData[1].label).toBe("Nov 1, 01:00");

      // Tooltips distinguish daylight saving (MDT) vs standard time (MST)
      const tip0 = lastTooltipProps.labelFormatter(lastChartData[0].label, [
        { payload: lastChartData[0] },
      ]);
      const tip1 = lastTooltipProps.labelFormatter(lastChartData[1].label, [
        { payload: lastChartData[1] },
      ]);

      expect(tip0).toMatch(/^Nov 1, 01:00 (MDT|GMT-6)$/);
      expect(tip1).toMatch(/^Nov 1, 01:00 (MST|GMT-7)$/);
      expect(tip0).not.toBe(tip1);
    });

    it("formats non-whole-hour timezones correctly (Asia/Kathmandu)", () => {
      const entries: ForecastEntry[] = [
        { lead_time_hours: 0, valid_time: "2026-09-10T00:00:00Z", temperature_2m: 20 },
      ];

      render(
        <Meteogram
          forecasts={entries}
          variableCode="temperature_2m"
          meta={{ name: "Temperature (2 m)", unit: "°C" }}
          timezone="Asia/Kathmandu"
        />
      );

      expect(lastChartData[0].label).toBe("Sep 10, 05:45");
      const tip = lastTooltipProps.labelFormatter(lastChartData[0].label, [
        { payload: lastChartData[0] },
      ]);
      expect(tip).toMatch(/^Sep 10, 05:45 (GMT\+5:45|\+0545)$/);
    });

    it("formats Mixed precipitation with constituent phases in tooltip", () => {
      const precipEntries: ForecastEntry[] = [
        {
          lead_time_hours: 3,
          valid_time: "2026-09-10T03:00:00Z",
          precipitation_amount_3h: 0.31,
          precipitation_type: "mixed",
          precipitation_transition: "mixed_transition",
          crain: 1,
          csnow: 1,
          cfrzr: 0,
          cicep: 0,
        },
      ];

      render(
        <Meteogram
          forecasts={precipEntries}
          variableCode="precipitation_amount_3h"
          meta={{ name: "3-Hour Precipitation Amount", unit: "mm" }}
        />
      );

      const formattedTooltip = lastTooltipProps.formatter(0.31, "3-Hour Precipitation Amount", {
        payload: lastChartData[0],
      });

      expect(formattedTooltip).toEqual([
        "0.31 mm · Mixed (Rain + Snow)",
        "3-Hour Precipitation Amount",
      ]);
    });

    it("formats 1.83 mm Mixed precipitation with (Rain + Snow) from real API response shape", () => {
      // Faithful reproduction of the real /v1/points API payload that previously manifested:
      // "3-Hour Precipitation Amount: 1.83 mm · Mixed"
      const realApiForecastEntries: ForecastEntry[] = [
        {
          lead_time_hours: 6,
          valid_time: "2026-09-10T06:00:00Z",
          cycle_time: "2026-09-10T00:00:00Z",
          precipitation_amount_3h: 1.83,
          precipitation_type: "mixed",
          precipitation_transition: "mixed_transition",
          precipitation_start_type: "none",
          precipitation_end_type: "mixed",
          precipitation_evidence: "strongly_inferred",
          crain: 1.0,
          csnow: 1.0,
          cfrzr: 0.0,
          cicep: 0.0,
          temperature_2m: 1.2,
        },
      ];

      render(
        <Meteogram
          forecasts={realApiForecastEntries}
          variableCode="precipitation_amount_3h"
          meta={{ name: "3-Hour Precipitation Amount", unit: "mm" }}
        />
      );

      const formattedTooltip = lastTooltipProps.formatter(1.83, "3-Hour Precipitation Amount", {
        payload: lastChartData[0],
      });

      expect(formattedTooltip).toEqual([
        "1.83 mm · Mixed (Rain + Snow)",
        "3-Hour Precipitation Amount",
      ]);
    });

    it("formats GEFS ensemble-mean Mixed tooltip from real API response shape with fractional flags", () => {
      const gefsApiForecastEntries: ForecastEntry[] = [
        {
          lead_time_hours: 6,
          valid_time: "2026-09-10T06:00:00Z",
          cycle_time: "2026-09-10T00:00:00Z",
          precipitation_amount_3h: 1.83,
          precipitation_type: "mixed",
          precipitation_transition: "mixed_transition",
          precipitation_start_type: "none",
          precipitation_end_type: "mixed",
          precipitation_evidence: "strongly_inferred",
          crain: 0.65,
          csnow: 0.55,
          cfrzr: 0.05,
          cicep: 0.0,
          temperature_2m: 0.8,
        },
      ];

      render(
        <Meteogram
          forecasts={gefsApiForecastEntries}
          variableCode="precipitation_amount_3h"
          meta={{ name: "3-Hour Precipitation Amount", unit: "mm" }}
        />
      );

      const formattedTooltip = lastTooltipProps.formatter(1.83, "3-Hour Precipitation Amount", {
        payload: lastChartData[0],
      });

      expect(formattedTooltip).toEqual([
        "1.83 mm · Mixed (Rain + Snow)",
        "3-Hour Precipitation Amount",
      ]);
    });
  });

  describe("independent wind direction track (Shape 2 notch arrowheads)", () => {
    it("calculates adaptive stride correctly based on point count and width", () => {
      // 10 points on 600px width: deltaX = 66.6px -> stride = 1
      expect(getWindArrowStride(10, 600)).toBe(1);
      // Single point edge case
      expect(getWindArrowStride(1, 600)).toBe(1);
      // 81 points on 320px width: deltaX = 4px -> stride = 4
      expect(getWindArrowStride(81, 320)).toBe(4);
      // 81 points on 640px width: deltaX = 8px -> stride = 2
      expect(getWindArrowStride(81, 640)).toBe(2);
    });

    it("renders wind direction track with flow arrowheads and calm circles for wind_10m", () => {
      const windEntries: ForecastEntry[] = [
        {
          lead_time_hours: 0,
          valid_time: "2026-07-21T00:00:00Z",
          wind_10m: 0.5,
          wind_direction_10m: 0,
          wind_cardinal_10m: "CALM",
        },
        {
          lead_time_hours: 3,
          valid_time: "2026-07-21T03:00:00Z",
          wind_10m: 18.0,
          wind_direction_10m: 270, // West wind -> flow direction 90° (East)
          wind_cardinal_10m: "W",
        },
        {
          lead_time_hours: 6,
          valid_time: "2026-07-21T06:00:00Z",
          wind_10m: 12.0,
          wind_direction_10m: 0, // North wind -> flow direction 180° (South)
          wind_cardinal_10m: "N",
        },
      ];

      const { container } = render(
        <Meteogram
          forecasts={windEntries}
          variableCode="wind_10m"
          meta={{ name: "10-Meter Wind", unit: "km/h" }}
        />
      );

      // Verify the track container is rendered
      const track = container.querySelector(".recharts-wind-direction-track");
      expect(track).toBeInTheDocument();

      // Lead 0 (0.5 km/h, CALM) should render a calm circle
      const calmCircles = container.querySelectorAll("[data-testid='wind-calm-circle']");
      expect(calmCircles.length).toBe(1);

      // Leads 3 and 6 should render flow arrows
      const arrows = container.querySelectorAll("[data-testid='wind-flow-arrow']");
      expect(arrows.length).toBe(2);

      // Verify Shape 2 notch-arrowhead polygon points
      const polygons = container.querySelectorAll("polygon");
      expect(polygons.length).toBe(2);
      expect(polygons[0]).toHaveAttribute("points", "0,-4.5 3.5,3.5 0,1.5 -3.5,3.5");

      // Verify flow angle rotation:
      // Lead 3: dir 270° -> (270 + 180) % 360 = 90°
      expect(arrows[0].getAttribute("transform")).toContain("rotate(90)");
      // Lead 6: dir 0° -> (0 + 180) % 360 = 180°
      expect(arrows[1].getAttribute("transform")).toContain("rotate(180)");
    });

    it("does not render wind direction track for non-wind variables", () => {
      const { container } = render(
        <Meteogram
          forecasts={temperatureEntries}
          variableCode="temperature_2m"
          meta={{ name: "Temperature (2 m)", unit: "°C" }}
        />
      );

      const track = container.querySelector(".recharts-wind-direction-track");
      expect(track).not.toBeInTheDocument();
      expect(container.querySelectorAll("[data-testid='wind-flow-arrow']").length).toBe(0);
      expect(container.querySelectorAll("[data-testid='wind-calm-circle']").length).toBe(0);
    });

    it("applies adaptive decimation dots on dense series and highlights active tooltip point", () => {
      // Create 81 dense forecast entries (full 240h horizon)
      const denseEntries: ForecastEntry[] = Array.from({ length: 81 }, (_, i) => ({
        lead_time_hours: i * 3,
        valid_time: new Date(Date.UTC(2026, 6, 21, i * 3)).toISOString(),
        wind_10m: 15.0 + (i % 10),
        wind_direction_10m: (i * 45) % 360,
        wind_cardinal_10m: "SW",
      }));

      const { container } = render(
        <Meteogram
          forecasts={denseEntries}
          variableCode="wind_10m"
          meta={{ name: "10-Meter Wind", unit: "km/h" }}
        />
      );

      // On 640px mock width, stride for 81 points is 2:
      // Half the points show arrows, and the other half render subtle dots
      const arrows = container.querySelectorAll("[data-testid='wind-flow-arrow']");
      expect(arrows.length).toBeLessThan(81);
      expect(arrows.length).toBeGreaterThanOrEqual(40);

      const dots = container.querySelectorAll(".recharts-wind-direction-track circle[r='1']");
      expect(dots.length).toBeGreaterThan(0);
    });
  });
});
