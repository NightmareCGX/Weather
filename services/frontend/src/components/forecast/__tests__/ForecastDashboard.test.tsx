import { render, screen } from "@testing-library/react";

import { ForecastDashboard } from "@/components/forecast/ForecastDashboard";
import { usePointForecast } from "@/hooks/usePointForecast";
import { useEnsemble } from "@/hooks/useEnsemble";
import { useEnsembleDistribution } from "@/hooks/useEnsembleDistribution";
import { useVariablesCatalog } from "@/hooks/useVariablesCatalog";
import { useForecastSelection } from "@/context/forecast-selection";
import type { PointForecast, SelectedLocation } from "@/lib/api/types";

jest.mock("../../../hooks/usePointForecast");
jest.mock("../../../hooks/useEnsemble");
jest.mock("../../../hooks/useEnsembleDistribution");
jest.mock("../../../hooks/useVariablesCatalog");
jest.mock("../../../context/forecast-selection");

// The chart components are covered by their own tests; stub them here so the
// dashboard test focuses on container behavior.
jest.mock("../../charts/Meteogram", () => ({
  Meteogram: ({ variableCode }: { variableCode: string }) => (
    <div data-testid="meteogram">{variableCode}</div>
  ),
}));
jest.mock("../../charts/EnsembleChart", () => ({
  EnsembleChart: () => <div data-testid="ensemble-chart" />,
}));
jest.mock("../../charts/EnsembleDistribution", () => ({
  EnsembleDistribution: () => <div data-testid="ensemble-distribution" />,
}));

const mockUsePointForecast = usePointForecast as jest.MockedFunction<typeof usePointForecast>;
const mockUseEnsemble = useEnsemble as jest.MockedFunction<typeof useEnsemble>;
const mockUseEnsembleDistribution = useEnsembleDistribution as jest.MockedFunction<
  typeof useEnsembleDistribution
>;
const mockUseVariablesCatalog = useVariablesCatalog as jest.MockedFunction<
  typeof useVariablesCatalog
>;
const mockUseForecastSelection = useForecastSelection as jest.MockedFunction<
  typeof useForecastSelection
>;

const location: SelectedLocation = {
  name: "Aspen",
  object: "city",
  latitude: 38.19,
  longitude: -106.82,
  elevation_m: null,
  region: "Colorado",
  country: "USA",
  id: "city_aspen",
  resolvedVia: "city",
};

const forecast: PointForecast = {
  location: {
    latitude: 38.19,
    longitude: -106.82,
    elevation_m: null,
    resolved_via: "city",
  },
  generated_at: "2026-07-21T00:00:00Z",
  model: "gfs",
  forecasts: [
    { lead_time_hours: 0, valid_time: "2026-07-21T00:00:00Z", temperature_2m: 10 },
    { lead_time_hours: 6, valid_time: "2026-07-21T06:00:00Z", temperature_2m: 13 },
  ],
};

function mockSelectionContext(overrides: Partial<ReturnType<typeof useForecastSelection>> = {}) {
  mockUseForecastSelection.mockReturnValue({
    availability: null,
    status: "success",
    error: null,
    selection: {
      model: "gfs",
      variable: "temperature_2m",
      initialTime: "2026-08-13T00:00:00Z",
      leadTimeHours: 6,
    },
    validTime: "2026-08-13T06:00:00Z",
    options: {
      models: [{ id: "gfs", name: "Global Forecast System", is_ensemble: false, variables: [] }],
      model: { id: "gfs", name: "Global Forecast System", is_ensemble: false, variables: [] },
      variables: [],
      initialTimes: [],
      variable: null,
      initialTime: null,
      leadTimes: [6],
    },
    setModel: jest.fn(),
    setVariable: jest.fn(),
    setInitialTime: jest.fn(),
    setLeadTimeHours: jest.fn(),
    retry: jest.fn(),
    ...overrides,
  });
}

beforeEach(() => {
  mockUseVariablesCatalog.mockReturnValue({
    variables: [
      { id: "temperature_2m", object: "variable", name: "2-Meter Temperature", unit: "°C" },
    ],
    status: "success",
    error: null,
  });
  // The distribution hook is separate from the statistics timeline.
  mockUseEnsembleDistribution.mockReturnValue({
    data: null,
    status: "idle",
    error: null,
  });
  mockSelectionContext();
});

function mockEnsembleModelSelected() {
  mockSelectionContext({
    selection: {
      model: "gefs",
      variable: "temperature_2m",
      initialTime: "2026-08-13T00:00:00Z",
      leadTimeHours: 6,
    },
    options: {
      models: [{ id: "gfs", name: "Global Forecast System", is_ensemble: false, variables: [] }],
      model: {
        id: "gefs",
        name: "Global Ensemble Forecast System",
        is_ensemble: true,
        variables: [],
      },
      variables: [],
      initialTimes: [],
      variable: null,
      initialTime: null,
      leadTimes: [6],
    },
  });
}

describe("ForecastDashboard", () => {
  it("shows loading states for both panels while fetching", () => {
    mockEnsembleModelSelected();
    mockUsePointForecast.mockReturnValue({ forecast: null, status: "loading", error: null });
    mockUseEnsemble.mockReturnValue({
      byLead: new Map(),
      status: "loading",
      error: null,
      model: "gefs",
    });

    render(<ForecastDashboard location={location} />);

    expect(screen.getByText(/Loading forecast…/)).toBeInTheDocument();
    expect(screen.getByText(/Loading ensemble statistics…/)).toBeInTheDocument();
    expect(screen.getByText("Aspen")).toBeInTheDocument();
  });

  it("renders the location summary and meteograms on success", () => {
    mockEnsembleModelSelected();
    mockUsePointForecast.mockReturnValue({ forecast, status: "success", error: null });
    mockUseEnsemble.mockReturnValue({
      byLead: new Map(),
      status: "error",
      error: "Ensemble unavailable.",
      model: "gefs",
    });

    render(<ForecastDashboard location={location} />);

    expect(screen.getByText("Aspen")).toBeInTheDocument();
    expect(screen.getByTestId("meteogram")).toHaveTextContent("temperature_2m");
  });

  it("degrades independently: an ensemble failure does not destroy the forecast", () => {
    mockEnsembleModelSelected();
    mockUsePointForecast.mockReturnValue({ forecast, status: "success", error: null });
    mockUseEnsemble.mockReturnValue({
      byLead: new Map(),
      status: "error",
      error: "Ensemble unavailable.",
      model: "gefs",
    });

    render(<ForecastDashboard location={location} />);

    // Core forecast still renders.
    expect(screen.getByTestId("meteogram")).toBeInTheDocument();
    // Ensemble panel shows its scoped error.
    expect(screen.getByText("Ensemble unavailable.")).toBeInTheDocument();
  });

  it("renders the ensemble chart and distribution independently", () => {
    mockUsePointForecast.mockReturnValue({ forecast, status: "success", error: null });
    mockUseEnsemble.mockReturnValue({
      byLead: new Map([
        [
          0,
          {
            model: "gfs",
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
      ]),
      status: "success",
      error: null,
      model: "gfs",
    });
    mockUseEnsembleDistribution.mockReturnValue({
      data: {
        model: "gfs",
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
        members: [8, 9, 10, 11, 12],
      },
      status: "success",
      error: null,
    });
    // The selected model is an ensemble model for this test.
    mockSelectionContext({
      selection: {
        model: "gefs",
        variable: "temperature_2m",
        initialTime: "2026-08-13T00:00:00Z",
        leadTimeHours: 6,
      },
      options: {
        models: [{ id: "gfs", name: "Global Forecast System", is_ensemble: false, variables: [] }],
        model: {
          id: "gefs",
          name: "Global Ensemble Forecast System",
          is_ensemble: true,
          variables: [],
        },
        variables: [],
        initialTimes: [],
        variable: null,
        initialTime: null,
        leadTimes: [6],
      },
    });

    render(<ForecastDashboard location={location} />);

    // The statistics fan chart renders from the statistics timeline…
    expect(screen.getByTestId("ensemble-chart")).toBeInTheDocument();
    // …and the distribution view renders from the focused members request.
    expect(screen.getByTestId("ensemble-distribution")).toBeInTheDocument();
  });

  it("shows the point forecast error state inline", () => {
    mockUsePointForecast.mockReturnValue({
      forecast: null,
      status: "error",
      error: "No forecast data covers this location.",
    });
    mockUseEnsemble.mockReturnValue({
      byLead: new Map(),
      status: "idle",
      error: null,
      model: "gfs",
    });

    render(<ForecastDashboard location={location} />);

    expect(screen.getByRole("alert")).toHaveTextContent("No forecast data covers this location.");
  });

  it("shows an ensemble empty state for a deterministic selected model", () => {
    // gfs is deterministic -> no ensemble data.
    mockUsePointForecast.mockReturnValue({ forecast, status: "success", error: null });

    render(<ForecastDashboard location={location} />);

    expect(
      screen.getByText("No ensemble data available for the selected forecast.")
    ).toBeInTheDocument();
    // No ensemble requests were made (the hooks are called with model=null and
    // stay idle).
    expect(mockUseEnsemble).toHaveBeenCalledWith(location, expect.any(Array), "temperature_2m", {
      model: null,
    });
  });
});
