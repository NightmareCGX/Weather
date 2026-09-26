import { fireEvent, render, screen, within } from "@testing-library/react";

import { Header } from "../Header";
import { useForecastSelection } from "../../../context/forecast-selection";
import type { ForecastAvailability, ModelAvailability } from "../../../lib/api/types";

jest.mock("../../../context/forecast-selection", () => ({
  useForecastSelection: jest.fn(),
}));

const mockUseForecastSelection = useForecastSelection as jest.MockedFunction<
  typeof useForecastSelection
>;

// A fixed "now" so staleness classification is deterministic.
const NOW = Date.parse("2026-09-17T12:00:00Z");

function cycle(hoursAgo: number): string {
  return new Date(NOW - hoursAgo * 3_600_000).toISOString();
}

function model(id: string, status: string, isEnsemble = false, hoursAgo = 0): ModelAvailability {
  return {
    id,
    name: id === "gfs" ? "Global Forecast System" : "Global Ensemble Forecast System",
    is_ensemble: isEnsemble,
    center_id: "noaa",
    center_name: "NOAA",
    cycle_cadence_hours: 6,
    variables: [
      {
        id: "temperature_2m",
        name: "2m Temperature",
        unit: "°C",
        initial_times: [{ value: cycle(hoursAgo), lead_time_hours: [0], status }],
        valid_times: [
          {
            valid_time: cycle(hoursAgo),
            source_cycle: cycle(hoursAgo),
            lead_time_hours: 0,
            servable: true,
            available_members: 1,
            expected_members: 1,
            coverage_ratio: 1,
          },
        ],
      },
    ],
  };
}

function availability(models: ModelAvailability[]): ForecastAvailability {
  return { models, serving_start_valid_time: cycle(0), generated_at: cycle(0) };
}

function renderHeader(overrides: { availability?: ForecastAvailability | null; status?: string }) {
  mockUseForecastSelection.mockReturnValue({
    availability: overrides.availability ?? null,
    status: (overrides.status ?? "success") as never,
    error: null,
    selection: null,
    validTime: null,
    options: {} as never,
    setModel: jest.fn(),
    setVariable: jest.fn(),
    retry: jest.fn(),
  });

  return render(<Header />);
}

beforeEach(() => {
  jest.useFakeTimers({ now: NOW });
});

afterEach(() => {
  jest.useRealTimers();
});

describe("Header status badge", () => {
  it("renders one badge per issuing center with a ready count", () => {
    renderHeader({
      availability: availability([model("gfs", "ready"), model("gefs", "ready", true)]),
    });

    expect(screen.getByRole("button", { name: /NOAA/ })).toBeInTheDocument();
    expect(screen.getByText("2/2")).toBeInTheDocument();
  });

  it("shows a syncing placeholder before availability loads", () => {
    renderHeader({ status: "loading" });
    expect(screen.getByText(/Syncing forecast feeds/)).toBeInTheDocument();
  });

  it("shows an offline placeholder when availability fails", () => {
    renderHeader({ status: "error" });
    expect(screen.getByText(/Forecast feeds offline/)).toBeInTheDocument();
  });

  it("reveals the center's models on hover, with per-model status", () => {
    renderHeader({
      availability: availability([model("gfs", "ready"), model("gefs", "processing", true)]),
    });

    // Collapsed: model rows are not present yet.
    expect(screen.queryByText("Global Forecast System")).not.toBeInTheDocument();

    fireEvent.mouseEnter(screen.getByRole("button", { name: /NOAA/ }));

    const panel = screen.getByText(/NOAA feeds/).closest("div") as HTMLElement;
    expect(within(panel).getByText("Global Forecast System")).toBeInTheDocument();
    expect(within(panel).getByText(/Ready/)).toBeInTheDocument();
    expect(within(panel).getByText(/Syncing/)).toBeInTheDocument();
    // Ensemble models are tagged; deterministic ones are not.
    expect(within(panel).getByTestId("ensemble-flag")).toBeInTheDocument();
  });

  it("marks the trigger as an expandable control and keeps it read-only", () => {
    renderHeader({ availability: availability([model("gfs", "ready")]) });

    const trigger = screen.getByRole("button", { name: /NOAA/ });
    expect(trigger).toHaveAttribute("aria-expanded", "false");
    expect(trigger).toHaveAttribute("aria-haspopup", "true");

    fireEvent.click(trigger);
    expect(trigger).toHaveAttribute("aria-expanded", "true");

    // Nothing in the badge is a selector: model choice lives in the layer controls.
    expect(screen.queryByRole("combobox")).not.toBeInTheDocument();
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
  });

  it("closes the panel on Escape", () => {
    renderHeader({ availability: availability([model("gfs", "ready")]) });

    const trigger = screen.getByRole("button", { name: /NOAA/ });
    fireEvent.click(trigger);
    expect(screen.getByText(/NOAA feeds/)).toBeInTheDocument();

    fireEvent.keyDown(document, { key: "Escape" });
    expect(screen.queryByText(/NOAA feeds/)).not.toBeInTheDocument();
  });

  it("flags a stalled feed as Stale rather than Ready", () => {
    renderHeader({
      availability: availability([model("gfs", "ready", false, 24)]),
    });

    fireEvent.mouseEnter(screen.getByRole("button", { name: /NOAA/ }));

    expect(screen.getByText(/Stale/)).toBeInTheDocument();
    expect(screen.getByText("0/1")).toBeInTheDocument();
  });

  it("stays open when a tap synthesizes hover immediately before the click", () => {
    // Touch browsers fire mouseenter and then click for one tap; an unguarded
    // toggle would open the panel and instantly re-close it.
    renderHeader({ availability: availability([model("gfs", "ready")]) });

    const trigger = screen.getByRole("button", { name: /NOAA/ });
    fireEvent.mouseEnter(trigger);
    fireEvent.click(trigger);

    expect(trigger).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText(/NOAA feeds/)).toBeInTheDocument();
  });

  it("closes the panel when the pointer goes down outside it", () => {
    renderHeader({ availability: availability([model("gfs", "ready")]) });

    fireEvent.click(screen.getByRole("button", { name: /NOAA/ }));
    expect(screen.getByText(/NOAA feeds/)).toBeInTheDocument();

    fireEvent.pointerDown(document.body);
    expect(screen.queryByText(/NOAA feeds/)).not.toBeInTheDocument();
  });
});
