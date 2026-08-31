import { render, screen, fireEvent } from "@testing-library/react";
import { EnsemblePhaseSupport } from "../EnsemblePhaseSupport";

describe("EnsemblePhaseSupport", () => {
  const defaultPhaseSupport = {
    dry: 0.1,
    rain: 0.52,
    snow: 0.26,
    freezing_rain: 0.08,
    ice_pellets: 0.03,
    unknown: 0.01,
  };

  const defaultTransitionFrequency = {
    rain_to_snow: 0.27,
    persistent_rain: 0.25,
    snow_to_rain: 0.1,
  };

  it("renders 100% stacked bar with all non-zero physical phase segments", () => {
    render(
      <EnsemblePhaseSupport
        phaseSupport={defaultPhaseSupport}
        transitionFrequency={defaultTransitionFrequency}
        selectedLead={12}
        memberCount={30}
      />
    );

    expect(
      screen.getByRole("img", {
        name: /Ensemble phase support composition at lead 12h/i,
      })
    ).toBeInTheDocument();

    // Check that segments exist for all 6 phases
    expect(screen.getByTestId("phase-segment-dry")).toBeInTheDocument();
    expect(screen.getByTestId("phase-segment-rain")).toBeInTheDocument();
    expect(screen.getByTestId("phase-segment-snow")).toBeInTheDocument();
    expect(screen.getByTestId("phase-segment-freezing_rain")).toBeInTheDocument();
    expect(screen.getByTestId("phase-segment-ice_pellets")).toBeInTheDocument();
    expect(screen.getByTestId("phase-segment-unknown")).toBeInTheDocument();

    // Invariant: no mixed segment
    expect(screen.queryByTestId("phase-segment-mixed")).not.toBeInTheDocument();
  });

  it("displays honest breakdown percentages summing to 100%", () => {
    render(
      <EnsemblePhaseSupport phaseSupport={defaultPhaseSupport} selectedLead={6} memberCount={30} />
    );

    expect(screen.getByText("Rain")).toBeInTheDocument();
    expect(screen.getAllByText("52%").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("Snow")).toBeInTheDocument();
    expect(screen.getAllByText("26%").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("Dry")).toBeInTheDocument();
    expect(screen.getByText("10%")).toBeInTheDocument();
    expect(screen.getByText("Freezing Rain")).toBeInTheDocument();
    expect(screen.getByText("8%")).toBeInTheDocument();
    expect(screen.getByText("Ice Pellets")).toBeInTheDocument();
    expect(screen.getByText("3%")).toBeInTheDocument();
    expect(screen.getByText("Unknown")).toBeInTheDocument();
    expect(screen.getByText("1%")).toBeInTheDocument();
  });

  it("renders secondary transition frequencies", () => {
    render(
      <EnsemblePhaseSupport
        phaseSupport={defaultPhaseSupport}
        transitionFrequency={defaultTransitionFrequency}
        selectedLead={12}
        memberCount={30}
      />
    );

    expect(screen.getByText(/Member Phase Transitions/i)).toBeInTheDocument();
    expect(screen.getByText("Rain → Snow")).toBeInTheDocument();
    expect(screen.getByText("· 27%")).toBeInTheDocument();
  });

  it("supports hover interaction on phase segments and badges", () => {
    render(
      <EnsemblePhaseSupport
        phaseSupport={defaultPhaseSupport}
        transitionFrequency={defaultTransitionFrequency}
        selectedLead={12}
        memberCount={30}
      />
    );

    const rainSegment = screen.getByTestId("phase-segment-rain");
    fireEvent.mouseEnter(rainSegment);
    expect(rainSegment).toHaveClass("brightness-110");

    fireEvent.mouseLeave(rainSegment);
    expect(rainSegment).not.toHaveClass("brightness-110");
  });
});
