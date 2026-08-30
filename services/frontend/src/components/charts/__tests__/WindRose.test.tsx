import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { WindRose } from "../WindRose";
import type { WindRose as WindRoseType } from "@/lib/api/types";

const MOCK_WIND_ROSE: WindRoseType = {
  member_count: 30,
  calm_count: 3,
  calm_percentage: 10.0,
  sectors: [
    { sector: "N", count: 6, probability: 0.2, bins: { light: 0.05, moderate: 0.1, strong: 0.05, gale: 0.0 } },
    { sector: "NE", count: 3, probability: 0.1, bins: { light: 0.1, moderate: 0.0, strong: 0.0, gale: 0.0 } },
    { sector: "E", count: 0, probability: 0.0, bins: { light: 0.0, moderate: 0.0, strong: 0.0, gale: 0.0 } },
    { sector: "SE", count: 0, probability: 0.0, bins: { light: 0.0, moderate: 0.0, strong: 0.0, gale: 0.0 } },
    { sector: "S", count: 3, probability: 0.1, bins: { light: 0.0, moderate: 0.1, strong: 0.0, gale: 0.0 } },
    { sector: "SW", count: 15, probability: 0.5, bins: { light: 0.1, moderate: 0.2, strong: 0.15, gale: 0.05 } },
    { sector: "W", count: 0, probability: 0.0, bins: { light: 0.0, moderate: 0.0, strong: 0.0, gale: 0.0 } },
    { sector: "NW", count: 0, probability: 0.0, bins: { light: 0.0, moderate: 0.0, strong: 0.0, gale: 0.0 } },
  ],
};

describe("WindRose", () => {
  it("renders calm percentage and cardinal labels", () => {
    render(<WindRose windRose={MOCK_WIND_ROSE} />);

    expect(screen.getByRole("img", { name: /ensemble wind rose chart/i })).toBeInTheDocument();
    expect(screen.getByText("CALM")).toBeInTheDocument();
    expect(screen.getByText("10%")).toBeInTheDocument();
    expect(screen.getByText("N")).toBeInTheDocument();
    expect(screen.getByText("SW")).toBeInTheDocument();
  });

  it("updates hover description on sector interaction", () => {
    render(<WindRose windRose={MOCK_WIND_ROSE} />);

    const swText = screen.getByText("SW");
    const parentGroup = swText.closest("g");
    expect(parentGroup).not.toBeNull();

    if (parentGroup) {
      fireEvent.mouseEnter(parentGroup);
      expect(screen.getByText(/50%/)).toBeInTheDocument();
      expect(screen.getByText(/15\/30 members/)).toBeInTheDocument();

      fireEvent.mouseLeave(parentGroup);
      expect(screen.getByText(/hover a sector/i)).toBeInTheDocument();
    }
  });

  it("renders speed bin legend", () => {
    render(<WindRose windRose={MOCK_WIND_ROSE} />);

    expect(screen.getByText(/Light/i)).toBeInTheDocument();
    expect(screen.getByText(/Moderate/i)).toBeInTheDocument();
    expect(screen.getByText(/Strong/i)).toBeInTheDocument();
    expect(screen.getByText(/Gale\+/i)).toBeInTheDocument();
  });
});
