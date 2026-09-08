import { fireEvent, render, screen } from "@testing-library/react";

import { SelectedLocationSummary } from "@/components/forecast/SelectedLocationSummary";
import type { SelectedLocation } from "@/lib/api/types";

const mockLocation: SelectedLocation = {
  id: "city_aspen",
  object: "city",
  name: "Aspen",
  region: "Colorado",
  country: "USA",
  elevation_m: 2400,
  latitude: 38.19,
  longitude: -106.82,
  resolvedVia: "city",
};

describe("SelectedLocationSummary", () => {
  it("renders location details without Close button when onClose is omitted", () => {
    render(<SelectedLocationSummary location={mockLocation} />);

    expect(screen.getByText("Aspen")).toBeInTheDocument();
    expect(screen.getByText("Colorado, USA")).toBeInTheDocument();
    expect(screen.getByText("2,400 m")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /close forecast panel/i })).not.toBeInTheDocument();
  });

  it("renders Close button with accessible name when onClose is provided", () => {
    const handleClose = jest.fn();
    render(<SelectedLocationSummary location={mockLocation} onClose={handleClose} />);

    const closeBtn = screen.getByRole("button", { name: "Close forecast panel" });
    expect(closeBtn).toBeInTheDocument();

    const svg = closeBtn.querySelector("svg");
    expect(svg).toHaveAttribute("aria-hidden", "true");
    expect(svg).toHaveAttribute("focusable", "false");

    fireEvent.click(closeBtn);
    expect(handleClose).toHaveBeenCalledTimes(1);
  });
});
