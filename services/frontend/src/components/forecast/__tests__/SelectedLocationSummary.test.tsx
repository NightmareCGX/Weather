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

  it("renders compact header with sticky classes containing title and Close button while metadata remains in details section", () => {
    const handleClose = jest.fn();
    const { container } = render(
      <SelectedLocationSummary location={mockLocation} onClose={handleClose} />
    );

    const header = container.querySelector(".sticky");
    expect(header).toBeInTheDocument();
    expect(header).toHaveClass("top-0", "z-10", "bg-white");

    // Close button and title are inside sticky header
    expect(header).toHaveTextContent("Aspen");
    expect(header?.querySelector("button")).toBeInTheDocument();

    // Coordinates metadata is in the details section, not inside the sticky header
    const detailsSection = screen.getByRole("region", { name: "Selected location" });
    expect(detailsSection).toBeInTheDocument();
    expect(detailsSection).not.toHaveClass("sticky");
    expect(detailsSection).toHaveTextContent("Latitude");
    expect(detailsSection).toHaveTextContent("Longitude");
    expect(detailsSection).toHaveTextContent("Elevation");
  });

  it("renders loading state when dynamic elevation request is pending", () => {
    const dynamicCoordLocation: SelectedLocation = {
      id: null,
      object: "coordinates",
      name: "39.1911, -106.8175",
      region: null,
      country: null,
      elevation_m: null,
      latitude: 39.1911,
      longitude: -106.8175,
      resolvedVia: "coordinates",
    };

    render(
      <SelectedLocationSummary
        location={dynamicCoordLocation}
        elevationStatus="loading"
        elevation_m={null}
      />
    );

    expect(screen.getByText("loading…")).toBeInTheDocument();
  });

  it("renders unavailable when elevation resolution fails or is null", () => {
    const dynamicCoordLocation: SelectedLocation = {
      id: null,
      object: "coordinates",
      name: "39.1911, -106.8175",
      region: null,
      country: null,
      elevation_m: null,
      latitude: 39.1911,
      longitude: -106.8175,
      resolvedVia: "coordinates",
    };

    render(
      <SelectedLocationSummary
        location={dynamicCoordLocation}
        elevationStatus="error"
        elevation_m={null}
      />
    );

    expect(screen.getByText("unavailable")).toBeInTheDocument();
  });

  it("renders dynamically resolved elevation when provided via props", () => {
    const dynamicCoordLocation: SelectedLocation = {
      id: null,
      object: "coordinates",
      name: "39.1911, -106.8175",
      region: null,
      country: null,
      elevation_m: null,
      latitude: 39.1911,
      longitude: -106.8175,
      resolvedVia: "coordinates",
    };

    render(
      <SelectedLocationSummary
        location={dynamicCoordLocation}
        elevationStatus="success"
        elevation_m={2404.2}
      />
    );

    expect(screen.getByText("2,404 m")).toBeInTheDocument();
  });
});
