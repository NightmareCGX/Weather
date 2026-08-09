import { render, screen } from "@testing-library/react";

import { Legend } from "@/components/map/Legend";
import type { SpatialLayer } from "@/lib/api/types";

const layer: SpatialLayer = {
  tile_url_template: "/v1/maps/gfs/temperature_2m/surface/{z}/{x}/{y}.png?lead_time_hours=12",
  min_zoom: 0,
  max_zoom: 9,
  lead_time_hours: 12,
  legend: {
    unit: "°C",
    stops: [
      [-40, "#0000ff"],
      [0, "#00ff00"],
      [40, "#ff0000"],
    ],
  },
};

describe("Legend", () => {
  it("renders nothing when there is no layer", () => {
    const { container } = render(<Legend layer={null} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders the unit and a gradient bar from the legend stops", () => {
    render(<Legend layer={layer} />);

    expect(screen.getByText("°C")).toBeInTheDocument();
    expect(screen.getByTestId("legend-gradient")).toHaveStyle({
      backgroundImage: expect.stringContaining("linear-gradient"),
    });
  });
});
