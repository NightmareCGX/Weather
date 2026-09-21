import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { WindRose } from "../WindRose";
import type { WindRose as WindRoseType } from "@/lib/api/types";

const MOCK_WIND_ROSE: WindRoseType = {
  member_count: 30,
  calm_count: 3,
  calm_percentage: 10.0,
  sectors: [
    {
      sector: "N",
      count: 6,
      probability: 0.2,
      bins: { light: 0.05, moderate: 0.1, strong: 0.05, gale: 0.0 },
    },
    {
      sector: "NE",
      count: 3,
      probability: 0.1,
      bins: { light: 0.1, moderate: 0.0, strong: 0.0, gale: 0.0 },
    },
    {
      sector: "E",
      count: 0,
      probability: 0.0,
      bins: { light: 0.0, moderate: 0.0, strong: 0.0, gale: 0.0 },
    },
    {
      sector: "SE",
      count: 0,
      probability: 0.0,
      bins: { light: 0.0, moderate: 0.0, strong: 0.0, gale: 0.0 },
    },
    {
      sector: "S",
      count: 3,
      probability: 0.1,
      bins: { light: 0.0, moderate: 0.1, strong: 0.0, gale: 0.0 },
    },
    {
      sector: "SW",
      count: 15,
      probability: 0.5,
      bins: { light: 0.1, moderate: 0.2, strong: 0.15, gale: 0.05 },
    },
    {
      sector: "W",
      count: 0,
      probability: 0.0,
      bins: { light: 0.0, moderate: 0.0, strong: 0.0, gale: 0.0 },
    },
    {
      sector: "NW",
      count: 0,
      probability: 0.0,
      bins: { light: 0.0, moderate: 0.0, strong: 0.0, gale: 0.0 },
    },
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

describe("WindRose with a stored rose", () => {
  // The stored rose's buckets are quantile edges of the cycle's own member set, so the labels
  // come from the edges rather than from the fixed physical ranges the member path uses. Reading
  // the member table against stored keys would draw the right heights under the wrong legend.
  const storedRose = {
    calm_percentage: 8.0,
    calm_count: 2,
    member_count: 30,
    bucket_edges_mps: [1, 4, 7, 10, 13, 16, 19, 22, 26],
    bins: {
      bucket_0: 0.05,
      bucket_1: 0.1,
      bucket_2: 0.15,
      bucket_3: 0.2,
      bucket_4: 0.15,
      bucket_5: 0.1,
      bucket_6: 0.05,
      bucket_7: 0.02,
    },
    sectors: [
      {
        sector: "N",
        count: 4,
        probability: 0.2,
        bins: {
          bucket_0: 0.01,
          bucket_1: 0.02,
          bucket_2: 0.03,
          bucket_3: 0.04,
          bucket_4: 0.03,
          bucket_5: 0.03,
          bucket_6: 0.02,
          bucket_7: 0.02,
        },
      },
      {
        sector: "S",
        count: 6,
        probability: 0.3,
        bins: {
          bucket_0: 0.04,
          bucket_1: 0.08,
          bucket_2: 0.12,
          bucket_3: 0.16,
          bucket_4: 0.12,
          bucket_5: 0.07,
          bucket_6: 0.03,
          bucket_7: 0.0,
        },
      },
    ],
  };

  it("labels the buckets from the edges the payload carries", () => {
    render(<WindRose windRose={storedRose} />);

    // The first bucket spans edges[0]..edges[1] = 1..4 m/s = 4..14 km/h.
    expect(screen.getByText("4–14 km/h")).toBeInTheDocument();
    // The last is open-ended, because the outermost edge bounds it.
    expect(screen.getByText("≥79 km/h")).toBeInTheDocument();
    // The member path's fixed labels are gone: they would be wrong for a quantile grid.
    expect(screen.queryByText(/\(Light\)/)).not.toBeInTheDocument();
    expect(screen.getByText("8%")).toBeInTheDocument();
  });

  it("still uses the physical labels for a member-derived rose", () => {
    render(<WindRose windRose={MOCK_WIND_ROSE} />);
    expect(screen.getByText("1.8–20 km/h (Light)")).toBeInTheDocument();
    expect(screen.queryByText(/km\/h$/)).not.toBeInTheDocument();
  });
});
