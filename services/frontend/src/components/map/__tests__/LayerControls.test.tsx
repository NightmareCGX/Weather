import { fireEvent, render, screen } from "@testing-library/react";

import { LayerControls } from "@/components/map/LayerControls";
import type { Model } from "@/lib/api/types";

const models: Model[] = [
  {
    id: "gfs",
    object: "model",
    name: "Global Forecast System",
    center_id: "noaa",
    is_ensemble: false,
    resolution_km: 25,
  },
  {
    id: "gefs",
    object: "model",
    name: "Global Ensemble Forecast System",
    center_id: "noaa",
    is_ensemble: true,
    resolution_km: 25,
  },
];

describe("LayerControls", () => {
  it("renders model, variable, and lead time controls", () => {
    render(
      <LayerControls
        models={models}
        model="gfs"
        variable="temperature_2m"
        leadTimeHours={12}
        onModelChange={jest.fn()}
        onVariableChange={jest.fn()}
        onLeadTimeChange={jest.fn()}
      />
    );

    expect(screen.getByLabelText("Model")).toBeInTheDocument();
    expect(screen.getByLabelText("Variable")).toBeInTheDocument();
    expect(screen.getByLabelText("Lead time")).toBeInTheDocument();
    expect(screen.getByText("Global Forecast System")).toBeInTheDocument();
  });

  it("fires change callbacks with the new values", () => {
    const onModelChange = jest.fn();
    const onVariableChange = jest.fn();
    const onLeadTimeChange = jest.fn();

    render(
      <LayerControls
        models={models}
        model="gfs"
        variable="temperature_2m"
        leadTimeHours={12}
        onModelChange={onModelChange}
        onVariableChange={onVariableChange}
        onLeadTimeChange={onLeadTimeChange}
      />
    );

    fireEvent.change(screen.getByLabelText("Model"), { target: { value: "gefs" } });
    fireEvent.change(screen.getByLabelText("Variable"), {
      target: { value: "precipitation_rate" },
    });
    fireEvent.change(screen.getByLabelText("Lead time"), { target: { value: "24" } });

    expect(onModelChange).toHaveBeenCalledWith("gefs");
    expect(onVariableChange).toHaveBeenCalledWith("precipitation_rate");
    expect(onLeadTimeChange).toHaveBeenCalledWith(24);
  });
});
