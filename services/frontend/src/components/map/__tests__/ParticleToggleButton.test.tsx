import { fireEvent, render, screen } from "@testing-library/react";
import { ParticleToggleButton } from "../ParticleToggleButton";

describe("ParticleToggleButton", () => {
  it("renders enabled state with correct attributes and title", () => {
    const onToggle = jest.fn();
    render(<ParticleToggleButton enabled={true} onToggle={onToggle} />);

    const button = screen.getByRole("button", { name: "Toggle wind particle animation" });
    expect(button).toBeInTheDocument();
    expect(button).toHaveAttribute("aria-pressed", "true");
    expect(button).toHaveAttribute("title", "Pause wind particle animation");

    fireEvent.click(button);
    expect(onToggle).toHaveBeenCalledTimes(1);
  });

  it("renders disabled/paused state with correct attributes and title", () => {
    const onToggle = jest.fn();
    render(<ParticleToggleButton enabled={false} onToggle={onToggle} />);

    const button = screen.getByRole("button", { name: "Toggle wind particle animation" });
    expect(button).toBeInTheDocument();
    expect(button).toHaveAttribute("aria-pressed", "false");
    expect(button).toHaveAttribute("title", "Enable wind particle animation");

    fireEvent.click(button);
    expect(onToggle).toHaveBeenCalledTimes(1);
  });

  it("honors disabled prop", () => {
    const onToggle = jest.fn();
    render(<ParticleToggleButton enabled={true} onToggle={onToggle} disabled={true} />);

    const button = screen.getByRole("button", { name: "Toggle wind particle animation" });
    expect(button).toBeDisabled();

    fireEvent.click(button);
    expect(onToggle).not.toHaveBeenCalled();
  });
});
