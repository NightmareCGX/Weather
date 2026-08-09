import { fireEvent, render, screen } from "@testing-library/react";

import { LocationSearch } from "@/components/search/LocationSearch";
import { useSearch } from "@/hooks/useSearch";
import type { SearchResult } from "@/lib/api/types";

jest.mock("../../../hooks/useSearch");

const mockUseSearch = useSearch as jest.MockedFunction<typeof useSearch>;

const results: SearchResult[] = [
  {
    id: "city_aspen",
    object: "city",
    name: "Aspen",
    region: "Colorado",
    country: "USA",
    elevation_m: null,
    latitude: 38.19,
    longitude: -106.82,
  },
  {
    id: "resort_aspen_mountain",
    object: "ski_resort",
    name: "Aspen Mountain",
    region: "Colorado",
    country: "USA",
    elevation_m: 3417,
    latitude: 38.19,
    longitude: -106.82,
  },
];

function mockResultsForQuery(query: string) {
  mockUseSearch.mockReturnValue({
    results: query.startsWith("A") ? results : [],
    status: "success",
    error: null,
  });
}

beforeEach(() => {
  mockUseSearch.mockReset();
  mockResultsForQuery("");
});

function typeQuery(input: HTMLElement, query: string) {
  fireEvent.change(input, { target: { value: query } });
  fireEvent.focus(input);
}

function optionByText(text: string): HTMLElement {
  const option = screen.getAllByRole("option").find((el) => el.textContent?.includes(text));
  if (option === undefined) {
    throw new Error(`No option containing "${text}"`);
  }
  return option;
}

describe("LocationSearch", () => {
  it("renders an accessible combobox input", () => {
    render(<LocationSearch onSelect={jest.fn()} />);
    const input = screen.getByRole("combobox", {
      name: "Search for a city, ski resort, or station",
    });
    expect(input).toBeInTheDocument();
    expect(input).toHaveAttribute("aria-expanded", "false");
  });

  it("shows results and selects a city on click", () => {
    mockResultsForQuery("Aspen");
    const onSelect = jest.fn();
    render(<LocationSearch onSelect={onSelect} />);
    const input = screen.getByRole("combobox");

    typeQuery(input, "Aspen");
    expect(screen.getByRole("listbox")).toBeInTheDocument();
    expect(
      screen.getAllByRole("option").some((option) => option.textContent?.includes("Aspen"))
    ).toBe(true);

    fireEvent.mouseDown(optionByText("Aspen"));
    expect(onSelect).toHaveBeenCalledTimes(1);
    expect(onSelect.mock.calls[0][0]).toMatchObject({
      object: "city",
      id: "city_aspen",
      resolvedVia: "city",
      latitude: 38.19,
      longitude: -106.82,
    });
  });

  it("supports keyboard navigation and Enter to select", () => {
    mockResultsForQuery("Aspen");
    const onSelect = jest.fn();
    render(<LocationSearch onSelect={onSelect} />);
    const input = screen.getByRole("combobox");

    typeQuery(input, "Aspen");
    fireEvent.keyDown(input, { key: "ArrowDown" });
    const options = screen.getAllByRole("option");
    expect(options[0]).toHaveAttribute("aria-selected", "true");

    fireEvent.keyDown(input, { key: "Enter" });
    expect(onSelect).toHaveBeenCalledTimes(1);
    expect(onSelect.mock.calls[0][0]).toMatchObject({ name: "Aspen" });
  });

  it("moves the highlight with ArrowUp and wraps around", () => {
    mockResultsForQuery("Aspen");
    const onSelect = jest.fn();
    render(<LocationSearch onSelect={onSelect} />);
    const input = screen.getByRole("combobox");

    typeQuery(input, "Aspen");
    fireEvent.keyDown(input, { key: "ArrowDown" });
    fireEvent.keyDown(input, { key: "ArrowDown" });
    const options = screen.getAllByRole("option");
    expect(options[1]).toHaveAttribute("aria-selected", "true");

    fireEvent.keyDown(input, { key: "ArrowUp" });
    expect(options[0]).toHaveAttribute("aria-selected", "true");
  });

  it("closes with Escape", () => {
    mockResultsForQuery("Aspen");
    render(<LocationSearch onSelect={jest.fn()} />);
    const input = screen.getByRole("combobox");

    typeQuery(input, "Aspen");
    expect(screen.getByRole("listbox")).toBeInTheDocument();
    fireEvent.keyDown(input, { key: "Escape" });
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
  });

  it("shows the empty state when there are no results", () => {
    render(<LocationSearch onSelect={jest.fn()} />);
    const input = screen.getByRole("combobox");
    typeQuery(input, "zzzznomatch");

    expect(screen.getByText("No matching locations.")).toBeInTheDocument();
  });

  it("shows an inline error state on API failure", () => {
    mockUseSearch.mockReturnValue({
      results: [],
      status: "error",
      error: "Failed to search locations.",
    });
    render(<LocationSearch onSelect={jest.fn()} />);
    const input = screen.getByRole("combobox");
    typeQuery(input, "Aspen");

    expect(screen.getByRole("alert")).toHaveTextContent("Failed to search locations.");
  });

  it("shows a loading state while searching", () => {
    mockUseSearch.mockReturnValue({ results: [], status: "loading", error: null });
    render(<LocationSearch onSelect={jest.fn()} />);
    const input = screen.getByRole("combobox");
    typeQuery(input, "Aspen");

    expect(screen.getByText("Searching…")).toBeInTheDocument();
  });

  it("clears the query and closes after selection", () => {
    mockResultsForQuery("Aspen");
    render(<LocationSearch onSelect={jest.fn()} />);
    const input = screen.getByRole("combobox");
    typeQuery(input, "Aspen");
    fireEvent.mouseDown(optionByText("Aspen"));

    expect(input).toHaveValue("");
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
  });

  it("respects a disabled prop", () => {
    render(<LocationSearch onSelect={jest.fn()} disabled />);
    expect(screen.getByRole("combobox")).toBeDisabled();
  });
});
