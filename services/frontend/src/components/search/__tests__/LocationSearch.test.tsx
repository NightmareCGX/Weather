import { act, fireEvent, render, screen } from "@testing-library/react";

import { LocationSearch } from "@/components/search/LocationSearch";
import { useSearch } from "@/hooks/useSearch";
import { ApiError, resolvePlace } from "../../../lib/api/client";
import type { SearchResult } from "@/lib/api/types";

jest.mock("../../../hooks/useSearch");
jest.mock("../../../lib/api/client", () => {
  const actual = jest.requireActual("../../../lib/api/client");
  return {
    ...actual,
    resolvePlace: jest.fn(),
  };
});

const mockUseSearch = useSearch as jest.MockedFunction<typeof useSearch>;
const mockResolvePlace = resolvePlace as jest.MockedFunction<typeof resolvePlace>;

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
  {
    id: "place_aspen",
    object: "place",
    name: "Aspen, CO, USA",
    place_id: "ChIJ_aspen_place",
    region: "Colorado",
    country: "USA",
    elevation_m: null,
    latitude: null as unknown as number,
    longitude: null as unknown as number,
  },
  {
    id: "place_boulder",
    object: "place",
    name: "Boulder, CO, USA",
    place_id: "ChIJ_boulder_place",
    region: "Colorado",
    country: "USA",
    elevation_m: null,
    latitude: null as unknown as number,
    longitude: null as unknown as number,
  },
];

function mockResultsForQuery(query: string) {
  mockUseSearch.mockReturnValue({
    results: query.trim().length > 0 ? results : [],
    status: "success",
    error: null,
    sessionToken: "test-session-token",
  });
}

beforeEach(() => {
  mockUseSearch.mockReset();
  mockResolvePlace.mockReset();
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

  it("resolves a place suggestion and selects it with coordinates", async () => {
    mockResultsForQuery("Aspen");
    mockResolvePlace.mockResolvedValueOnce({
      id: "resolved_aspen",
      object: "place",
      name: "Aspen, CO, USA",
      place_id: "ChIJ_aspen_place",
      region: "Colorado",
      country: "USA",
      elevation_m: 2400,
      latitude: 39.1911,
      longitude: -106.8175,
    });

    const onSelect = jest.fn();
    render(<LocationSearch onSelect={onSelect} />);
    const input = screen.getByRole("combobox");

    typeQuery(input, "Aspen");
    fireEvent.mouseDown(optionByText("Aspen, CO, USA"));

    expect(mockResolvePlace).toHaveBeenCalledTimes(1);
    expect(mockResolvePlace).toHaveBeenCalledWith({
      placeId: "ChIJ_aspen_place",
      sessionToken: "test-session-token",
      signal: expect.any(AbortSignal),
    });

    await act(async () => {
      await Promise.resolve();
    });

    expect(onSelect).toHaveBeenCalledTimes(1);
    expect(onSelect).toHaveBeenLastCalledWith(
      expect.objectContaining({
        object: "coordinates",
        id: "ChIJ_aspen_place",
        resolvedVia: "coordinates",
        latitude: 39.1911,
        longitude: -106.8175,
        name: "Aspen, CO, USA",
      })
    );
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
    mockResultsForQuery("zzzznomatch");
    mockUseSearch.mockReturnValue({
      results: [],
      status: "success",
      error: null,
      sessionToken: "test-session-token",
    });
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
      sessionToken: "test-session-token",
    });
    render(<LocationSearch onSelect={jest.fn()} />);
    const input = screen.getByRole("combobox");
    typeQuery(input, "Aspen");

    expect(screen.getByRole("alert")).toHaveTextContent("Failed to search locations.");
  });

  it("shows a loading state while searching", () => {
    mockUseSearch.mockReturnValue({
      results: [],
      status: "loading",
      error: null,
      sessionToken: "test-session-token",
    });
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

  describe("concurrency and async ownership", () => {
    it("A. Slow Place A → synchronous City B: Place A resolution does not commit over City B", async () => {
      mockResultsForQuery("Aspen");
      let resolvePlaceA!: (value: SearchResult) => void;
      mockResolvePlace.mockImplementationOnce(
        () => new Promise<SearchResult>((resolve) => (resolvePlaceA = resolve))
      );

      const onSelect = jest.fn();
      render(<LocationSearch onSelect={onSelect} />);
      const input = screen.getByRole("combobox");

      // 1. User starts Place A resolution
      typeQuery(input, "Aspen");
      fireEvent.mouseDown(optionByText("Aspen, CO, USA"));
      expect(mockResolvePlace).toHaveBeenCalledTimes(1);
      expect(onSelect).not.toHaveBeenCalled();

      // 2. User immediately selects City B
      typeQuery(input, "Aspen");
      fireEvent.mouseDown(optionByText("Aspen"));

      // 3. City B commits immediately
      expect(onSelect).toHaveBeenCalledTimes(1);
      expect(onSelect).toHaveBeenLastCalledWith(
        expect.objectContaining({ id: "city_aspen", object: "city" })
      );

      // 4. Stale Place A resolves late
      await act(async () => {
        resolvePlaceA({
          id: "resolved_aspen",
          object: "place",
          name: "Aspen Resolved Canonical",
          place_id: "ChIJ_aspen_place",
          region: "Colorado",
          country: "USA",
          elevation_m: null,
          latitude: 39.1911,
          longitude: -106.8175,
        });
      });

      // 5. Place A must not commit; City B remains authoritative
      expect(onSelect).toHaveBeenCalledTimes(1);
      expect(onSelect).toHaveBeenLastCalledWith(
        expect.objectContaining({ id: "city_aspen", object: "city" })
      );
    });

    it("B & C. Slow Place A → fast Place B (stale success): late A success does not overwrite current B", async () => {
      mockResultsForQuery("Aspen");
      let resolvePlaceA!: (value: SearchResult) => void;
      let resolvePlaceB!: (value: SearchResult) => void;
      mockResolvePlace
        .mockImplementationOnce(
          () => new Promise<SearchResult>((resolve) => (resolvePlaceA = resolve))
        )
        .mockImplementationOnce(
          () => new Promise<SearchResult>((resolve) => (resolvePlaceB = resolve))
        );

      const onSelect = jest.fn();
      render(<LocationSearch onSelect={onSelect} />);
      const input = screen.getByRole("combobox");

      // 1. Select Place A
      typeQuery(input, "Aspen");
      fireEvent.mouseDown(optionByText("Aspen, CO, USA"));

      // 2. Select Place B
      typeQuery(input, "Boulder");
      fireEvent.mouseDown(optionByText("Boulder, CO, USA"));

      // 3. Place B resolves first
      await act(async () => {
        resolvePlaceB({
          id: "resolved_boulder",
          object: "place",
          name: "Boulder Canonical",
          place_id: "ChIJ_boulder_place",
          region: "Colorado",
          country: "USA",
          elevation_m: null,
          latitude: 40.015,
          longitude: -105.2705,
        });
      });

      expect(onSelect).toHaveBeenCalledTimes(1);
      expect(onSelect).toHaveBeenLastCalledWith(
        expect.objectContaining({ name: "Boulder Canonical", latitude: 40.015 })
      );

      // 4. Stale Place A resolves second
      await act(async () => {
        resolvePlaceA({
          id: "resolved_aspen",
          object: "place",
          name: "Aspen Canonical",
          place_id: "ChIJ_aspen_place",
          region: "Colorado",
          country: "USA",
          elevation_m: null,
          latitude: 39.1911,
          longitude: -106.8175,
        });
      });

      // 5. Stale Place A must be discarded; Boulder remains authoritative
      expect(onSelect).toHaveBeenCalledTimes(1);
      expect(onSelect).toHaveBeenLastCalledWith(
        expect.objectContaining({ name: "Boulder Canonical", latitude: 40.015 })
      );
    });

    it("D. Stale failure: late Place A rejection does not overwrite current selection with fallback", async () => {
      mockResultsForQuery("Aspen");
      let rejectPlaceA!: (error: unknown) => void;
      mockResolvePlace.mockImplementationOnce(
        () => new Promise<SearchResult>((_, reject) => (rejectPlaceA = reject))
      );

      const onSelect = jest.fn();
      render(<LocationSearch onSelect={onSelect} />);
      const input = screen.getByRole("combobox");

      // 1. Select Place A
      typeQuery(input, "Aspen");
      fireEvent.mouseDown(optionByText("Aspen, CO, USA"));

      // 2. Select City B
      typeQuery(input, "Aspen");
      fireEvent.mouseDown(optionByText("Aspen"));
      expect(onSelect).toHaveBeenCalledTimes(1);

      // 3. Stale Place A fails with ApiError
      await act(async () => {
        rejectPlaceA(
          new ApiError("Place resolution failed", 502, "server_error", "server_error", null, null)
        );
      });

      // 4. Unresolved fallback from Place A must NOT be committed
      expect(onSelect).toHaveBeenCalledTimes(1);
      expect(onSelect).toHaveBeenLastCalledWith(
        expect.objectContaining({ id: "city_aspen", object: "city" })
      );
    });

    it("F. Component unmount: pending place resolution does not commit state after unmount", async () => {
      mockResultsForQuery("Aspen");
      let resolvePlaceA!: (value: SearchResult) => void;
      mockResolvePlace.mockImplementationOnce(
        () => new Promise<SearchResult>((resolve) => (resolvePlaceA = resolve))
      );

      const onSelect = jest.fn();
      const { unmount } = render(<LocationSearch onSelect={onSelect} />);
      const input = screen.getByRole("combobox");

      // 1. Select Place A
      typeQuery(input, "Aspen");
      fireEvent.mouseDown(optionByText("Aspen, CO, USA"));

      // 2. Unmount component
      unmount();

      // 3. Place A resolves after unmount
      await act(async () => {
        resolvePlaceA({
          id: "resolved_aspen",
          object: "place",
          name: "Aspen Canonical",
          place_id: "ChIJ_aspen_place",
          region: "Colorado",
          country: "USA",
          elevation_m: null,
          latitude: 39.1911,
          longitude: -106.8175,
        });
      });

      // 4. onSelect must not be called after unmount
      expect(onSelect).not.toHaveBeenCalled();
    });
  });
});
