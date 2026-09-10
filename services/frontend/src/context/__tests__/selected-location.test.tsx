import { act, renderHook } from "@testing-library/react";
import type { ReactNode } from "react";

import {
  SelectedLocationProvider,
  useSelectedLocation,
  useSelectedLocationTimezone,
} from "@/context/selected-location";
import type { SelectedLocation } from "@/lib/api/types";

const mockLocationA: SelectedLocation = {
  name: "Denver",
  object: "city",
  id: "city_denver",
  resolvedVia: "city",
  latitude: 39.7392,
  longitude: -104.9903,
  elevation_m: 1609,
  region: "Colorado",
  country: "United States",
};

const mockLocationB: SelectedLocation = {
  name: "Boulder",
  object: "city",
  id: "city_boulder",
  resolvedVia: "city",
  latitude: 40.015,
  longitude: -105.2705,
  elevation_m: 1655,
  region: "Colorado",
  country: "United States",
};

const wrapper = ({ children }: { children: ReactNode }) => (
  <SelectedLocationProvider>{children}</SelectedLocationProvider>
);

describe("SelectedLocationContext selectionGeneration guard", () => {
  it("initializes with null location and generation 0", () => {
    const { result } = renderHook(() => useSelectedLocation(), { wrapper });
    expect(result.current.selectedLocation).toBeNull();
    expect(result.current.selectionGeneration).toBe(0);
  });

  it("increments selectionGeneration on synchronous selectLocation", () => {
    const { result } = renderHook(() => useSelectedLocation(), { wrapper });

    act(() => {
      result.current.selectLocation(mockLocationA);
    });

    expect(result.current.selectedLocation).toEqual(mockLocationA);
    expect(result.current.selectionGeneration).toBe(1);

    act(() => {
      result.current.selectLocation(mockLocationB);
    });

    expect(result.current.selectedLocation).toEqual(mockLocationB);
    expect(result.current.selectionGeneration).toBe(2);
  });

  it("increments selectionGeneration on clearSelection", () => {
    const { result } = renderHook(() => useSelectedLocation(), { wrapper });

    act(() => {
      result.current.selectLocation(mockLocationA);
    });
    expect(result.current.selectionGeneration).toBe(1);

    act(() => {
      result.current.clearSelection();
    });

    expect(result.current.selectedLocation).toBeNull();
    expect(result.current.selectionGeneration).toBe(2);
  });

  it("commits an async selection when no newer action intervened", () => {
    const { result } = renderHook(() => useSelectedLocation(), { wrapper });

    let token = 0;
    act(() => {
      token = result.current.beginAsyncSelection();
    });
    expect(token).toBe(1);
    expect(result.current.selectionGeneration).toBe(1);

    let committed = false;
    act(() => {
      committed = result.current.commitAsyncSelection(mockLocationA, token);
    });

    expect(committed).toBe(true);
    expect(result.current.selectedLocation).toEqual(mockLocationA);
  });

  it("rejects an async selection if a map click or search selection intervened", () => {
    const { result } = renderHook(() => useSelectedLocation(), { wrapper });

    // Step 1: Start async selection (e.g. Locate Me)
    let token = 0;
    act(() => {
      token = result.current.beginAsyncSelection();
    });
    expect(token).toBe(1);

    // Step 2: User explicitly clicks the map or selects a search result while async is pending
    act(() => {
      result.current.selectLocation(mockLocationB);
    });
    expect(result.current.selectionGeneration).toBe(2);
    expect(result.current.selectedLocation).toEqual(mockLocationB);

    // Step 3: Old async response arrives late
    let committed = true;
    act(() => {
      committed = result.current.commitAsyncSelection(mockLocationA, token);
    });

    // Stale async selection must be rejected and must NOT overwrite the explicit selection
    expect(committed).toBe(false);
    expect(result.current.selectedLocation).toEqual(mockLocationB);
  });

  it("rejects an async selection if clearSelection intervened", () => {
    const { result } = renderHook(() => useSelectedLocation(), { wrapper });

    let token = 0;
    act(() => {
      token = result.current.beginAsyncSelection();
    });

    act(() => {
      result.current.clearSelection();
    });

    let committed = true;
    act(() => {
      committed = result.current.commitAsyncSelection(mockLocationA, token);
    });

    expect(committed).toBe(false);
    expect(result.current.selectedLocation).toBeNull();
  });
});

describe("SelectedLocationContext selectedTimezone derivation", () => {
  const mockTokyo: SelectedLocation = {
    name: "Tokyo",
    object: "city",
    id: "city_tokyo",
    resolvedVia: "city",
    latitude: 35.6762,
    longitude: 139.6503,
    elevation_m: 40,
    region: "Tokyo",
    country: "Japan",
  };

  it("initializes with null selectedTimezone", () => {
    const { result } = renderHook(() => useSelectedLocation(), { wrapper });
    expect(result.current.selectedTimezone).toBeNull();
  });

  it("derives IANA timezone when a location is selected", () => {
    const { result } = renderHook(() => useSelectedLocation(), { wrapper });

    act(() => {
      result.current.selectLocation(mockLocationA);
    });

    expect(result.current.selectedTimezone).toBe("America/Denver");
  });

  it("updates timezone when selected location changes", () => {
    const { result } = renderHook(() => useSelectedLocation(), { wrapper });

    act(() => {
      result.current.selectLocation(mockLocationA);
    });
    expect(result.current.selectedTimezone).toBe("America/Denver");

    act(() => {
      result.current.selectLocation(mockTokyo);
    });
    expect(result.current.selectedTimezone).toBe("Asia/Tokyo");
  });

  it("resets selectedTimezone to null when selection is cleared", () => {
    const { result } = renderHook(() => useSelectedLocation(), { wrapper });

    act(() => {
      result.current.selectLocation(mockLocationA);
    });
    expect(result.current.selectedTimezone).toBe("America/Denver");

    act(() => {
      result.current.clearSelection();
    });
    expect(result.current.selectedTimezone).toBeNull();
  });

  it("useSelectedLocationTimezone hook returns timezone and safely falls back to null", () => {
    // Within provider
    const { result } = renderHook(
      () => {
        const loc = useSelectedLocation();
        const tz = useSelectedLocationTimezone();
        return { loc, tz };
      },
      { wrapper }
    );

    expect(result.current.tz).toBeNull();

    act(() => {
      result.current.loc.selectLocation(mockLocationA);
    });
    expect(result.current.tz).toBe("America/Denver");

    // Outside provider
    const outside = renderHook(() => useSelectedLocationTimezone());
    expect(outside.result.current).toBeNull();
  });
});
